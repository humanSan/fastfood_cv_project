"""
Mask Generator Module — wraps EfficientSAM3 to produce class-agnostic masks.

Replaces FastSAM in the original Semantic-Fast-SAM pipeline with EfficientSAM3's
lightweight EfficientViT backbone.  Uses an automatic grid of point prompts
(similar to SAM's "segment everything" mode) to generate masks.
"""

import sys
import os
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure the cloned efficientsam3 repo is importable.  The repo layout puts
# the `sam3` Python package at <repo_root>/sam3/.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "efficientsam3")
if os.path.isdir(_REPO_ROOT) and _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO_ROOT))

try:
    from sam3.model_builder import build_efficientsam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    _HAS_EFFICIENTSAM3 = True
except ImportError:
    _HAS_EFFICIENTSAM3 = False


def _generate_point_grid(h: int, w: int, n_per_side: int = 32):
    """Return an (N, 2) array of (x, y) point prompts on a uniform grid."""
    xs = np.linspace(0, w, n_per_side + 2)[1:-1]
    ys = np.linspace(0, h, n_per_side + 2)[1:-1]
    xv, yv = np.meshgrid(xs, ys)
    return np.stack([xv.ravel(), yv.ravel()], axis=1)  # (N, 2)


class EfficientSAM3MaskGenerator:
    """Generates class-agnostic masks using EfficientSAM3 with point-grid prompts."""

    def __init__(
        self,
        checkpoint_path: str = "weights/efficient_sam3_efficientvit_s.pt",
        backbone_type: str = "efficientvit",
        model_name: str = "b0",
        n_points_per_side: int = 32,
        device: str = "cuda",
    ):
        self.device = device
        self.n_points_per_side = n_points_per_side

        if not _HAS_EFFICIENTSAM3:
            raise ImportError(
                "EfficientSAM3 is not installed.  Run `python setup_repos.py` "
                "to clone the repo, then `pip install -e efficientsam3[stage1]`."
            )

        print(f"[MaskGenerator] Loading EfficientSAM3 ({backbone_type}/{model_name}) …")
        self.model = build_efficientsam3_image_model(
            checkpoint_path=checkpoint_path,
            backbone_type=backbone_type,
            model_name=model_name,
            enable_inst_interactivity=True,
        )
        self.processor = Sam3Processor(self.model)
        print("[MaskGenerator] Model loaded.")

    @torch.no_grad()
    def generate_masks(self, image):
        """
        Generate class-agnostic masks for the entire image.

        Args:
            image: PIL.Image (RGB) or numpy array (H, W, 3) uint8 BGR/RGB.

        Returns:
            masks:  list[np.ndarray]  — each element is a boolean mask of shape (H, W).
            scores: list[float]       — confidence score per mask.
        """
        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image

        w, h = pil_image.size

        # Encode the image once
        inference_state = self.processor.set_image(pil_image)

        # Build a grid of point prompts
        grid_points = _generate_point_grid(h, w, self.n_points_per_side)

        all_masks = []
        all_scores = []

        # Query the model with each grid point as a positive prompt
        for pt in grid_points:
            try:
                masks_np, scores_np, _ = self.model.predict_inst(
                    inference_state,
                    point_coords=[pt.tolist()],
                    point_labels=[1],
                )
                if masks_np is not None and len(masks_np) > 0:
                    # predict_inst may return multiple masks per prompt;
                    # keep only the highest-scoring one.
                    best_idx = int(np.argmax(scores_np))
                    mask = masks_np[best_idx].astype(bool)
                    score = float(scores_np[best_idx])

                    # Skip tiny / nearly-empty masks
                    if mask.sum() < 100:
                        continue

                    all_masks.append(mask)
                    all_scores.append(score)
            except Exception:
                continue  # skip failed points

        # Deduplicate overlapping masks via simple IoU-based NMS
        if all_masks:
            all_masks, all_scores = self._nms(all_masks, all_scores, iou_thresh=0.7)

        return all_masks, all_scores

    # ------------------------------------------------------------------
    @staticmethod
    def _nms(masks, scores, iou_thresh=0.7):
        """Non-maximum suppression over binary masks."""
        order = np.argsort(scores)[::-1]
        keep_masks, keep_scores = [], []

        kept_areas = []
        for idx in order:
            m = masks[idx]
            suppress = False
            for km in keep_masks:
                inter = np.logical_and(m, km).sum()
                union = np.logical_or(m, km).sum()
                if union > 0 and inter / union > iou_thresh:
                    suppress = True
                    break
            if not suppress:
                keep_masks.append(m)
                keep_scores.append(scores[idx])

        return keep_masks, keep_scores
