"""
Superpixel-Guided Mask Generator — replaces the uniform point grid with
SLIC superpixel centroids for content-aware prompting of EfficientSAM3.

Instead of blindly placing N×N points across the image (which wastes prompts
on large uniform regions and under-samples complex areas), this module:
  1. Runs SLIC to partition the image into ~n_superpixels content-aware regions.
  2. Computes the centroid of each superpixel.
  3. Feeds *only* those centroids as point prompts to EfficientSAM3.

This yields fewer but more intelligently-placed prompts, improving both
efficiency and coverage of visually complex areas.
"""

import sys
import os
import numpy as np
import torch
from PIL import Image as PILImage

from skimage.segmentation import slic
from skimage.measure import regionprops

# ---------------------------------------------------------------------------
# Ensure the cloned efficientsam3 repo is importable.
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


def _slic_centroids(image_np: np.ndarray, n_segments: int = 200,
                    compactness: float = 10.0, sigma: float = 1.0):
    """
    Run SLIC on an RGB image and return superpixel centroids.

    Args:
        image_np:    (H, W, 3) uint8 RGB array.
        n_segments:  Target number of superpixels.
        compactness: Balances colour vs spatial proximity (higher = more square).
        sigma:       Gaussian smoothing before SLIC.

    Returns:
        centroids:   (N, 2) array of (x, y) centroid coordinates.
        labels:      (H, W) integer superpixel label map.
    """
    labels = slic(image_np, n_segments=n_segments, compactness=compactness,
                  sigma=sigma, start_label=0, channel_axis=2)

    regions = regionprops(labels + 1)  # regionprops needs labels starting at 1
    centroids = []
    for region in regions:
        cy, cx = region.centroid  # regionprops returns (row, col) = (y, x)
        centroids.append([cx, cy])

    return np.array(centroids), labels


class SuperpixelMaskGenerator:
    """
    Generates class-agnostic masks using EfficientSAM3 with SLIC-centroid
    prompts instead of a uniform grid.
    """

    def __init__(
        self,
        checkpoint_path: str = "weights/efficient_sam3_efficientvit_s.pt",
        backbone_type: str = "efficientvit",
        model_name: str = "b0",
        n_superpixels: int = 200,
        compactness: float = 10.0,
        sigma: float = 1.0,
        device: str = "cuda",
    ):
        self.device = device
        self.n_superpixels = n_superpixels
        self.compactness = compactness
        self.sigma = sigma

        if not _HAS_EFFICIENTSAM3:
            raise ImportError(
                "EfficientSAM3 is not installed.  Run `python setup_repos.py` "
                "to clone the repo, then `pip install -e efficientsam3`."
            )

        print(f"[SuperpixelMaskGen] Loading EfficientSAM3 ({backbone_type}/{model_name}) …")
        self.model = build_efficientsam3_image_model(
            checkpoint_path=checkpoint_path,
            backbone_type=backbone_type,
            model_name=model_name,
            enable_inst_interactivity=True,
        )
        self.processor = Sam3Processor(self.model)
        print("[SuperpixelMaskGen] Model loaded.")

    @torch.no_grad()
    def generate_masks(self, image):
        """
        Generate class-agnostic masks using SLIC-centroid prompts.

        Args:
            image: PIL.Image (RGB) or numpy array (H, W, 3) uint8.

        Returns:
            masks:          list[np.ndarray] — each element is a boolean (H, W) mask.
            scores:         list[float]      — confidence score per mask.
            slic_labels:    np.ndarray (H, W) — the SLIC superpixel map (for vis).
            centroids:      np.ndarray (N, 2) — the (x, y) centroids used as prompts.
        """
        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
            img_np = image
        else:
            pil_image = image
            img_np = np.array(image)

        w, h = pil_image.size

        # ── Step 1: SLIC superpixel decomposition ──────────────────────
        centroids, slic_labels = _slic_centroids(
            img_np, n_segments=self.n_superpixels,
            compactness=self.compactness, sigma=self.sigma,
        )
        print(f"  [SLIC] Generated {len(centroids)} superpixels → {len(centroids)} prompt points")

        # ── Step 2: Encode image once ──────────────────────────────────
        inference_state = self.processor.set_image(pil_image)

        all_masks = []
        all_scores = []

        # ── Step 3: Prompt with each centroid ──────────────────────────
        for pt in centroids:
            try:
                masks_np, scores_np, _ = self.model.predict_inst(
                    inference_state,
                    point_coords=[pt.tolist()],
                    point_labels=[1],
                )
                if masks_np is not None and len(masks_np) > 0:
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

        # ── Step 4: Deduplicate via NMS ────────────────────────────────
        if all_masks:
            all_masks, all_scores = self._nms(all_masks, all_scores, iou_thresh=0.7)

        return all_masks, all_scores, slic_labels, centroids

    # ------------------------------------------------------------------
    @staticmethod
    def _nms(masks, scores, iou_thresh=0.7):
        """Non-maximum suppression over binary masks."""
        order = np.argsort(scores)[::-1]
        keep_masks, keep_scores = [], []

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
