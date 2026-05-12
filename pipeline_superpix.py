"""
Semantic-EfficientSAM3 Pipeline — Superpixel Variant.

Identical to the standard pipeline except Stage 1 uses SLIC superpixel
centroids as point prompts instead of a uniform grid.  This provides
content-aware, adaptive prompting:
  - Large uniform regions (plate, table) → few prompts
  - Complex textured regions (mixed salad) → many prompts

Outputs are saved to separate directories so results can be compared
side-by-side with the standard pipeline.

Usage:
    python pipeline_superpix.py --image path/to/food_image.jpg
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
from PIL import Image

from skimage.segmentation import slic

from models.mask_generator_superpix import SuperpixelMaskGenerator
from models.semantic_branch import ClosedSetBranch
from models.fusion import build_semantic_map
from utils.data_loader import colourise_segmentation, overlay_segmentation
from configs.foodseg103_classes import ID2LABEL


def refine_with_superpixels(semantic_map: np.ndarray, image_np: np.ndarray,
                            n_segments: int = 1000, compactness: float = 10.0,
                            sigma: float = 1.0) -> np.ndarray:
    """
    Refine a semantic map by snapping label boundaries to superpixel edges.

    For each fine-grained SLIC superpixel, the majority predicted label inside
    it wins.  This forces the output edges to follow the real colour / texture
    gradients of the image, removing jagged aliasing artifacts from the deep
    learning masks.

    Args:
        semantic_map:  (H, W) int array of predicted class IDs.
        image_np:      (H, W, 3) uint8 RGB array.
        n_segments:    Number of fine superpixels (higher = tighter edges).
        compactness:   SLIC compactness (higher = more square superpixels).
        sigma:         Gaussian smoothing before SLIC.

    Returns:
        refined_map:   (H, W) int array with edge-refined class IDs.
    """
    fine_labels = slic(image_np, n_segments=n_segments, compactness=compactness,
                       sigma=sigma, start_label=0, channel_axis=2)

    refined_map = np.copy(semantic_map)
    for sp_id in np.unique(fine_labels):
        sp_mask = fine_labels == sp_id
        region_labels = semantic_map[sp_mask]
        if len(region_labels) == 0:
            continue
        # Majority vote inside this superpixel
        counts = np.bincount(region_labels.astype(np.int64))
        refined_map[sp_mask] = int(counts.argmax())

    return refined_map


class SuperpixelPipeline:
    """
    Full inference pipeline (superpixel variant):
        Image → SLIC centroids → EfficientSAM3 masks → (SegFormer + MobileCLIP) → Fusion → Semantic map
    """

    def __init__(
        self,
        mask_generator_ckpt: str = "weights/efficient_sam3_efficientvit_s.pt",
        mask_backbone_type: str = "efficientvit",
        mask_model_name: str = "b0",
        segformer_ckpt: str = "weights/segformer_foodseg103_best",
        n_superpixels: int = 200,
        compactness: float = 10.0,
        sigma: float = 1.0,
        refine_edges: bool = False,
        refine_n_segments: int = 1000,
        refine_compactness: float = 10.0,
        device: str = "cuda",
    ):
        if device == "cuda" and not torch.cuda.is_available():
            print("[Warning] CUDA is not available. Falling back to CPU.")
            device = "cpu"
        self.device = device
        self.refine_edges = refine_edges
        self.refine_n_segments = refine_n_segments
        self.refine_compactness = refine_compactness

        print("=" * 60)
        print("  Semantic-EfficientSAM3  Pipeline (Superpixel) — Init")
        print(f"  Device: {device}")
        print("=" * 60)

        # Stage 1: Superpixel mask generator
        self.mask_gen = SuperpixelMaskGenerator(
            checkpoint_path=mask_generator_ckpt,
            backbone_type=mask_backbone_type,
            model_name=mask_model_name,
            n_superpixels=n_superpixels,
            compactness=compactness,
            sigma=sigma,
            device=device,
        )

        # Stage 2: Closed-set semantic branch
        self.closed_set = ClosedSetBranch(
            checkpoint_dir=segformer_ckpt,
            device=device,
        )

        print("=" * 60)
        print("  Superpixel Pipeline ready.")
        print("=" * 60)

    def run(self, image, verbose: bool = True, refine_edges: bool = None):
        """
        Run the simplified superpixel pipeline on a single image.

        Args:
            image: PIL.Image (RGB) or numpy array (H, W, 3).
            verbose: Whether to print per-mask results.
            refine_edges: Override the instance-level setting.  If None, uses
                          the value set at construction time.

        Returns:
            semantic_map:  np.ndarray (H, W) with integer class IDs.
            masks:         list of (H, W) boolean masks.
            labels_info:   list of dicts with per-mask metadata.
            slic_labels:   np.ndarray (H, W) — the SLIC superpixel map.
            centroids:     np.ndarray (N, 2) — (x, y) centroids used as prompts.
        """
        do_refine = refine_edges if refine_edges is not None else self.refine_edges
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
            np_image = image
        else:
            pil_image = image
            np_image = np.array(image)

        h, w = np_image.shape[:2]

        # ── Stage 1: SLIC → centroid prompts → EfficientSAM3 masks ────
        t0 = time.time()
        masks, mask_scores, slic_labels, centroids = self.mask_gen.generate_masks(pil_image)
        t_mask = time.time() - t0

        if verbose:
            print(f"\n[Stage 1] SLIC({len(centroids)} centroids) → {len(masks)} masks in {t_mask:.2f}s")

        if not masks:
            print("[Warning] No masks generated.")
            return np.zeros((h, w), dtype=np.int32), [], [], slic_labels, centroids

        # ── Stage 2: Closed-set probability map ──────────────────────────
        t0 = time.time()
        prob_map_cs = self.closed_set.predict_semantic_map(pil_image)
        t_cs = time.time() - t0

        if verbose:
            print(f"[Stage 2] Closed-set segmentation in {t_cs:.2f}s")

        # ── Stage 3: Label each mask ─────────────────────────
        t0 = time.time()
        final_labels = []
        labels_info = []

        for i, (mask, score) in enumerate(zip(masks, mask_scores)):
            # Closed-set prediction
            cs_id, cs_conf = self.closed_set.label_mask(prob_map_cs, mask)

            final_id = cs_id
            final_name = ID2LABEL.get(cs_id, "unknown")

            final_labels.append(final_id)
            info = {
                "mask_idx": i,
                "mask_score": score,
                "closed_set_id": cs_id,
                "closed_set_name": final_name,
                "closed_set_confidence": cs_conf,
                "final_id": final_id,
                "final_name": final_name,
            }
            labels_info.append(info)

            if verbose:
                print(
                    f"  Mask {i:3d} | Label: {final_name:20s} (conf: {cs_conf:.2f})"
                )

        t_label = time.time() - t0

        if verbose:
            print(f"[Stage 3] Labelled {len(masks)} masks in {t_label:.2f}s")
            print(f"[Total] {t_mask + t_cs + t_label:.2f}s")

        # ── Compose final semantic map ─────────────────────────────────
        final_map = build_semantic_map((h, w), masks, final_labels)

        # ── Optional: Superpixel edge refinement ───────────────────────
        if do_refine:
            t0 = time.time()
            final_map = refine_with_superpixels(
                final_map, np_image,
                n_segments=self.refine_n_segments,
                compactness=self.refine_compactness,
            )
            t_refine = time.time() - t0
            if verbose:
                print(f"[Refine] Edge refinement with {self.refine_n_segments} "
                      f"fine superpixels in {t_refine:.2f}s")

        return final_map, masks, labels_info, slic_labels, centroids


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Semantic-EfficientSAM3 Pipeline (Superpixel Variant)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out_dir", default="output_superpix", help="Directory to save results")
    parser.add_argument("--mask_ckpt", default="weights/efficient_sam3_efficientvit_s.pt")
    parser.add_argument("--segformer_ckpt", default="weights/segformer_foodseg103_best")
    parser.add_argument("--n_superpixels", type=int, default=200, help="Target number of SLIC superpixels")
    parser.add_argument("--compactness", type=float, default=10.0, help="SLIC compactness parameter")
    parser.add_argument("--refine_edges", action="store_true",
                        help="Enable post-processing edge refinement with fine superpixels")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pipe = SuperpixelPipeline(
        mask_generator_ckpt=args.mask_ckpt,
        segformer_ckpt=args.segformer_ckpt,
        n_superpixels=args.n_superpixels,
        compactness=args.compactness,
        refine_edges=args.refine_edges,
        device=args.device,
    )

    image = Image.open(args.image).convert("RGB")
    np_image = np.array(image)

    seg_map, masks, labels_info, slic_labels, centroids = pipe.run(image, verbose=True)

    # Save results
    basename = os.path.splitext(os.path.basename(args.image))[0]

    # 1. Colourised segmentation map
    colour_map = colourise_segmentation(seg_map)
    Image.fromarray(colour_map).save(os.path.join(args.out_dir, f"{basename}_segmap.png"))

    # 2. Overlay on original image
    overlay = overlay_segmentation(np_image, seg_map, alpha=0.5)
    Image.fromarray(overlay).save(os.path.join(args.out_dir, f"{basename}_overlay.png"))

    print(f"\n[Saved] Results written to {args.out_dir}/")


if __name__ == "__main__":
    main()
