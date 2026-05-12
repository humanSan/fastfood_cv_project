"""
Semantic-FastSAM Pipeline — Ultralytics FastSAM Variant.

Identical to the main pipeline except Stage 1 uses Ultralytics FastSAM
instead of EfficientSAM3 to generate the class-agnostic masks.
This provides a direct 1:1 comparison of the two mask generators.

Usage:
    python pipeline_fastsam.py --image path/to/food_image.jpg
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
from PIL import Image

from models.mask_generator_fastsam import FastSAMMaskGenerator
from models.semantic_branch import ClosedSetBranch
from models.fusion import build_semantic_map
from utils.data_loader import colourise_segmentation, overlay_segmentation
from configs.foodseg103_classes import ID2LABEL


class FastSAMPipeline:
    """
    Full inference pipeline (FastSAM variant):
        Image → FastSAM masks → (SegFormer + MobileCLIP) → Fusion → Semantic map
    """

    def __init__(
        self,
        fastsam_ckpt: str = "FastSAM-x.pt",
        segformer_ckpt: str = "weights/segformer_foodseg103_best",
        fastsam_conf: float = 0.25,
        fastsam_iou: float = 0.7,
        fastsam_imgsz: int = 1024,
        device: str = "cuda",
    ):
        if device == "cuda" and not torch.cuda.is_available():
            print("[Warning] CUDA is not available. Falling back to CPU.")
            device = "cpu"
        self.device = device

        print("=" * 60)
        print("  Semantic-FastSAM Pipeline (Ultralytics) — Init")
        print("=" * 60)

        # Stage 1: FastSAM mask generator
        self.mask_gen = FastSAMMaskGenerator(
            checkpoint_path=fastsam_ckpt,
            device=device,
            conf=fastsam_conf,
            iou=fastsam_iou,
            imgsz=fastsam_imgsz,
        )

        # Stage 2a: Closed-set semantic branch
        self.closed_set = ClosedSetBranch(
            checkpoint_dir=segformer_ckpt,
            device=device,
        )

        print("=" * 60)
        print("  FastSAM Pipeline ready.")
        print("=" * 60)

    def run(self, image, verbose: bool = True):
        """
        Run the simplified FastSAM pipeline on a single image.

        Args:
            image: PIL.Image (RGB) or numpy array (H, W, 3).
            verbose: Whether to print per-mask results.

        Returns:
            semantic_map:  np.ndarray (H, W) with integer class IDs.
            masks:         list of (H, W) boolean masks.
            labels_info:   list of dicts with per-mask metadata.
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
            np_image = image
        else:
            pil_image = image
            np_image = np.array(image)

        h, w = np_image.shape[:2]

        # ── Stage 1: FastSAM masks ─────────────────────────────────────
        t0 = time.time()
        masks, mask_scores = self.mask_gen.generate_masks(pil_image)
        t_mask = time.time() - t0

        if verbose:
            print(f"\n[Stage 1] FastSAM → generated {len(masks)} masks in {t_mask:.2f}s")

        if not masks:
            print("[Warning] No masks generated.")
            return np.zeros((h, w), dtype=np.int32), [], []

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

        return final_map, masks, labels_info


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Semantic-FastSAM Pipeline (Ultralytics Variant)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out_dir", default="output_fastsam", help="Directory to save results")
    parser.add_argument("--fastsam_ckpt", default="FastSAM-s.pt", help="Ultralytics FastSAM weights")
    parser.add_argument("--segformer_ckpt", default="weights/segformer_foodseg103_best")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pipe = FastSAMPipeline(
        fastsam_ckpt=args.fastsam_ckpt,
        segformer_ckpt=args.segformer_ckpt,
        device=args.device,
    )

    image = Image.open(args.image).convert("RGB")
    np_image = np.array(image)

    seg_map, masks, labels_info = pipe.run(image, verbose=True)

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
