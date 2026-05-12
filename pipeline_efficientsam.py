"""
Semantic-EfficientSAM3 Pipeline — main orchestrator.

Ties together:
  1. EfficientSAM3 mask generator  (replaces FastSAM)
  2. ClosedSetBranch               (SegFormer fine-tuned on FoodSeg103)
  3. OpenVocabBranch                (EfficientSAM3 MobileCLIP text encoder)
  4. Fusion module                  (SFS decision heuristic)

Usage:
    python pipeline.py --image path/to/food_image.jpg
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
from PIL import Image

from models.mask_generator import EfficientSAM3MaskGenerator
from models.semantic_branch import ClosedSetBranch
from models.fusion import build_semantic_map
from utils.data_loader import colourise_segmentation, overlay_segmentation
from configs.foodseg103_classes import ID2LABEL


class SemanticEfficientSAM3Pipeline:
    """
    Full inference pipeline:
        Image → EfficientSAM3 masks → (SegFormer + MobileCLIP) → Fusion → Semantic map
    """

    def __init__(
        self,
        mask_generator_ckpt: str = "weights/efficient_sam3_efficientvit_s.pt",
        mask_backbone_type: str = "efficientvit",
        mask_model_name: str = "b0",
        segformer_ckpt: str = "weights/segformer_foodseg103_best",
        n_points_per_side: int = 32,
        device: str = "cuda",
    ):
        if device == "cuda" and not torch.cuda.is_available():
            print("[Warning] CUDA is not available. Falling back to CPU.")
            device = "cpu"
        self.device = device

        print("=" * 60)
        print("  Semantic-EfficientSAM3  Pipeline — Initialising")
        print(f"  Device: {device}")
        print("=" * 60)

        # Stage 1: Mask generator
        self.mask_gen = EfficientSAM3MaskGenerator(
            checkpoint_path=mask_generator_ckpt,
            backbone_type=mask_backbone_type,
            model_name=mask_model_name,
            n_points_per_side=n_points_per_side,
            device=device,
        )

        # Stage 2: Closed-set semantic branch
        self.closed_set = ClosedSetBranch(
            checkpoint_dir=segformer_ckpt,
            device=device,
        )

        print("=" * 60)
        print("  Pipeline ready.")
        print("=" * 60)

    def run(self, image, verbose: bool = True):
        """
        Run the simplified pipeline on a single image.

        Args:
            image: PIL.Image (RGB) or numpy array (H, W, 3).
            verbose: Whether to print per-mask results.

        Returns:
            semantic_map: np.ndarray (H, W) with integer class IDs.
            masks:        list of (H, W) boolean masks.
            labels_info:  list of dicts with per-mask metadata.
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
            np_image = image
        else:
            pil_image = image
            np_image = np.array(image)

        h, w = np_image.shape[:2]

        # ── Stage 1: Generate class-agnostic masks ─────────────────────
        t0 = time.time()
        masks, mask_scores = self.mask_gen.generate_masks(pil_image)
        t_mask = time.time() - t0

        if verbose:
            print(f"\n[Stage 1] Generated {len(masks)} masks in {t_mask:.2f}s")

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

        return final_map, masks, labels_info


# CLI

def main():
    parser = argparse.ArgumentParser(description="Semantic-EfficientSAM3 Pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out_dir", default="output", help="Directory to save results")
    parser.add_argument("--model", type=str, default="s", choices=["s", "m", "l"], help="EfficientSAM3 model size (s, m, l).")
    parser.add_argument("--segformer_ckpt", default="weights/segformer_foodseg103")
    parser.add_argument("--n_points", type=int, default=32, help="Points per side for grid")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Map shorthand to weights and model names
    model_map = {
        "s": ("weights/efficient_sam3_efficientvit_s.pt", "b0"),
        "m": ("weights/efficient_sam3_efficientvit_m.pt", "b1"),
        "l": ("weights/efficient_sam3_efficientvit_l.pt", "b2"),
    }
    ckpt_path, model_name = model_map[args.model]

    pipe = SemanticEfficientSAM3Pipeline(
        mask_generator_ckpt=ckpt_path,
        mask_model_name=model_name,
        segformer_ckpt=args.segformer_ckpt,
        n_points_per_side=args.n_points,
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
