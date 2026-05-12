"""
Single-image test for the FastSAM pipeline variant.

Generates a 1×4 visualisation showing each pipeline stage:
  1. Input Image
  2. FastSAM Masks
  3. SegFormer Semantic Map
  4. Final Fusion

Usage:
    python test_fastsam.py
    python test_fastsam.py --image path/to/food.jpg
"""

import os
import argparse
import random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_fastsam import FastSAMPipeline
from utils.data_loader import FoodSeg103Dataset, overlay_segmentation, colourise_segmentation


def visualize_stages(image, masks, semantic_map, final_map, gt_map, out_path):
    """1×5 subplot showing stages and ground truth."""
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    img_np = np.array(image)
    h, w = img_np.shape[:2]

    # ── 1. Original Image ─────────────────────────────────────────────
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # ── 2. FastSAM Masks ──────────────────────────────────────────────
    mask_map = np.full((h, w), 0, dtype=np.uint8)
    for i, m in enumerate(masks):
        mask_map[m > 0] = (i % 102) + 1  # avoid 0 (background)
    axes[1].imshow(overlay_segmentation(img_np, mask_map, alpha=0.6))
    axes[1].set_title(f"Stage 1: FastSAM Masks ({len(masks)})", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # ── 3. SegFormer Semantic Map ─────────────────────────────────────
    if hasattr(semantic_map, 'cpu'):
        semantic_map = semantic_map.cpu().numpy()
    axes[2].imshow(overlay_segmentation(img_np, semantic_map, alpha=0.6))
    axes[2].set_title("Stage 2: SegFormer", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    # ── 4. Final Result ───────────────────────────────────────────────
    axes[3].imshow(overlay_segmentation(img_np, final_map, alpha=0.6))
    axes[3].set_title("Stage 3: Final Result", fontsize=14, fontweight="bold")
    axes[3].axis("off")

    # ── 5. Ground Truth ───────────────────────────────────────────────
    if gt_map is not None:
        axes[4].imshow(overlay_segmentation(img_np, gt_map, alpha=0.6))
        axes[4].set_title("Ground Truth", fontsize=14, fontweight="bold")
    else:
        axes[4].text(0.5, 0.5, "GT Not Available", ha='center', va='center')
        axes[4].set_title("Ground Truth (N/A)", fontsize=14, fontweight="bold")
    axes[4].axis("off")

    fig.suptitle("FastSAM Pipeline — Stage Visualisation",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Semantic-FastSAM Pipeline on a single image."
    )
    parser.add_argument("--image", type=str, help="Path to a specific image file (optional).")
    parser.add_argument("--out_dir", type=str, default="test_output_fastsam",
                        help="Directory to save the visualisation.")
    parser.add_argument("--data_dir", type=str, default="data/FoodSeg103",
                        help="Path to dataset if picking a random image.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Get the image
    if args.image:
        print(f"Loading user-specified image: {args.image}")
        image = Image.open(args.image).convert("RGB")
        gt_mask = None  # No GT for user images unless we find it in the same folder
    else:
        print("No image specified. Picking a random image from the FoodSeg103 validation set...")
        dataset = FoodSeg103Dataset(root=args.data_dir, split="validation")
        sample = dataset[random.randint(0, len(dataset) - 1)]
        image = sample["image"]
        gt_mask = sample.get("mask", None)
        print("Loaded random image.")

    # 2. Initialise Pipeline
    print("\nInitialising FastSAM Pipeline (loading models)...")
    pipe = FastSAMPipeline(
        fastsam_ckpt="FastSAM-x.pt",
        segformer_ckpt="weights/segformer_foodseg103_best",
        device="cuda",
    )

    # 3. Run Pipeline
    print("\nRunning FastSAM Pipeline...")
    final_map, masks, labels_info = pipe.run(image, verbose=True)

    # Get the raw SegFormer map for visualisation (apply argmax to the probability map)
    print("Extracting raw SegFormer semantic map for visualisation...")
    prob_map = pipe.closed_set.predict_semantic_map(image)
    semantic_map = prob_map.argmax(dim=-1).cpu().numpy()

    # 4. Visualise
    out_path = os.path.join(args.out_dir, "fastsam_pipeline_stages.png")
    visualize_stages(image, masks, semantic_map, final_map, gt_mask, out_path)

    print("\nFastSAM pipeline test complete!")


if __name__ == "__main__":
    main()
