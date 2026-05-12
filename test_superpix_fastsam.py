"""
Single-image test for the Superpixel pipeline variant.

Generates a 1×5 visualisation showing each pipeline stage:
  1. Input Image
  2. SLIC Superpixels + Centroids
  3. EfficientSAM3 Masks (from centroid prompts)
  4. SegFormer Semantic Map
  5. Final Fusion

Usage:
    python test_superpix.py
    python test_superpix.py --image path/to/food.jpg
"""

import os
import argparse
import random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries

from pipeline_superpix import SuperpixelPipeline
from utils.data_loader import FoodSeg103Dataset, overlay_segmentation, colourise_segmentation


def visualize_superpix_stages(image, slic_labels, centroids, masks,
                              semantic_map, final_map, gt_map, out_path):
    """1×6 subplot showing stages and ground truth."""
    fig, axes = plt.subplots(1, 6, figsize=(36, 6))

    img_np = np.array(image)
    h, w = img_np.shape[:2]

    # ── 1. Original Image ─────────────────────────────────────────────
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # ── 2. SLIC Superpixels + Centroids ───────────────────────────────
    boundaries = mark_boundaries(img_np / 255.0, slic_labels, color=(1, 1, 0), mode="thick")
    axes[1].imshow(boundaries)
    if centroids is not None and len(centroids) > 0:
        axes[1].scatter(centroids[:, 0], centroids[:, 1],
                        c="red", s=8, zorder=5, edgecolors="white", linewidths=0.3)
    axes[1].set_title(f"SLIC Superpixels ({len(centroids)} centroids)", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    # ── 3. EfficientSAM3 Masks ────────────────────────────────────────
    mask_map = np.full((h, w), 0, dtype=np.uint8)
    for i, m in enumerate(masks):
        mask_map[m > 0] = (i % 102) + 1  # avoid 0 (background)
    axes[2].imshow(overlay_segmentation(img_np, mask_map, alpha=0.6))
    axes[2].set_title(f"Stage 1: Masks ({len(masks)})", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    # ── 4. SegFormer Semantic Map ─────────────────────────────────────
    if hasattr(semantic_map, 'cpu'):
        semantic_map = semantic_map.cpu().numpy()
    axes[3].imshow(overlay_segmentation(img_np, semantic_map, alpha=0.6))
    axes[3].set_title("Stage 2: SegFormer", fontsize=13, fontweight="bold")
    axes[3].axis("off")

    # ── 5. Final Result ───────────────────────────────────────────────
    axes[4].imshow(overlay_segmentation(img_np, final_map, alpha=0.6))
    axes[4].set_title("Stage 3: Final Result", fontsize=13, fontweight="bold")
    axes[4].axis("off")

    # ── 6. Ground Truth ───────────────────────────────────────────────
    if gt_map is not None:
        axes[5].imshow(overlay_segmentation(img_np, gt_map, alpha=0.6))
        axes[5].set_title("Ground Truth", fontsize=13, fontweight="bold")
    else:
        axes[5].text(0.5, 0.5, "GT Not Available", ha='center', va='center')
        axes[5].set_title("Ground Truth (N/A)", fontsize=13, fontweight="bold")
    axes[5].axis("off")

    fig.suptitle("Superpixel Pipeline — Stage Visualisation",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Semantic-EfficientSAM3 Superpixel Pipeline on a single image."
    )
    parser.add_argument("--image", type=str, help="Path to a specific image file (optional).")
    parser.add_argument("--out_dir", type=str, default="test_output_superpix",
                        help="Directory to save the visualisation.")
    parser.add_argument("--data_dir", type=str, default="data/FoodSeg103",
                        help="Path to dataset if picking a random image.")
    parser.add_argument("--n_superpixels", type=int, default=200,
                        help="Target number of SLIC superpixels.")
    parser.add_argument("--compactness", type=float, default=10.0,
                        help="SLIC compactness parameter.")
    parser.add_argument("--model", type=str, default="s", choices=["s", "m", "l"], help="EfficientSAM3 model size (s, m, l).")
    parser.add_argument("--refine_edges", action="store_true",
                        help="Enable post-processing edge refinement with fine superpixels.")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Get the image
    if args.image:
        print(f"Loading user-specified image: {args.image}")
        image = Image.open(args.image).convert("RGB")
        gt_mask = None
    else:
        print("No image specified. Picking a random image from the FoodSeg103 validation set...")
        dataset = FoodSeg103Dataset(root=args.data_dir, split="validation")
        sample = dataset[random.randint(0, len(dataset) - 1)]
        image = sample["image"]
        gt_mask = sample.get("mask", None)
        print("Loaded random image.")

    # 2. Initialise Pipeline
    print("\nInitialising Superpixel Pipeline (loading models)...")

    # Map shorthand to weights and model names
    model_map = {
        "s": ("weights/efficient_sam3_efficientvit_s.pt", "b0"),
        "m": ("weights/efficient_sam3_efficientvit_m.pt", "b1"),
        "l": ("weights/efficient_sam3_efficientvit_l.pt", "b2"),
    }
    ckpt_path, model_name = model_map[args.model]

    pipe = SuperpixelPipeline(
        mask_generator_ckpt=ckpt_path,
        mask_model_name=model_name,
        segformer_ckpt="weights/segformer_foodseg103_best",
        n_superpixels=args.n_superpixels,
        compactness=args.compactness,
        refine_edges=args.refine_edges,
        device="cuda",
    )

    # 3. Run Pipeline
    print("\nRunning Superpixel Pipeline...")
    final_map, masks, labels_info, slic_labels, centroids = pipe.run(image, verbose=True)

    # Get the raw SegFormer map for visualisation (apply argmax to the probability map)
    print("Extracting raw SegFormer semantic map for visualisation...")
    prob_map = pipe.closed_set.predict_semantic_map(image)
    semantic_map = prob_map.argmax(dim=-1).cpu().numpy()

    # 4. Visualise
    out_path = os.path.join(args.out_dir, "superpix_pipeline_stages.png")
    visualize_superpix_stages(image, slic_labels, centroids, masks,
                              semantic_map, final_map, gt_mask, out_path)

    print("\nSuperpixel pipeline test complete!")


if __name__ == "__main__":
    main()
