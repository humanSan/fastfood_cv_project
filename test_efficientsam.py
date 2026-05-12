import os
import argparse
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from pipeline_efficientsam import SemanticEfficientSAM3Pipeline
from utils.data_loader import FoodSeg103Dataset, overlay_segmentation, colourise_segmentation

def visualize_stages(image, masks, semantic_map, final_map, gt_map, out_path):
    """Creates a 1x5 subplot showing the intermediate stages and ground truth."""
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    
    img_np = np.array(image)

    # 1. Original Image
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # 2. Stage 1: EfficientSAM3 Masks (Random Colors)
    # We create a pseudo-map where each mask gets a unique ID
    h, w = img_np.shape[:2]
    mask_map = np.full((h, w), 255, dtype=np.uint8)
    for i, m in enumerate(masks):
        mask_map[m > 0] = i % 103  # Wrap around just to give it a color
    
    mask_color = colourise_segmentation(mask_map)
    axes[1].imshow(overlay_segmentation(img_np, mask_map, alpha=0.6))
    axes[1].set_title("Stage 1: EfficientSAM3 Masks", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # 3. Stage 2: SegFormer Semantic Map
    axes[2].imshow(overlay_segmentation(img_np, semantic_map, alpha=0.6))
    axes[2].set_title("Stage 2: SegFormer (Expert)", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    # 4. Final Result Map
    axes[3].imshow(overlay_segmentation(img_np, final_map, alpha=0.6))
    axes[3].set_title("Stage 3: Final Result", fontsize=14, fontweight="bold")
    axes[3].axis("off")

    # 5. Ground Truth
    if gt_map is not None:
        axes[4].imshow(overlay_segmentation(img_np, gt_map, alpha=0.6))
        axes[4].set_title("Ground Truth", fontsize=14, fontweight="bold")
    else:
        axes[4].text(0.5, 0.5, "GT Not Available", ha='center', va='center')
        axes[4].set_title("Ground Truth (N/A)", fontsize=14, fontweight="bold")
    axes[4].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved Visualization] {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Test Semantic-EfficientSAM3 Pipeline on a single image.")
    parser.add_argument("--image", type=str, help="Path to a specific image file (optional).")
    parser.add_argument("--model", type=str, default="s", choices=["s", "m", "l"], help="EfficientSAM3 model size (s, m, l).")
    parser.add_argument("--out_dir", type=str, default="test_output", help="Directory to save the visualization.")
    parser.add_argument("--data_dir", type=str, default="data/FoodSeg103", help="Path to dataset if picking random image.")
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

    # 2. Initialize Pipeline
    print("\nInitializing Pipeline (loading models)...")
    
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
        segformer_ckpt="weights/segformer_foodseg103_best",
        device="cuda"
    )

    # 3. Run Pipeline Stages
    print("\nRunning Pipeline...")
    
    # We call the pipeline which returns the final map and the raw masks
    final_map, masks, labels_info = pipe.run(image, verbose=True)
    
    # To visualize Stage 2, we can just ask the ClosedSet branch for its raw semantic map
    print("Extracting raw SegFormer semantic map for visualization...")
    prob_map = pipe.closed_set.predict_semantic_map(image)
    semantic_map = prob_map.argmax(dim=-1).cpu().numpy()

    # 4. Visualize
    out_path = os.path.join(args.out_dir, "pipeline_stages.png")
    visualize_stages(image, masks, semantic_map, final_map, gt_mask, out_path)
    
    print("\nPipeline test complete!")

if __name__ == "__main__":
    main()
