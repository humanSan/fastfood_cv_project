"""
Standalone test script for the fine-tuned SegFormer model.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.semantic_branch import ClosedSetBranch
from utils.data_loader import colourise_segmentation, overlay_segmentation
from configs.foodseg103_classes import ID2LABEL

def main():
    parser = argparse.ArgumentParser(description="SegFormer Standalone Test")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--ckpt", default="weights/segformer_foodseg103_best", help="Path to SegFormer weights")
    parser.add_argument("--out_dir", default="output_segformer")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize model
    model = ClosedSetBranch(checkpoint_dir=args.ckpt, device=args.device)

    if not os.path.exists(args.image):
        print(f"Image {args.image} not found.")
        return

    image = Image.open(args.image).convert("RGB")
    
    print(f"\nProcessing {args.image} with SegFormer...")
    
    # Predict
    prob_map = model.predict_semantic_map(image)
    seg_map = prob_map.argmax(dim=-1).cpu().numpy().astype(np.uint8)

    # Visualisation
    color_seg = colourise_segmentation(seg_map)
    overlay = overlay_segmentation(np.array(image), seg_map)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(np.array(image))
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(color_seg)
    axes[1].set_title("SegFormer Prediction")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    out_path = os.path.join(args.out_dir, os.path.basename(args.image))
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    print(f"Results saved to {out_path}")

    # Print detected classes
    unique_ids = np.unique(seg_map)
    print("\nDetected Classes:")
    for uid in unique_ids:
        if uid in ID2LABEL:
            print(f"  - {ID2LABEL[uid]}")

if __name__ == "__main__":
    main()
