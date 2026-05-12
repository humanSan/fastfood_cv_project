"""
Test script for the Hybrid Ensemble Pipeline.
"""

import os
import sys
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from pipeline_hybrid_ensemble import SemanticHybridEnsemblePipeline
from utils.data_loader import colourise_segmentation, overlay_segmentation

def main():
    parser = argparse.ArgumentParser(description="Test Hybrid Ensemble Pipeline")
    parser.add_argument("--image", default="data/FoodSeg103/Images/img_dir/test/00000001.jpg")
    parser.add_argument("--model", type=str, default="s", choices=["n", "s", "m", "l", "x"], help="YOLOE-26 model size")
    parser.add_argument("--no_masks", action="store_true", help="Disable mask generation")
    parser.add_argument("--sp_prompt", action="store_true", help="Use superpixel centroids for MobileSAM")
    parser.add_argument("--n_superpixels", type=int, default=200, help="Number of superpixels")
    parser.add_argument("--no_yolo", action="store_true", help="Disable YOLOE verification stage")
    parser.add_argument("--sf_bias", type=float, default=0.15, help="SegFormer confidence bias")
    parser.add_argument("--out_dir", default="output")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]  # Prevent ultralytics from intercepting our CLI args

    os.makedirs(args.out_dir, exist_ok=True)

    yolo_path = f"yoloe-26{args.model}-seg.pt"
    pipe = SemanticHybridEnsemblePipeline(
        yolo_path=yolo_path, 
        sf_bias=args.sf_bias, 
        device=args.device,
        use_yolo=(not args.no_yolo)
    )

    if not os.path.exists(args.image):
        print(f"Image {args.image} not found.")
        return

    image = Image.open(args.image).convert("RGB")
    
    print(f"\nProcessing {args.image}...")
    seg_map, masks, labels_info, timing = pipe.run(
        image, 
        use_masks=(not args.no_masks), 
        sp_prompt=args.sp_prompt,
        n_superpixels=args.n_superpixels,
        no_yolo=args.no_yolo,
        verbose=True
    )

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(np.array(image))
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(colourise_segmentation(seg_map))
    axes[1].set_title("Hybrid Ensemble Result")
    axes[1].axis("off")
    
    axes[2].imshow(overlay_segmentation(np.array(image), seg_map, alpha=0.5))
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "hybrid_test_result.png"))
    plt.close()

    print(f"\nDone! Results saved to {args.out_dir}/")

if __name__ == "__main__":
    main()
