import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Add current directory to path so we can import from models
sys.path.append(os.getcwd())

from ultralytics import SAM, FastSAM
from utils.data_loader import overlay_segmentation

def run_mobilesam(image_pil, model_path, device):
    print(f"Running MobileSAM ({model_path})...")
    model = SAM(model_path)
    model.to(device)
    results = model.predict(image_pil, device=device, verbose=False)
    if results and results[0].masks is not None:
        # Combine all masks into one visualization map for simplicity
        combined_mask = np.zeros(results[0].masks.data.shape[1:], dtype=np.uint8)
        for i, m in enumerate(results[0].masks.data):
            mask_np = m.cpu().numpy().astype(bool)
            combined_mask[mask_np] = (i % 100) + 1
        return combined_mask
    return None

def run_fastsam(image_pil, model_path, device):
    print(f"Running FastSAM ({model_path})...")
    model = FastSAM(model_path)
    model.to(device)
    results = model.predict(image_pil, device=device, verbose=False)
    if results and results[0].masks is not None:
        combined_mask = np.zeros(results[0].masks.data.shape[1:], dtype=np.uint8)
        for i, m in enumerate(results[0].masks.data):
            mask_np = m.cpu().numpy().astype(bool)
            combined_mask[mask_np] = (i % 100) + 1
        return combined_mask
    return None

def main():
    parser = argparse.ArgumentParser(description="Compare FastSAM and MobileSAM")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mobile_path", default="mobile_sam.pt")
    parser.add_argument("--fast_path", default="FastSAM-s.pt")
    parser.add_argument("--out_dir", default="visualize_output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found.")
        return

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)

    # Run models
    mobile_mask = run_mobilesam(image, args.mobile_path, args.device)
    fast_mask = run_fastsam(image, args.fast_path, args.device)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    if mobile_mask is not None:
        axes[0].imshow(overlay_segmentation(image_np, mobile_mask, alpha=0.5))
        axes[0].set_title(f"MobileSAM\n({args.mobile_path})", fontsize=14, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, "MobileSAM Failed", ha='center')
    axes[0].axis("off")

    if fast_mask is not None:
        axes[1].imshow(overlay_segmentation(image_np, fast_mask, alpha=0.5))
        axes[1].set_title(f"FastSAM\n({args.fast_path})", fontsize=14, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, "FastSAM Failed\n(Weights missing?)", ha='center')
    axes[1].axis("off")

    plt.tight_layout()
    basename = os.path.basename(args.image)
    out_path = os.path.join(args.out_dir, f"compare_{basename}")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"\nSaved comparison visualization to {out_path}")

if __name__ == "__main__":
    main()
