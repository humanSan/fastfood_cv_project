import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries

def main():
    parser = argparse.ArgumentParser(description="Visualize SLIC Superpixels")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--n_segments", type=int, default=200, help="Number of superpixels")
    parser.add_argument("--compactness", type=float, default=10.0, help="SLIC compactness")
    parser.add_argument("--out_dir", default="visualize_output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found.")
        return

    # Load image
    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)

    # Run SLIC
    print(f"Running SLIC with {args.n_segments} segments...")
    segments = slic(image_np, n_segments=args.n_segments, compactness=args.compactness,
                    start_label=0, channel_axis=2)

    # Draw boundaries
    out_image = mark_boundaries(image_np, segments, color=(1, 1, 0)) # Yellow boundaries

    # Save
    basename = os.path.basename(args.image)
    out_path = os.path.join(args.out_dir, f"superpixel_{basename}")
    plt.imsave(out_path, out_image)

    print(f"Saved superpixel visualization to {out_path}")

if __name__ == "__main__":
    main()
