"""
FoodSeg103 Data Loader Utilities.

Provides:
  - FoodSeg103Dataset: A PyTorch Dataset for loading image/mask pairs stored
    on disk (after running download_dataset.py).
  - Helper functions for colourising and overlaying predictions.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import sys
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

from configs.foodseg103_classes import FOODSEG103_CLASSES, NUM_CLASSES


class FoodSeg103Dataset(Dataset):
    """
    Loads FoodSeg103 images and annotation masks from disk.

    Expected directory layout (created by download_dataset.py):
        data/FoodSeg103/<split>/images/<id>.jpg
        data/FoodSeg103/<split>/annotations/<id>.png
    """

    def __init__(self, root: str = "data/FoodSeg103", split: str = "test", transform=None):
        self.root = os.path.join(root, split)
        self.image_dir = os.path.join(self.root, "images")
        self.mask_dir = os.path.join(self.root, "annotations")
        self.transform = transform

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}. "
                "Run download_dataset.py first."
            )

        self.filenames = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")

        mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
        else:
            # No ground truth available (inference-only mode)
            mask = None

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "mask": mask,
            "filename": img_name,
        }


# ─── Visualisation helpers ──────────────────────────────────────────────────

def _random_colour_palette(n: int, seed: int = 42) -> np.ndarray:
    """Return a (n, 3) uint8 colour palette."""
    rng = np.random.RandomState(seed)
    palette = rng.randint(0, 255, (n, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # background = black
    return palette


_PALETTE = _random_colour_palette(NUM_CLASSES)


def colourise_segmentation(seg_map: np.ndarray) -> np.ndarray:
    """Convert an integer segmentation map to an RGB image."""
    h, w = seg_map.shape[:2]
    # Ensure indices are within palette range (map ignore/oob labels to background)
    seg_map_clip = seg_map.copy()
    seg_map_clip[seg_map_clip >= NUM_CLASSES] = 0
    rgb = _PALETTE[seg_map_clip.ravel()].reshape(h, w, 3)
    return rgb


def overlay_segmentation(image: np.ndarray, seg_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a colourised segmentation map over the original image."""
    colour = colourise_segmentation(seg_map)
    if image.shape[:2] != seg_map.shape[:2]:
        from PIL import Image as PILImage
        colour = np.array(
            PILImage.fromarray(colour).resize(
                (image.shape[1], image.shape[0]), PILImage.NEAREST
            )
        )
    blended = (alpha * colour.astype(np.float32) + (1 - alpha) * image.astype(np.float32)).astype(np.uint8)
    return blended
