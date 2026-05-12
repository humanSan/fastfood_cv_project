"""
Fusion Module — merges predictions from the Closed-Set and Open-Vocabulary
branches following the Semantic-Fast-SAM (SFS) decision heuristic.

Decision logic:
  1. If the closed-set branch is confident (confidence ≥ threshold) AND
     the open-vocabulary top-1 agrees or has low score  →  keep closed-set label.
  2. If the open-vocabulary branch proposes a label with high similarity
     that is inside the FoodSeg103 taxonomy                →  use open-vocab label.
  3. Otherwise fall back to closed-set label.
  4. Masks that remain unlabeled get class 0 ("background").
"""

import numpy as np
import torch
from typing import List, Tuple, Optional

import sys, os
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

from configs.foodseg103_classes import NUM_CLASSES, ID2LABEL


# ═══════════════════════════════════════════════════════════════════════════
# Compose Final Map
# ═══════════════════════════════════════════════════════════════════════════


def build_semantic_map(
    image_shape: Tuple[int, int],           # (H, W)
    masks: List[np.ndarray],                # list of (H, W) bool
    final_labels: List[int],                # class ID per mask
) -> np.ndarray:
    """
    Compose all labelled masks into a single semantic segmentation map.

    Later masks in the list take priority over earlier ones (painter's algorithm),
    matching the original SFS behaviour where higher-confidence masks are
    processed last.

    Args:
        image_shape:  (H, W) of the original image.
        masks:        Binary masks for each detected region.
        final_labels: Integer class ID for each mask.

    Returns:
        semantic_map: np.ndarray of shape (H, W) with integer class IDs.
    """
    semantic_map = np.zeros(image_shape[:2], dtype=np.int32)
    for mask, label in zip(masks, final_labels):
        semantic_map[mask] = label
    return semantic_map
