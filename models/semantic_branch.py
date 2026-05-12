"""
Semantic Labeling Branches for the Semantic-EfficientSAM3 pipeline.

Two parallel branches assign semantic labels to each class-agnostic mask:

1. ClosedSetBranch  — A SegFormer-B0 fine-tuned on FoodSeg103 that produces
   a full-image semantic map.  For each mask, a majority vote over the map
   gives a closed-set label.

2. OpenVocabBranch  — Uses EfficientSAM3's MobileCLIP text encoder to score
   each mask region against the 103 FoodSeg103 class names, replacing the
   heavy BLIP captioning + CLIP ranking pipeline from original SFS.
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root so configs is importable
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

from configs.foodseg103_classes import FOODSEG103_CLASSES, NUM_CLASSES, ID2LABEL


# ═══════════════════════════════════════════════════════════════════════════
# Branch A — Closed-Set Semantic Branch (SegFormer fine-tuned on FoodSeg103)
# ═══════════════════════════════════════════════════════════════════════════

class ClosedSetBranch:
    """
    Produces a full-image semantic segmentation map using a SegFormer model
    fine-tuned on FoodSeg103. For each binary mask, the predicted class is
    determined by probability-weighted voting (soft voting) over the semantic 
    map within the mask region.
    """

    def __init__(self, checkpoint_dir: str = "weights/segformer_foodseg103", device: str = "cuda"):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.device = device
        print(f"[ClosedSetBranch] Loading SegFormer from {checkpoint_dir} …")

        self.processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        self.processor.do_reduce_labels = False

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint_dir,
        ).to(device)
        self.model.eval()
        print("[ClosedSetBranch] Model loaded.")

    @torch.no_grad()
    def predict_semantic_map(self, image):
        """
        Run full-image semantic segmentation.

        Args:
            image: PIL.Image (RGB) or numpy array (H, W, 3).

        Returns:
            semantic_map: torch.Tensor of shape (H, W) with integer class IDs.
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        w, h = pil_image.size
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, num_classes, h', w')
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=True)
        prob_map = F.softmax(logits, dim=1).squeeze(0)  # (C, H, W)
        prob_map = prob_map.permute(1, 2, 0)  # (H, W, C)
        return prob_map  # tensor on self.device

    @staticmethod
    def label_mask(prob_map: torch.Tensor, mask: np.ndarray):
        """
        Assign a closed-set label to a binary mask via probability-weighted vote.

        Args:
            prob_map: (H, W, C) float tensor of softmax probabilities.
            mask: (H, W) boolean numpy array.

        Returns:
            class_id:   int — the class with the highest total probability sum.
            confidence: float — average probability of the winning class across mask pixels.
        """
        # Efficiently extract probabilities for mask pixels
        # prob_map is (H, W, C), mask is (H, W)
        mask_tensor = torch.from_numpy(mask).to(prob_map.device)
        mask_probs = prob_map[mask_tensor] # (N_mask_pixels, C)

        if mask_probs.numel() == 0:
            return 0, 0.0

        # Sum probabilities across the mask
        class_sums = mask_probs.sum(dim=0) # (C,)
        class_id = int(class_sums.argmax())
        
        # Confidence is the average probability of the winning class
        confidence = float(class_sums[class_id]) / float(mask_probs.size(0))
        
        return class_id, confidence

