"""
FastSAM Mask Generator.

Replaces EfficientSAM3 with Ultralytics FastSAM to produce class-agnostic masks.
This allows for direct comparison of mask generation performance (latency and
accuracy) between FastSAM and EfficientSAM3 within the same pipeline.
"""

import os
import numpy as np
import torch
from PIL import Image as PILImage

try:
    from ultralytics import FastSAM
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False


class FastSAMMaskGenerator:
    """
    Generates class-agnostic masks using Ultralytics FastSAM.
    """

    def __init__(
        self,
        checkpoint_path: str = "FastSAM-s.pt",
        device: str = "cuda",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 1024,
    ):
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        if not _HAS_ULTRALYTICS:
            raise ImportError(
                "Ultralytics is not installed. Run `pip install ultralytics`."
            )

        print(f"[FastSAMGenerator] Loading FastSAM from {checkpoint_path} …")
        self.model = FastSAM(checkpoint_path)
        print("[FastSAMGenerator] Model loaded.")

    @torch.no_grad()
    def generate_masks(self, image):
        """
        Generate class-agnostic masks for the entire image using FastSAM.

        Args:
            image: PIL.Image (RGB) or numpy array (H, W, 3) uint8.

        Returns:
            masks:  list[np.ndarray]  — each element is a boolean mask of shape (H, W).
            scores: list[float]       — confidence score per mask.
        """
        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image

        w, h = pil_image.size

        # Run FastSAM inference
        # retina_masks=True gives higher resolution masks matching the original image size
        results = self.model(
            pil_image,
            device=self.device,
            retina_masks=True,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )

        all_masks = []
        all_scores = []

        if len(results) > 0 and results[0].masks is not None:
            # .data is (N, H, W) tensor on the device
            masks_tensor = results[0].masks.data
            # .conf is (N,) tensor of confidence scores
            scores_tensor = results[0].boxes.conf

            # Ensure masks are resized back to exactly (h, w) if they aren't already
            if masks_tensor.shape[1:] != (h, w):
                import torch.nn.functional as F
                masks_tensor = F.interpolate(
                    masks_tensor.unsqueeze(1), size=(h, w), mode="nearest"
                ).squeeze(1)

            masks_np = masks_tensor.cpu().numpy().astype(bool)
            scores_np = scores_tensor.cpu().numpy().astype(float)

            for i in range(len(masks_np)):
                mask = masks_np[i]
                score = scores_np[i]

                # Skip tiny / nearly-empty masks
                if mask.sum() < 100:
                    continue

                all_masks.append(mask)
                all_scores.append(score)

        return all_masks, all_scores
