"""
Mask Generator Module — uses Ultralytics SAM3 for high-speed mask generation.

Utilizes Ultralytics' optimized inference engine and TensorRT (if available) 
to process prompts in batches, significantly reducing latency compared to 
standard research implementations.
"""

import numpy as np
import torch
from ultralytics import SAM

class UltralyticsSAM3MaskGenerator:
    """Generates class-agnostic masks using Ultralytics SAM3."""

    def __init__(
        self,
        model_path: str = "sam3-s.pt",  # Example Ultralytics SAM3 weights
        device: str = "cuda",
        conf: float = 0.25,
        iou: float = 0.7,
    ):
        self.device = device
        self.conf = conf
        self.iou = iou

        print(f"[MaskGenerator] Loading Ultralytics SAM3 from {model_path} …")
        self.model = SAM(model_path)
        print(f"[MaskGenerator] Model loaded on {device}.")

    @torch.no_grad()
    def generate_masks(self, image):
        """
        Generate class-agnostic masks for the entire image using 'everything' mode.
        
        Args:
            image: PIL.Image or numpy array.

        Returns:
            masks:  list[np.ndarray]
            scores: list[float]
        """
        # Ultralytics 'everything' mode is much faster than point-grid loop
        results = self.model.predict(image, device=self.device, conf=self.conf, iou=self.iou, verbose=False)
        
        if not results or results[0].masks is None:
            return [], []

        # Extract masks and scores
        masks_tensor = results[0].masks.data  # (N, H, W)
        scores_tensor = results[0].probs if hasattr(results[0], 'probs') else results[0].boxes.conf
        
        masks = [m.cpu().numpy().astype(bool) for m in masks_tensor]
        scores = [float(s) for s in scores_tensor.cpu().numpy()]

        return masks, scores
