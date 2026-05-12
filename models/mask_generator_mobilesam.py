"""
Mask Generator Module — uses MobileSAM (via Ultralytics) for fast, class-agnostic 
segment-everything.
"""

import torch
import numpy as np
from ultralytics import SAM
from skimage.segmentation import slic
from skimage.measure import regionprops

class MobileSAMMaskGenerator:
    """Generates class-agnostic masks using MobileSAM."""

    def __init__(
        self,
        model_path: str = "mobile_sam.pt",
        device: str = "cuda",
        conf: float = 0.25,
    ):
        self.device = device
        self.conf = conf

        print(f"[MaskGenerator] Loading MobileSAM from {model_path} …")
        # Ultralytics SAM handles MobileSAM weights automatically
        self.model = SAM(model_path)
        self.model.to(device)
        print(f"[MaskGenerator] MobileSAM loaded on {device}.")

    def _slic_centroids(self, image_np, n_segments=200, compactness=10.0):
        labels = slic(image_np, n_segments=n_segments, compactness=compactness,
                      start_label=0, channel_axis=2)
        regions = regionprops(labels + 1)
        centroids = []
        for region in regions:
            cy, cx = region.centroid
            centroids.append([cx, cy])
        return np.array(centroids)

    @torch.no_grad()
    def generate_masks(self, image, use_superpixels=False, n_superpixels=200):
        """
        Run MobileSAM in 'everything' mode or superpixel-prompted mode.
        """
        if not use_superpixels:
            results = self.model.predict(image, device=self.device, conf=self.conf, verbose=False)
            if not results or results[0].masks is None:
                return [], []
            masks_tensor = results[0].masks.data
            scores_tensor = results[0].boxes.conf if results[0].boxes is not None else torch.ones(len(masks_tensor))
            masks = [m.cpu().numpy().astype(bool) for m in masks_tensor]
            scores = [float(s) for s in scores_tensor.cpu().numpy()]
            return masks, scores
        else:
            # Superpixel-prompted mode
            np_image = np.array(image)
            centroids = self._slic_centroids(np_image, n_segments=n_superpixels)
            
            # To get separate masks per point, we batch prompts as [N, 1, 2]
            prompt_points = centroids[:, np.newaxis, :] # (N, 1, 2)
            prompt_labels = np.ones((len(centroids), 1)) # (N, 1)
            
            results = self.model.predict(image, points=prompt_points, labels=prompt_labels, 
                                         device=self.device, conf=self.conf, verbose=False)
            
            if results:
                masks_data = results[0].masks.data # (N, H, W)
                scores_data = results[0].boxes.conf # (N,)
                if masks_data is not None:
                    # Filter by conf
                    keep = scores_data >= self.conf
                    masks_data = masks_data[keep]
                    scores_data = scores_data[keep]
                    return self._nms(masks_data, scores_data)
            
            return [], []

    def _nms(self, masks_tensor, scores_tensor, iou_thresh=0.7):
        """GPU-accelerated Non-maximum suppression over binary masks."""
        if len(masks_tensor) == 0:
            return [], []
            
        # Ensure they are on device
        masks_tensor = masks_tensor.to(self.device)
        scores_tensor = scores_tensor.to(self.device)
        
        # Sort by score
        order = torch.argsort(scores_tensor, descending=True)
        masks_tensor = masks_tensor[order]
        scores_tensor = scores_tensor[order]
        
        keep_indices = []
        num_masks = len(masks_tensor)
        
        # Iterative suppression
        suppressed = torch.zeros(num_masks, dtype=torch.bool, device=self.device)
        
        for i in range(num_masks):
            if suppressed[i]:
                continue
            keep_indices.append(i)
            
            m1 = masks_tensor[i]
            # Vectorized IoU with remaining
            # This is done on GPU
            others = masks_tensor[i+1:]
            if len(others) == 0:
                break
                
            intersection = (m1 & others).sum(dim=(1, 2))
            union = (m1 | others).sum(dim=(1, 2))
            iou = intersection.float() / (union.float() + 1e-6)
            
            # Update suppression list
            too_high = iou > iou_thresh
            suppressed[i+1:][too_high] = True
            
        final_masks = [m.cpu().numpy().astype(bool) for m in masks_tensor[keep_indices]]
        final_scores = [float(s) for s in scores_tensor[keep_indices]]
        return final_masks, final_scores
