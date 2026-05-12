"""
Semantic-Hybrid-Ensemble Pipeline.

Combines:
  1. MobileSAM (Stage 1: High-recall masks)
  2. SegFormer (Stage 2: Specialist food labeling)
  3. YOLO26    (Stage 3: Object-level verification)
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLOE

from models.mask_generator_mobilesam import MobileSAMMaskGenerator
from models.semantic_branch import ClosedSetBranch
from models.fusion import build_semantic_map
from utils.data_loader import colourise_segmentation, overlay_segmentation
from configs.foodseg103_classes import ID2LABEL, FOODSEG103_CLASSES

class SemanticHybridEnsemblePipeline:
    """
    Hybrid Pipeline: MobileSAM (Masks) + SegFormer (Food Expert) + YOLO26 (Verifier).
    """

    def __init__(
        self,
        mobilesam_path: str = "mobile_sam.pt",
        yolo_path: str = "yoloe26s.pt",
        segformer_ckpt: str = "weights/segformer_foodseg103_best",
        sf_bias: float = 0.15,
        device: str = "cuda",
        use_yolo: bool = True,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.sf_bias = sf_bias

        print("=" * 60)
        print("  Semantic-Hybrid-Ensemble Pipeline — Initialising")
        print(f"  Device: {device}")
        print("=" * 60)

        # 1. Mask Generator (MobileSAM)
        self.mask_gen = MobileSAMMaskGenerator(model_path=mobilesam_path, device=device)

        # 2. Food Expert (SegFormer)
        self.closed_set = ClosedSetBranch(checkpoint_dir=segformer_ckpt, device=device)

        # 3. Object Verifier (YOLOE-26)
        self.use_yolo = use_yolo
        if use_yolo:
            print(f"[Verifier] Loading YOLOE-26 from {yolo_path} …")
            self.yolo = YOLOE(yolo_path)
            self.yolo.to(device)
            # Set text prompts for the 103 food classes (excluding background)
            self.yolo.set_classes(FOODSEG103_CLASSES[1:])
            print(f"[Verifier] YOLOE-26 loaded with {len(FOODSEG103_CLASSES)-1} food classes.")
        else:
            print("[Verifier] YOLOE verification disabled (not loading into VRAM).")
            self.yolo = None

        print("=" * 60)
        print("  Hybrid Ensemble ready.")
        print("=" * 60)

    def run(self, image, use_masks: bool = True, sp_prompt: bool = False, n_superpixels: int = 200, no_yolo: bool = False, verbose: bool = True):
        if not hasattr(self, "_diag_done"):
            print(f"\n[Diagnostics] Verifying CUDA usage:")
            # Check SegFormer
            sf_dev = next(self.closed_set.model.parameters()).device
            print(f"  - SegFormer Device: {sf_dev}")
            # Check YOLO
            if self.yolo:
                yolo_dev = next(self.yolo.model.parameters()).device
                print(f"  - YOLOE Device: {yolo_dev}")
            else:
                print("  - YOLOE Device: Disabled")
            # Check MobileSAM
            sam_dev = next(self.mask_gen.model.model.parameters()).device
            print(f"  - MobileSAM Device: {sam_dev}")
            print("-" * 30)
            self._diag_done = True
        """
        Run the hybrid pipeline.
        
        Args:
            image: PIL.Image or numpy array.
            use_masks: If True, use MobileSAM for mask-based fusion.
                       If False, use direct pixel-level fusion (Faster).
            sp_prompt: If True, use SLIC superpixel centroids as prompts for MobileSAM.
            n_superpixels: Number of superpixels to generate if sp_prompt is True.
            verbose: Whether to print detailed logs.
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
            np_image = image
        else:
            pil_image = image
            np_image = np.array(image)

        h, w = np_image.shape[:2]

        # ── Stage 1: MobileSAM Masks (Optional) ──────────────────────
        masks, mask_scores = [], []
        t_mask = 0
        if use_masks:
            t0 = time.time()
            masks, mask_scores = self.mask_gen.generate_masks(
                pil_image, 
                use_superpixels=sp_prompt, 
                n_superpixels=n_superpixels
            )
            t_mask = time.time() - t0
            mode_str = "Superpixel" if sp_prompt else "Everything"
            if verbose:
                print(f"\n[Stage 1] MobileSAM ({mode_str}) found {len(masks)} regions in {t_mask:.3f}s")

        # ── Stage 2: SegFormer (The Food Specialist) ──────────────────
        t0 = time.time()
        prob_map_cs = self.closed_set.predict_semantic_map(pil_image)
        t_cs = time.time() - t0
        if verbose:
            print(f"[Stage 2] SegFormer analysis in {t_cs:.3f}s")

        # ── Stage 3: YOLOE-26 (The Open-Vocabulary Verifier) ────────────
        yolo_dets = []
        t_yolo = 0
        if not no_yolo and self.use_yolo:
            t0 = time.time()
            yolo_results = self.yolo.predict(
                pil_image, 
                device=self.device, 
                verbose=False
            )
            t_yolo = time.time() - t0
            if verbose:
                print(f"[Stage 3] YOLO26 verification in {t_yolo:.3f}s")

            if yolo_results and yolo_results[0].boxes is not None:
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                confs = yolo_results[0].boxes.conf.cpu().numpy()
                cls_ids = yolo_results[0].boxes.cls.cpu().numpy()
                names = yolo_results[0].names
                for b, c, i in zip(boxes, confs, cls_ids):
                    yolo_dets.append({'box': b, 'conf': c, 'name': names[int(i)]})
        else:
            if verbose:
                print("[Stage 3] YOLOE verification skipped (--no_yolo).")

        # ── Stage 4: Fusion ──────────────────────────────────────────
        t0 = time.time()
        
        if not use_masks:
            # --- PIXEL-LEVEL FUSION ---
            # 1. Start with SegFormer's best guess for every pixel
            final_map = prob_map_cs.argmax(dim=-1).cpu().numpy().astype(np.int32)
            sf_conf_map = prob_map_cs.max(dim=-1).values.cpu().numpy()
            
            # 2. Use YOLO to override pixels where it is more confident
            for det in yolo_dets:
                db = det['box'].astype(int)
                yolo_conf = det['conf']
                
                # Map YOLO name to ID
                yolo_name = det['name'].lower().replace(" ", "")
                yolo_id = -1
                for fid, fname in ID2LABEL.items():
                    if fname.lower().replace(" ", "") == yolo_name:
                        yolo_id = fid
                        break
                
                if yolo_id != -1:
                    # Check confidence in this region
                    region_sf_conf = np.mean(sf_conf_map[db[1]:db[3], db[0]:db[2]])
                    # YOLO only overrules if it's significantly more confident
                    if yolo_conf > (region_sf_conf + self.sf_bias):
                        # Overrule pixels in this box
                        final_map[db[1]:db[3], db[0]:db[2]] = yolo_id
                        if verbose:
                            print(f" [!] Pixel-Override | {det['name']} (YOLO: {yolo_conf:.2f} > SF: {region_sf_conf:.2f} + {self.sf_bias})")
            
            t_fuse = time.time() - t0
            if verbose:
                print(f"[Stage 4] Pixel-Fusion in {t_fuse:.3f}s")
            
            timing = {
                "stage1_mask": t_mask,
                "stage2_sf": t_cs,
                "stage3_yolo": t_yolo,
                "stage4_fuse": t_fuse,
                "total": t_mask + t_cs + t_yolo + t_fuse
            }
            return final_map, [], [], timing

        else:
            # --- MASK-BASED FUSION (Existing) ---
            final_labels = []
            labels_info = []
            
            for i, (mask, score) in enumerate(zip(masks, mask_scores)):
                sf_id, sf_conf = self.closed_set.label_mask(prob_map_cs, mask)
                final_id = sf_id
                final_name = ID2LABEL.get(sf_id, "unknown")
                method = "SegFormer"

                yolo_match = None
                mask_coords = np.argwhere(mask)
                if mask_coords.size > 0:
                    y1, x1 = mask_coords.min(axis=0)
                    y2, x2 = mask_coords.max(axis=0)
                    cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
                    
                    for det in yolo_dets:
                        db = det['box']
                        if db[0] <= cx <= db[2] and db[1] <= cy <= db[3]:
                            yolo_name = det['name'].lower().replace(" ", "")
                            yolo_id = -1
                            for fid, fname in ID2LABEL.items():
                                if fname.lower().replace(" ", "") == yolo_name:
                                    yolo_id = fid
                                    break
                            
                            if yolo_id != -1 and det['conf'] > (sf_conf + self.sf_bias):
                                final_id = yolo_id
                                final_name = ID2LABEL[yolo_id]
                                method = f"YOLO-Overrule ({det['conf']:.2f} > {sf_conf:.2f} + {self.sf_bias})"
                                yolo_match = det
                            break

                labels_info.append({"mask_idx": i, "label": final_name, "method": method})
                final_labels.append(final_id)
                if verbose and i < 15:
                    overrule = " [!] " if "Overrule" in method else "     "
                    print(f"{overrule} Mask {i:3d} | {final_name:20s} (via {method})")

            final_map = build_semantic_map((h, w), masks, final_labels)
            t_fuse = time.time() - t0
            if verbose:
                print(f"[Stage 4] Mask-Fusion in {t_fuse:.3f}s")
            
            timing = {
                "stage1_mask": t_mask,
                "stage2_sf": t_cs,
                "stage3_yolo": t_yolo,
                "stage4_fuse": t_fuse,
                "total": t_mask + t_cs + t_yolo + t_fuse
            }
            return final_map, masks, labels_info, timing

def main():
    parser = argparse.ArgumentParser(description="Hybrid Ensemble Pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out_dir", default="output_hybrid")
    parser.add_argument("--model", type=str, default="s", choices=["n", "s", "m", "l", "x"], help="YOLOE-26 model size")
    parser.add_argument("--no_masks", action="store_true", help="Disable MobileSAM and use pixel-level fusion")
    parser.add_argument("--sp_prompt", action="store_true", help="Use superpixel centroids as prompts for MobileSAM")
    parser.add_argument("--n_superpixels", type=int, default=200, help="Number of superpixels to generate")
    parser.add_argument("--no_yolo", action="store_true", help="Disable YOLOE verification stage")
    parser.add_argument("--sf_bias", type=float, default=0.15, help="SegFormer confidence bias")
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
    
    image = Image.open(args.image).convert("RGB")
    seg_map, masks, labels_info, timing = pipe.run(
        image, 
        use_masks=(not args.no_masks), 
        sp_prompt=args.sp_prompt,
        n_superpixels=args.n_superpixels,
        no_yolo=args.no_yolo,
        verbose=True
    )

    basename = os.path.splitext(os.path.basename(args.image))[0]
    colour_map = colourise_segmentation(seg_map)
    Image.fromarray(colour_map).save(os.path.join(args.out_dir, f"{basename}_segmap.png"))
    overlay = overlay_segmentation(np.array(image), seg_map, alpha=0.5)
    Image.fromarray(overlay).save(os.path.join(args.out_dir, f"{basename}_overlay.png"))

    print(f"\n[Saved] Results written to {args.out_dir}/")

if __name__ == "__main__":
    main()
