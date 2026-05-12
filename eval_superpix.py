"""
Evaluation script for the Superpixel variant of Semantic-EfficientSAM3.

Identical metrics to evaluate.py but uses the SuperpixelPipeline and saves
all outputs to `evaluation_metrics_superpix/` for side-by-side comparison.

Usage:
    python eval_superpix.py --data_dir data/FoodSeg103 --split test
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_superpix import SuperpixelPipeline
from utils.data_loader import FoodSeg103Dataset, overlay_segmentation, colourise_segmentation
from configs.foodseg103_classes import FOODSEG103_CLASSES, NUM_CLASSES


# ═══════════════════════════════════════════════════════════════════════════
# Metrics Calculation
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics_batch(pred: np.ndarray, target: np.ndarray, num_classes: int):
    """
    Compute raw intersection, union, and pixel counts for a single image.
    Ignores target pixels with value 255.
    """
    valid = target != 255
    pred_valid = pred[valid]
    target_valid = target[valid]

    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    target_area = np.zeros(num_classes)
    pred_area = np.zeros(num_classes)

    for c in range(num_classes):
        p = pred_valid == c
        t = target_valid == c
        intersection[c] = np.logical_and(p, t).sum()
        union[c] = np.logical_or(p, t).sum()
        target_area[c] = t.sum()
        pred_area[c] = p.sum()

    correct_pixels = (pred_valid == target_valid).sum()
    total_pixels = valid.sum()

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    mask = (target_valid < num_classes) & (pred_valid < num_classes)
    np.add.at(confusion, (pred_valid[mask], target_valid[mask]), 1)

    return intersection, union, target_area, pred_area, correct_pixels, total_pixels, confusion


# ═══════════════════════════════════════════════════════════════════════════
# Plotting Helpers
# ═══════════════════════════════════════════════════════════════════════════

def plot_per_class_metric(metric_values, metric_name, out_path, top_n=30):
    """Horizontal bar chart for top N classes of a specific metric."""
    valid_mask = ~np.isnan(metric_values)
    class_names = [FOODSEG103_CLASSES[i] for i in range(NUM_CLASSES) if valid_mask[i]]
    values = metric_values[valid_mask] * 100

    if len(values) == 0:
        return

    order = np.argsort(values)[::-1][:top_n]
    class_names = [class_names[i] for i in order]
    values = values[order]

    fig, ax = plt.subplots(figsize=(10, max(6, len(class_names) * 0.35)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(class_names)))
    y_pos = np.arange(len(class_names))

    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric_name} (%)", fontsize=13)
    ax.set_title(f"Per-Class {metric_name} (Top {len(class_names)} — Superpixel)",
                 fontsize=15, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x", alpha=0.3)

    for i, v in enumerate(values):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] {out_path}")


def plot_confusion_matrix(confusion, out_path, top_n=25):
    """Normalised confusion matrix heatmap for the most-frequent classes."""
    gt_counts = confusion.sum(axis=0)
    top_indices = np.argsort(gt_counts)[::-1][:top_n]

    sub = confusion[np.ix_(top_indices, top_indices)].astype(float)
    row_sums = sub.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    sub_norm = sub / row_sums

    labels = [FOODSEG103_CLASSES[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sub_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted Class", fontsize=13)
    ax.set_ylabel("Ground Truth Class", fontsize=13)
    ax.set_title(f"Normalised Confusion Matrix — Superpixel (Top {top_n})",
                 fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] {out_path}")


def plot_latency_histogram(latencies, out_path):
    """Histogram of inference latencies."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latencies, bins=30, color="#FF9800", edgecolor="black", alpha=0.8)
    ax.axvline(np.mean(latencies), color="red", linestyle="dashed", linewidth=2,
               label=f"Mean: {np.mean(latencies):.3f}s")
    ax.set_xlabel("Inference Time (seconds)", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Inference Latency Distribution — Superpixel", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] {out_path}")


def save_sample_predictions(images, gt_maps, pred_maps, out_dir, n=6):
    """Save a grid of sample predictions: Image | Ground Truth | Prediction."""
    n = min(n, len(images))
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img_np = np.array(images[i])
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Input Image", fontsize=12, fontweight="bold")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay_segmentation(img_np, gt_maps[i], 0.6))
        axes[i, 1].set_title("Ground Truth", fontsize=12, fontweight="bold")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay_segmentation(img_np, pred_maps[i], 0.6))
        axes[i, 2].set_title("Prediction (Superpixel)", fontsize=12, fontweight="bold")
        axes[i, 2].axis("off")

    fig.suptitle("Superpixel Pipeline — Sample Predictions", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "sample_predictions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main Evaluation Loop
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # Load dataset
    dataset = FoodSeg103Dataset(root=args.data_dir, split=args.split)
    # Slice dataset if max_samples is set
    num_to_eval = len(dataset)
    if args.max_samples is not None:
        num_to_eval = min(args.max_samples, num_to_eval)

    print(f"Evaluating on {num_to_eval} / {len(dataset)} images from {args.data_dir}/{args.split}")

    # Map shorthand to weights and model names
    model_map = {
        "s": ("weights/efficient_sam3_efficientvit_s.pt", "b0"),
        "m": ("weights/efficient_sam3_efficientvit_m.pt", "b1"),
        "l": ("weights/efficient_sam3_efficientvit_l.pt", "b2"),
    }
    ckpt_path, model_name = model_map[args.model]

    # Build pipeline
    pipe = SuperpixelPipeline(
        mask_generator_ckpt=ckpt_path,
        mask_model_name=model_name,
        segformer_ckpt=args.segformer_ckpt,
        n_superpixels=args.n_superpixels,
        compactness=args.compactness,
        refine_edges=args.refine_edges,
        device=args.device,
    )

    # Accumulators
    total_intersection = np.zeros(NUM_CLASSES)
    total_union = np.zeros(NUM_CLASSES)
    total_target_area = np.zeros(NUM_CLASSES)
    total_pred_area = np.zeros(NUM_CLASSES)
    total_correct = 0
    total_pixels = 0
    total_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    latencies = []
    sample_images, sample_gt, sample_pred = [], [], []

    for i in tqdm(range(num_to_eval), desc="Evaluating (Superpixel)"):
        sample = dataset[i]
        image = sample["image"]
        gt_mask = sample["mask"]

        if gt_mask is None:
            continue

        # Run pipeline
        t0 = time.time()
        pred_map, _, _, _, _ = pipe.run(image, verbose=False)
        latencies.append(time.time() - t0)

        # Compute metrics for this image
        inter, uni, t_area, p_area, correct, t_pix, conf = compute_metrics_batch(
            pred_map, gt_mask, NUM_CLASSES
        )

        total_intersection += inter
        total_union += uni
        total_target_area += t_area
        total_pred_area += p_area
        total_correct += correct
        total_pixels += t_pix
        total_confusion += conf

        # Collect sample images for visualisations
        if len(sample_images) < args.save_n_visuals:
            sample_images.append(image)
            sample_gt.append(gt_mask)
            sample_pred.append(pred_map)

    # ── Aggregate Metrics ──────────────────────────────────────────────
    per_class_iou = np.full(NUM_CLASSES, np.nan)
    per_class_acc = np.full(NUM_CLASSES, np.nan)
    per_class_dice = np.full(NUM_CLASSES, np.nan)

    for c in range(NUM_CLASSES):
        if total_union[c] > 0:
            per_class_iou[c] = total_intersection[c] / total_union[c]
        if total_target_area[c] > 0:
            per_class_acc[c] = total_intersection[c] / total_target_area[c]
        if (total_pred_area[c] + total_target_area[c]) > 0:
            per_class_dice[c] = (2 * total_intersection[c]) / (total_pred_area[c] + total_target_area[c])

    miou = float(np.nanmean(per_class_iou))
    macc = float(np.nanmean(per_class_acc))
    mdice = float(np.nanmean(per_class_dice))
    pixel_acc = total_correct / total_pixels if total_pixels > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"  Superpixel Evaluation Results  ({len(latencies)} valid images)")
    print("=" * 60)
    print(f"  mIoU:              {miou * 100:.2f}%")
    print(f"  mAcc (Class Acc):  {macc * 100:.2f}%")
    print(f"  Mean Dice:         {mdice * 100:.2f}%")
    print(f"  Pixel Accuracy:    {pixel_acc * 100:.2f}%")
    print(f"  Avg Latency:       {np.mean(latencies):.3f}s  (FPS: {1.0 / np.mean(latencies):.1f})")
    print("=" * 60)

    # ── Generate Plots & Reports ───────────────────────────────────────
    print("\nGenerating presentation charts...")

    plot_per_class_metric(per_class_iou, "IoU",
                          os.path.join(args.out_dir, "per_class_iou.png"))
    plot_per_class_metric(per_class_dice, "Dice Score",
                          os.path.join(args.out_dir, "per_class_dice.png"))
    plot_per_class_metric(per_class_acc, "Class Accuracy",
                          os.path.join(args.out_dir, "per_class_acc.png"))

    plot_confusion_matrix(total_confusion, os.path.join(args.out_dir, "confusion_matrix.png"))
    plot_latency_histogram(latencies, os.path.join(args.out_dir, "latency_histogram.png"))

    save_sample_predictions(sample_images, sample_gt, sample_pred,
                            args.out_dir, n=args.save_n_visuals)

    # ── Text Report ────────────────────────────────────────────────────
    report_path = os.path.join(args.out_dir, "evaluation_summary.txt")
    with open(report_path, "w") as f:
        f.write("Semantic-EfficientSAM3 (Superpixel) — Evaluation Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Dataset Split:    FoodSeg103 ({args.split})\n")
        f.write(f"Images Evaluated: {len(latencies)}\n")
        f.write(f"SLIC Superpixels: {args.n_superpixels}\n")
        f.write(f"SLIC Compactness: {args.compactness}\n")
        f.write(f"Edge Refinement:  {'ON' if args.refine_edges else 'OFF'}\n")
        f.write(f"mIoU:             {miou * 100:.2f}%\n")
        f.write(f"Mean Class Acc:   {macc * 100:.2f}%\n")
        f.write(f"Mean Dice:        {mdice * 100:.2f}%\n")
        f.write(f"Pixel Accuracy:   {pixel_acc * 100:.2f}%\n")
        f.write(f"Avg Latency:      {np.mean(latencies):.3f}s\n")
        f.write(f"Avg FPS:          {1.0 / np.mean(latencies):.1f}\n\n")

        f.write("Per-Class Metrics:\n")
        f.write(f"{'Class':<25s}  {'IoU (%)':>8}  {'Dice (%)':>8}  {'Acc (%)':>8}\n")
        f.write("-" * 55 + "\n")
        for c in range(NUM_CLASSES):
            if not np.isnan(per_class_iou[c]):
                f.write(f"{FOODSEG103_CLASSES[c]:<25s}  {per_class_iou[c]*100:8.2f}  "
                        f"{per_class_dice[c]*100:8.2f}  {per_class_acc[c]*100:8.2f}\n")

    # ── JSON Export ────────────────────────────────────────────────────
    results_json = {
        "pipeline": "superpixel" + ("_refined" if args.refine_edges else ""),
        "slic_n_superpixels": args.n_superpixels,
        "slic_compactness": args.compactness,
        "refine_edges": args.refine_edges,
        "global_metrics": {
            "mIoU": miou,
            "mAcc": macc,
            "mDice": mdice,
            "pixel_accuracy": pixel_acc,
            "avg_latency": np.mean(latencies),
            "fps": 1.0 / np.mean(latencies),
        },
        "per_class_metrics": {
            FOODSEG103_CLASSES[c]: {
                "IoU": float(per_class_iou[c]) if not np.isnan(per_class_iou[c]) else None,
                "Dice": float(per_class_dice[c]) if not np.isnan(per_class_dice[c]) else None,
                "Accuracy": float(per_class_acc[c]) if not np.isnan(per_class_acc[c]) else None,
            } for c in range(NUM_CLASSES)
        },
    }

    json_path = os.path.join(args.out_dir, "evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\n[Saved] All superpixel evaluation artifacts saved to: {args.out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Semantic-EfficientSAM3 (Superpixel) on FoodSeg103"
    )
    parser.add_argument("--data_dir", default="data/FoodSeg103", help="FoodSeg103 root directory")
    parser.add_argument("--split", default="validation", help="Dataset split to evaluate")
    parser.add_argument("--out_dir", default="evaluation_metrics_superpix",
                        help="Directory to save results")
    parser.add_argument("--model", type=str, default="s", choices=["s", "m", "l"], help="EfficientSAM3 model size (s, m, l).")
    parser.add_argument("--segformer_ckpt", default="weights/segformer_foodseg103_best")
    parser.add_argument("--n_superpixels", type=int, default=256,
                        help="Target number of SLIC superpixels")
    parser.add_argument("--compactness", type=float, default=10.0,
                        help="SLIC compactness parameter")
    parser.add_argument("--refine_edges", action="store_true",
                        help="Enable post-processing edge refinement with fine superpixels")

    parser.add_argument("--save_n_visuals", type=int, default=6,
                        help="Number of sample predictions to save")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate (None = all)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    evaluate(args)
