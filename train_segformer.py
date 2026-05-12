"""
SegFormer-B0 Training Script for FoodSeg103
============================================

Trains a SegFormer-B0 (pretrained on ADE20K) on the FoodSeg103 dataset and
produces comprehensive training statistics, charts, and visualisations
suitable for a presentation:

  1. Training & validation loss curves
  2. mIoU & pixel accuracy curves per epoch
  3. Per-class IoU bar chart (final epoch)
  4. Normalised confusion matrix heatmap
  5. Sample prediction visualisations (side-by-side: image / GT / pred)
  6. Learning-rate schedule plot
  7. Summary statistics saved as a text report

All outputs are saved to  training_output/

Usage:
    python train_segformer.py
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from configs.foodseg103_classes import FOODSEG103_CLASSES, NUM_CLASSES

# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class FoodSeg103Dataset(Dataset):
    def __init__(self, hf_dataset_split, processor):
        self.dataset = hf_dataset_split
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        segmentation_map = np.array(item['label'])

        inputs = self.processor(
            images=image,
            segmentation_maps=segmentation_map,
            return_tensors="pt",
        )
        # Squeeze the batch dimension added by the processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs


# ═══════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(pred_map: np.ndarray, gt_map: np.ndarray, num_classes: int):
    """
    Compute per-class IoU, mean IoU, and pixel accuracy between a single
    predicted segmentation map and its ground-truth.

    Returns:
        per_class_iou  (ndarray): shape (num_classes,), NaN for absent classes
        miou           (float)
        pixel_acc      (float)
        intersection   (ndarray): shape (num_classes,)
        union          (ndarray): shape (num_classes,)
        correct_pixels (int)
        total_pixels   (int)
    """
    per_class_iou = np.full(num_classes, np.nan)
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)

    for c in range(num_classes):
        p = pred_map == c
        g = gt_map == c
        inter = np.logical_and(p, g).sum()
        uni = np.logical_or(p, g).sum()
        intersection[c] = inter
        union[c] = uni
        if uni > 0:
            per_class_iou[c] = inter / uni

    valid = ~np.isnan(per_class_iou)
    miou = float(np.nanmean(per_class_iou)) if valid.any() else 0.0

    correct = (pred_map == gt_map).sum()
    total = pred_map.size
    pixel_acc = correct / total if total > 0 else 0.0

    return per_class_iou, miou, pixel_acc, intersection, union, int(correct), int(total)


# ═══════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_losses, val_losses, out_path):
    """Training vs validation loss per epoch."""
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, "o-", color="#2196F3", linewidth=2, label="Train Loss")
    ax.plot(epochs, val_losses, "s-", color="#F44336", linewidth=2, label="Val Loss")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Training & Validation Loss", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] {out_path}")


def plot_miou_accuracy_curves(mious, pixel_accs, out_path):
    """mIoU and pixel accuracy per epoch (dual y-axis)."""
    epochs = range(1, len(mious) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_iou = "#4CAF50"
    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("mIoU (%)", color=color_iou, fontsize=13)
    ax1.plot(epochs, [m * 100 for m in mious], "o-", color=color_iou, linewidth=2, label="mIoU")
    ax1.tick_params(axis="y", labelcolor=color_iou)

    ax2 = ax1.twinx()
    color_acc = "#FF9800"
    ax2.set_ylabel("Pixel Accuracy (%)", color=color_acc, fontsize=13)
    ax2.plot(epochs, [a * 100 for a in pixel_accs], "s-", color=color_acc, linewidth=2, label="Pixel Acc")
    ax2.tick_params(axis="y", labelcolor=color_acc)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=12)

    ax1.set_title("Validation mIoU & Pixel Accuracy", fontsize=15, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] {out_path}")


def plot_per_class_iou(per_class_iou, out_path, top_n=30):
    """Horizontal bar chart of per-class IoU (top N non-NaN classes)."""
    valid_mask = ~np.isnan(per_class_iou)
    class_names = [FOODSEG103_CLASSES[i] for i in range(NUM_CLASSES) if valid_mask[i]]
    values = per_class_iou[valid_mask] * 100

    # Sort descending and take top_n
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
    ax.set_xlabel("IoU (%)", fontsize=13)
    ax.set_title(f"Per-Class IoU (Top {len(class_names)} Classes)", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x", alpha=0.3)

    # Value annotations
    for i, v in enumerate(values):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] {out_path}")


def plot_confusion_matrix(confusion, out_path, top_n=25):
    """
    Normalised confusion matrix heatmap for the most-frequent classes.
    """
    # Find the top_n classes by total ground-truth pixels
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
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Ground Truth", fontsize=13)
    ax.set_title(f"Normalised Confusion Matrix (Top {top_n} Classes)", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] {out_path}")


def plot_lr_schedule(lr_history, out_path):
    """Learning rate over training steps."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lr_history, color="#9C27B0", linewidth=1.5)
    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Learning Rate", fontsize=13)
    ax.set_title("Learning Rate Schedule", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] {out_path}")


def _random_palette(n, seed=42):
    rng = np.random.RandomState(seed)
    p = rng.randint(0, 255, (n, 3), dtype=np.uint8)
    p[0] = [0, 0, 0]
    return p

_PALETTE = _random_palette(NUM_CLASSES)

def colourise(seg_map):
    return _PALETTE[seg_map.ravel()].reshape(*seg_map.shape[:2], 3)


def save_sample_predictions(images, gt_maps, pred_maps, out_dir, n=6):
    """
    Save a grid of sample predictions: Original | Ground Truth | Prediction.
    """
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img_np = np.array(images[i])
        gt_colour = colourise(gt_maps[i])
        pred_colour = colourise(pred_maps[i])

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Input Image", fontsize=12, fontweight="bold")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_colour)
        axes[i, 1].set_title("Ground Truth", fontsize=12, fontweight="bold")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_colour)
        axes[i, 2].set_title("Prediction", fontsize=12, fontweight="bold")
        axes[i, 2].axis("off")

    fig.suptitle("Sample Predictions", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "sample_predictions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Validation loop
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, dataloader, device, num_classes, collect_samples=False, max_samples=6):
    """
    Run validation and return aggregate metrics.

    Returns dict with keys:
        val_loss, miou, pixel_acc, per_class_iou,
        total_intersection, total_union, confusion,
        sample_images, sample_gt, sample_pred  (if collect_samples)
    """
    model.eval()
    running_loss = 0.0
    total_intersection = np.zeros(num_classes)
    total_union = np.zeros(num_classes)
    total_correct = 0
    total_pixels = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    sample_images, sample_gt, sample_pred = [], [], []

    for batch in tqdm(dataloader, desc="  Validating", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device).long()

        outputs = model(pixel_values=pixel_values, labels=labels)
        running_loss += outputs.loss.item()

        logits = outputs.logits  # (B, C, h', w')
        # Upsample logits to label size
        upsampled = F.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=True
        )
        preds = upsampled.argmax(dim=1).cpu().numpy()  # (B, H, W)
        gt = labels.cpu().numpy()                       # (B, H, W)

        for b in range(preds.shape[0]):
            p, g = preds[b], gt[b]
            # Ignore pixels with label 255 (unlabelled)
            valid = g != 255
            p_valid, g_valid = p[valid], g[valid]

            _, _, _, inter, uni, correct, total = compute_metrics(p_valid, g_valid, num_classes)
            total_intersection += inter
            total_union += uni
            total_correct += correct
            total_pixels += total

            # Accumulate confusion matrix
            mask = (g_valid < num_classes) & (p_valid < num_classes)
            np.add.at(confusion, (p_valid[mask], g_valid[mask]), 1)

            # Collect sample images for visualisation
            if collect_samples and len(sample_images) < max_samples:
                # Denormalise pixel_values back to image
                img_tensor = pixel_values[b].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_denorm = (img_tensor * std + mean).clamp(0, 1)
                img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                sample_images.append(img_np)
                sample_gt.append(gt[b])
                sample_pred.append(preds[b])

    # Aggregate
    per_class_iou = np.full(num_classes, np.nan)
    for c in range(num_classes):
        if total_union[c] > 0:
            per_class_iou[c] = total_intersection[c] / total_union[c]

    valid_ious = per_class_iou[~np.isnan(per_class_iou)]
    miou = float(np.mean(valid_ious)) if len(valid_ious) > 0 else 0.0
    pixel_acc = total_correct / total_pixels if total_pixels > 0 else 0.0
    val_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    result = {
        "val_loss": val_loss,
        "miou": miou,
        "pixel_acc": pixel_acc,
        "per_class_iou": per_class_iou,
        "total_intersection": total_intersection,
        "total_union": total_union,
        "confusion": confusion,
    }
    if collect_samples:
        result["sample_images"] = sample_images
        result["sample_gt"] = sample_gt
        result["sample_pred"] = sample_pred

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════════════════════

def train():
    OUT_DIR = "training_output"
    WEIGHTS_DIR = "weights"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # ── Load dataset ───────────────────────────────────────────────────
    print("Loading FoodSeg103 dataset from HuggingFace …")
    dataset = load_dataset("EduardoPacheco/FoodSeg103")

    num_labels = NUM_CLASSES  # 104

    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    processor.do_reduce_labels = False

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    train_dataset = FoodSeg103Dataset(dataset["train"], processor)
    val_split = "validation" if "validation" in dataset else "test"
    val_dataset = FoodSeg103Dataset(dataset[val_split], processor)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}  (split: '{val_split}')")
    print(f"  Num classes:   {num_labels}")

    # ── Optimiser & scheduler ──────────────────────────────────────────
    from transformers import get_cosine_schedule_with_warmup

    EPOCHS = 12
    ENCODER_LR = 6e-5
    DECODER_LR = 6e-4

    # Differential Learning Rates
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if "decode_head" in name:
            decoder_params.append(param)
        else:
            encoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": ENCODER_LR},
        {"params": decoder_params, "lr": DECODER_LR},
    ], weight_decay=0.01)

    total_steps = EPOCHS * len(train_dataloader)
    warmup_steps = int(total_steps * 0.1)  # 10% warmup phase

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\nStarting training on {device} for {EPOCHS} epochs …")
    print(f"  Optimizer: AdamW (Encoder LR={ENCODER_LR}, Decoder LR={DECODER_LR})")
    print(f"  Scheduler: Cosine w/ Warmup ({warmup_steps} warmup steps → {total_steps} total steps)")

    # ── History accumulators ───────────────────────────────────────────
    history = {
        "train_loss": [],
        "val_loss": [],
        "miou": [],
        "pixel_acc": [],
        "per_class_iou": [],
        "lr": [],
    }
    lr_per_step = []
    best_miou = 0.0
    training_start = time.time()

    # ══════════════════════════════════════════════════════════════════
    #  TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════
    try:
        for epoch in range(EPOCHS):
            model.train()
            running_train_loss = 0.0
            epoch_start = time.time()

            progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
            for batch in progress:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device).long()

                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                running_train_loss += loss.item()
                # Record the *decoder* learning rate as it is the primary one
                lr_per_step.append(optimizer.param_groups[1]["lr"])
                progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr_per_step[-1]:.2e}"})

            avg_train_loss = running_train_loss / len(train_dataloader)

            # ── Validation ─────────────────────────────────────────────────
            collect = (epoch == EPOCHS - 1)  # collect samples on final epoch
            val_result = validate(model, val_dataloader, device, num_labels,
                                  collect_samples=collect, max_samples=6)

            epoch_time = time.time() - epoch_start

            # Record history
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_result["val_loss"])
            history["miou"].append(val_result["miou"])
            history["pixel_acc"].append(val_result["pixel_acc"])
            history["per_class_iou"].append(val_result["per_class_iou"].tolist())
            history["lr"].append(optimizer.param_groups[0]["lr"])

            print(
                f"  Epoch {epoch+1:2d} │ "
                f"Train Loss: {avg_train_loss:.4f} │ "
                f"Val Loss: {val_result['val_loss']:.4f} │ "
                f"mIoU: {val_result['miou']*100:.2f}% │ "
                f"Pixel Acc: {val_result['pixel_acc']*100:.2f}% │ "
                f"Time: {epoch_time:.0f}s"
            )

            # Save checkpoint
            ckpt_dir = os.path.join(WEIGHTS_DIR, f"segformer_foodseg103_epoch_{epoch+1}")
            model.save_pretrained(ckpt_dir)

            # Track best
            if val_result["miou"] > best_miou:
                best_miou = val_result["miou"]
                best_ckpt = os.path.join(WEIGHTS_DIR, "segformer_foodseg103_best")
                model.save_pretrained(best_ckpt)
                print(f"  ★ New best mIoU: {best_miou*100:.2f}% → saved to {best_ckpt}")

    except KeyboardInterrupt:
        print("\n\n[!] Training stopped early by user (Ctrl+C).")
        EPOCHS = len(history["train_loss"])
        if EPOCHS == 0:
            print("No epochs were completed. Exiting without generating reports.")
            return
        print(f"Generating reports for the {EPOCHS} completed epochs...")

    total_time = time.time() - training_start
    print(f"\nTraining complete in {total_time/60:.1f} minutes.")

    # ══════════════════════════════════════════════════════════════════
    #  GENERATE ALL PLOTS & REPORTS
    # ══════════════════════════════════════════════════════════════════
    print("\nGenerating training report …")

    # 1. Loss curves
    plot_loss_curves(history["train_loss"], history["val_loss"],
                     os.path.join(OUT_DIR, "loss_curves.png"))

    # 2. mIoU & accuracy curves
    plot_miou_accuracy_curves(history["miou"], history["pixel_acc"],
                              os.path.join(OUT_DIR, "miou_accuracy_curves.png"))

    # 3. Per-class IoU (final epoch)
    final_iou = np.array(history["per_class_iou"][-1])
    plot_per_class_iou(final_iou, os.path.join(OUT_DIR, "per_class_iou.png"))

    # 4. Confusion matrix (final epoch)
    plot_confusion_matrix(val_result["confusion"],
                          os.path.join(OUT_DIR, "confusion_matrix.png"))

    # 5. Learning rate schedule
    plot_lr_schedule(lr_per_step, os.path.join(OUT_DIR, "lr_schedule.png"))

    # 6. Sample predictions (final epoch)
    if collect and val_result.get("sample_images"):
        save_sample_predictions(
            val_result["sample_images"],
            val_result["sample_gt"],
            val_result["sample_pred"],
            OUT_DIR,
        )

    # 7. Summary text report
    report_path = os.path.join(OUT_DIR, "training_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Semantic-EfficientSAM3 — SegFormer-B2 Training Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Dataset:          FoodSeg103 ({len(train_dataset)} train / {len(val_dataset)} val)\n")
        f.write(f"Model:            SegFormer-B2 (pretrained ADE20K → fine-tuned FoodSeg103)\n")
        f.write(f"Num Classes:      {num_labels}\n")
        f.write(f"Epochs:           {EPOCHS}\n")
        f.write(f"Batch Size:       8\n")
        f.write(f"Optimizer:        AdamW (Encoder LR={ENCODER_LR}, Decoder LR={DECODER_LR})\n")
        f.write(f"Scheduler:        Cosine w/ Warmup\n")
        f.write(f"Device:           {device}\n")
        f.write(f"Total Time:       {total_time/60:.1f} min\n\n")
        f.write(f"Best mIoU:        {best_miou*100:.2f}%\n")
        f.write(f"Final mIoU:       {history['miou'][-1]*100:.2f}%\n")
        f.write(f"Final Pixel Acc:  {history['pixel_acc'][-1]*100:.2f}%\n")
        f.write(f"Final Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Val Loss:   {history['val_loss'][-1]:.4f}\n\n")

        f.write("Per-Epoch Summary:\n")
        f.write(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'mIoU (%)':>10}  {'Pix Acc (%)':>12}\n")
        f.write("-" * 55 + "\n")
        for e in range(EPOCHS):
            f.write(
                f"{e+1:5d}  {history['train_loss'][e]:10.4f}  "
                f"{history['val_loss'][e]:10.4f}  "
                f"{history['miou'][e]*100:10.2f}  "
                f"{history['pixel_acc'][e]*100:12.2f}\n"
            )

        f.write("\n\nPer-Class IoU (Final Epoch):\n")
        f.write(f"{'Class':<25s}  {'IoU (%)':>8}\n")
        f.write("-" * 40 + "\n")
        for c in range(NUM_CLASSES):
            if not np.isnan(final_iou[c]):
                f.write(f"{FOODSEG103_CLASSES[c]:<25s}  {final_iou[c]*100:8.2f}\n")

    print(f"  [Report] {report_path}")

    # 8. Save raw history as JSON for further analysis
    json_path = os.path.join(OUT_DIR, "training_history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [JSON] {json_path}")

    print(f"\n✓ All outputs saved to {OUT_DIR}/")
    print(f"✓ Best checkpoint:  {best_ckpt}")


if __name__ == "__main__":
    train()
