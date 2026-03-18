"""
physics_visualize.py
====================
Visualisation for the physics-guided model:
  1. Training curves with per-VED-bin Dice
  2. VED-stratified performance bar chart
  3. Failure mode distribution vs VED (stacked bar)
  4. Comparison: standard UNet vs Physics-IRM UNet
  5. Dice vs τ per VED bin (threshold sensitivity)
"""

import csv
import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from scipy import ndimage

from physics_guided_train import (UNet, PBPhysicsDataset,
                                  compute_metrics, NUM_ENVS,
                                  IMAGE_DIR, LABEL_DIR, IMG_SIZE, BATCH_SIZE)
from ved_metadata import parse_filename

SAVE_PATH        = "pb_physics_best.pth"
BASELINE_PATH    = "pb_best_model.pth"      # your existing best model
PHYSICS_CSV      = "pb_physics_results.csv"
BASELINE_CSV     = "pb_sweep_results.csv"   # optional, for comparison
OUT_DIR          = "physics_outputs"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
regime_names     = {0: "low", 1: "stable", 2: "high"}
regime_colors    = {0: "#e74c3c", 1: "#2ecc71", 2: "#3498db"}


def load_model(path, dropout=0.3):
    model = UNet(bottleneck_dropout=dropout).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def get_val_loader():
    full_ds = PBPhysicsDataset(IMAGE_DIR, LABEL_DIR, img_size=IMG_SIZE, augment=False)
    n       = len(full_ds)
    random.seed(42)
    indices = list(range(n))
    random.shuffle(indices)
    val_idx = indices[int(n * 0.8):]
    return DataLoader(Subset(full_ds, val_idx),
                      batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# -------------------------------------------------------------------
# 1. Training curves (per-VED-bin Dice)
# -------------------------------------------------------------------
def plot_training_curves():
    epochs, train_l, val_l = [], [], []
    env_dice = {e: [] for e in range(NUM_ENVS)}

    with open(PHYSICS_CSV) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            train_l.append(float(row['train_loss']))
            val_l.append(float(row['val_loss']))
            for e in range(NUM_ENVS):
                key = f"dice_{regime_names[e]}"
                env_dice[e].append(float(row.get(key, 0)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_l, label='Train Loss', lw=2)
    axes[0].plot(epochs, val_l,   label='Val Loss',   lw=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('[Physics-IRM] Loss Curves'); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for e in range(NUM_ENVS):
        axes[1].plot(epochs, env_dice[e],
                     label=f'{regime_names[e]} VED',
                     color=regime_colors[e], lw=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Dice')
    axes[1].set_title('[Physics-IRM] Per-VED-Bin Dice')
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "physics_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")


# -------------------------------------------------------------------
# 2. VED-stratified bar chart
# -------------------------------------------------------------------
def plot_ved_stratified_bar(tau_dict=None):
    if tau_dict is None:
        tau_dict = {0: 0.5, 1: 0.5, 2: 0.5}

    val_loader = get_val_loader()
    model      = load_model(SAVE_PATH)

    env_dice = {e: [] for e in range(NUM_ENVS)}
    env_iou  = {e: [] for e in range(NUM_ENVS)}

    with torch.no_grad():
        for images, masks, env_ids, _ in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            for b in range(images.shape[0]):
                e   = env_ids[b].item()
                tau = tau_dict.get(e, 0.5)
                d, iou = compute_metrics(logits[b:b+1], masks[b:b+1], tau=tau)
                env_dice[e].append(d)
                env_iou[e].append(iou)

    fig, ax = plt.subplots(figsize=(8, 5))
    x       = np.arange(NUM_ENVS)
    w       = 0.35
    dice_m  = [np.mean(env_dice[e]) if env_dice[e] else 0 for e in range(NUM_ENVS)]
    iou_m   = [np.mean(env_iou[e])  if env_iou[e]  else 0 for e in range(NUM_ENVS)]

    bars1 = ax.bar(x - w/2, dice_m, w, label='Dice',
                   color=[regime_colors[e] for e in range(NUM_ENVS)], alpha=0.85)
    bars2 = ax.bar(x + w/2, iou_m,  w, label='IoU',
                   color=[regime_colors[e] for e in range(NUM_ENVS)], alpha=0.5,
                   hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{regime_names[e]}\nVED" for e in range(NUM_ENVS)])
    ax.set_ylabel('Score'); ax.set_ylim(0, 1)
    ax.set_title('[Physics-IRM] VED-Stratified Performance')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "physics_ved_bar.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")
    return dict(zip(range(NUM_ENVS), dice_m))


# -------------------------------------------------------------------
# 3. Failure mode distribution vs VED (stacked bar)
# -------------------------------------------------------------------
def plot_failure_mode_distribution(tau_dict=None):
    if tau_dict is None:
        tau_dict = {0: 0.5, 1: 0.5, 2: 0.5}

    val_loader = get_val_loader()
    model      = load_model(SAVE_PATH)

    # Counters: env → {FN, FP, over, under, correct}
    counts = {e: {"FN":0, "FP":0, "over":0, "under":0, "correct":0}
              for e in range(NUM_ENVS)}

    with torch.no_grad():
        for images, masks, env_ids, _ in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            for b in range(images.shape[0]):
                e    = env_ids[b].item()
                tau  = tau_dict.get(e, 0.5)
                pred = (torch.sigmoid(logits[b:b+1]) >= tau).float()
                gt   = masks[b:b+1]

                pred_np = pred[0,0].cpu().numpy()
                gt_np   = gt[0,0].cpu().numpy()

                gt_area   = gt_np.sum()
                pred_area = pred_np.sum()

                if gt_area == 0 and pred_area == 0:
                    counts[e]["correct"] += 1
                elif gt_area == 0 and pred_area > 0:
                    counts[e]["FP"] += 1
                elif gt_area > 0 and pred_area == 0:
                    counts[e]["FN"] += 1
                else:
                    ratio = pred_area / (gt_area + 1e-6)
                    if ratio < 0.5:
                        counts[e]["under"] += 1
                    elif ratio > 2.0:
                        counts[e]["over"] += 1
                    else:
                        counts[e]["correct"] += 1

    fig, ax = plt.subplots(figsize=(9, 5))
    categories = ["correct", "FN", "FP", "under", "over"]
    colors_cat  = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db", "#9b59b6"]
    x  = np.arange(NUM_ENVS)
    bottom = np.zeros(NUM_ENVS)

    for cat, col in zip(categories, colors_cat):
        vals = [counts[e][cat] for e in range(NUM_ENVS)]
        ax.bar(x, vals, bottom=bottom, label=cat, color=col, alpha=0.85)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{regime_names[e]}\n(env {e})" for e in range(NUM_ENVS)])
    ax.set_ylabel('Image count')
    ax.set_title('[Physics-IRM] Failure Mode Distribution by VED Regime')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "physics_failure_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")

    # Print table
    print("\n  Failure mode table:")
    header = f"{'Regime':<10}" + "".join(f"  {c:<10}" for c in categories)
    print(f"  {header}")
    for e in range(NUM_ENVS):
        row = f"  {regime_names[e]:<10}" + "".join(
            f"  {counts[e][c]:<10}" for c in categories)
        print(row)


# -------------------------------------------------------------------
# 4. Dice vs threshold per VED bin
# -------------------------------------------------------------------
def plot_dice_vs_tau():
    val_loader = get_val_loader()
    model      = load_model(SAVE_PATH)
    tau_grid   = np.arange(0.1, 0.91, 0.05)

    env_preds = {e: [] for e in range(NUM_ENVS)}
    env_masks = {e: [] for e in range(NUM_ENVS)}

    with torch.no_grad():
        for images, masks, env_ids, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            for b in range(images.shape[0]):
                e = env_ids[b].item()
                env_preds[e].append(logits[b:b+1].cpu())
                env_masks[e].append(masks[b:b+1])

    fig, ax = plt.subplots(figsize=(9, 5))
    for e in range(NUM_ENVS):
        if not env_preds[e]:
            continue
        preds_cat = torch.cat(env_preds[e])
        masks_cat = torch.cat(env_masks[e])
        dices = []
        for tau in tau_grid:
            d, _ = compute_metrics(preds_cat, masks_cat, tau=float(tau))
            dices.append(d)
        best_tau = tau_grid[np.argmax(dices)]
        ax.plot(tau_grid, dices,
                label=f"{regime_names[e]} VED (best τ={best_tau:.2f})",
                color=regime_colors[e], lw=2)
        ax.axvline(best_tau, color=regime_colors[e], linestyle='--', alpha=0.5)

    ax.set_xlabel('Threshold τ'); ax.set_ylabel('Dice')
    ax.set_title('[Physics-IRM] Dice vs τ per VED Bin')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.axvline(0.5, color='black', linestyle=':', label='τ=0.5 (default)')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "physics_dice_vs_tau.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")


# -------------------------------------------------------------------
# 5. Side-by-side: standard vs physics-IRM predictions
# -------------------------------------------------------------------
def plot_comparison(num_samples=6, tau_dict=None):
    if tau_dict is None:
        tau_dict = {0: 0.5, 1: 0.5, 2: 0.5}

    val_loader   = get_val_loader()
    model_phys   = load_model(SAVE_PATH, dropout=0.3)

    # Try to load baseline model
    baseline_available = os.path.exists(BASELINE_PATH)
    if baseline_available:
        model_base = load_model(BASELINE_PATH, dropout=0.0)
    else:
        model_base = None
        print("  (Baseline model not found – showing physics model only)")

    samples = []
    for images, masks, env_ids, veds in val_loader:
        for b in range(images.shape[0]):
            if len(samples) >= num_samples:
                break
            samples.append((images[b:b+1], masks[b:b+1],
                             env_ids[b].item(), veds[b].item()))
        if len(samples) >= num_samples:
            break

    cols   = 4 if baseline_available else 3
    titles = (["Input", "GT", "Baseline", "Physics-IRM"]
              if baseline_available else ["Input", "GT", "Physics-IRM"])

    fig, axes = plt.subplots(num_samples, cols,
                             figsize=(cols * 4, num_samples * 3.5))
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=12, fontweight='bold')

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i, (img, mask, env_id, ved) in enumerate(samples):
        img_dev = img.to(device)
        tau     = tau_dict.get(env_id, 0.5)

        with torch.no_grad():
            pred_phys = (torch.sigmoid(model_phys(img_dev)) >= tau).float()
            if model_base:
                pred_base = (torch.sigmoid(model_base(img_dev)) >= 0.5).float()

        img_vis  = (img[0] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        gt_vis   = mask[0, 0].numpy()
        ph_vis   = pred_phys[0, 0].cpu().numpy()

        axes[i, 0].imshow(img_vis); axes[i, 0].axis('off')
        axes[i, 0].set_xlabel(f"VED={ved:.1f} ({['low','stable','high'][env_id]})",
                              fontsize=8)

        axes[i, 1].imshow(gt_vis, cmap='Greens', vmin=0, vmax=1)
        axes[i, 1].axis('off')

        col = 2
        if baseline_available:
            ba_vis = pred_base[0, 0].cpu().numpy()
            axes[i, col].imshow(ba_vis, cmap='Reds', vmin=0, vmax=1)
            axes[i, col].axis('off')
            col += 1

        axes[i, col].imshow(ph_vis, cmap='Blues', vmin=0, vmax=1)
        axes[i, col].axis('off')
        axes[i, col].set_xlabel(f"τ={tau:.2f}", fontsize=8)

    plt.suptitle('[Physics-IRM vs Baseline] VED-stratified predictions', fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "physics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")


# ===================================================================
# Main
# ===================================================================
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    # Default tau dict (replace with output of tune_thresholds after training)
    tau_dict = {0: 0.4, 1: 0.5, 2: 0.5}   # low VED → lower tau to catch weak sigs

    print("=== 1. Training Curves ===")
    plot_training_curves()

    print("\n=== 2. VED-Stratified Bar Chart ===")
    plot_ved_stratified_bar(tau_dict)

    print("\n=== 3. Failure Mode Distribution ===")
    plot_failure_mode_distribution(tau_dict)

    print("\n=== 4. Dice vs Threshold ===")
    plot_dice_vs_tau()

    print("\n=== 5. Baseline vs Physics-IRM Comparison ===")
    plot_comparison(num_samples=6, tau_dict=tau_dict)

    print(f"\nAll figures saved to ./{OUT_DIR}/")
