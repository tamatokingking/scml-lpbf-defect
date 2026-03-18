"""
physics_guided_train.py
=======================
Stage 3: VED-stratified Invariant Risk Minimisation (IRM) for UNet.

The standard segmentation loss is augmented with a stability penalty:

    L_total = L_seg  +  λ * Σ_e (L_e − L̄)²

where L_e is the segmentation loss on VED-environment e and L̄ is the
mean across environments. This discourages the model from exploiting
environment-specific spurious correlations (e.g., texture artifacts
that only appear at high VED).

Also implements:
  - VED-conditioned decision threshold τ(VED)
  - Per-VED-bin evaluation
  - Physics-aware augmentation (horizontal stripe noise for PB artifacts)

Usage:
    python physics_guided_train.py
"""

import os
import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from ved_metadata import parse_filename, ved_to_env_id, VED_REF


# ===================================================================
# Config
# ===================================================================
IMAGE_DIR  = r"C:\Users\Zimin\Desktop\CURREN~1\Research\829C1~1.202\3AC67~1.AII\4DFBA~1.FIN\VERSIO~1\LPBF_P~1\14996806\ORIGIN~1\ORIGIN~1\PB"
LABEL_DIR  = r"C:\Users\Zimin\Desktop\CURREN~1\Research\829C1~1.202\3AC67~1.AII\4DFBA~1.FIN\VERSIO~1\LPBF_P~1\14996806\ORIGIN~1\ORIGIN~1\PB_label"

IMG_SIZE    = 512
BATCH_SIZE  = 4
NUM_EPOCHS  = 60
PATIENCE    = 10
LR          = 1e-4
WEIGHT_DECAY= 1e-4
IRM_LAMBDA  = 1.0      # stability penalty weight
VAL_SPLIT   = 0.2
NUM_ENVS    = 3        # low / stable / high

SAVE_PATH   = "pb_physics_best.pth"
RESULTS_CSV = "pb_physics_results.csv"

# Per-VED-bin thresholds (will be tuned on val set)
TAU_INIT    = {0: 0.5, 1: 0.5, 2: 0.5}   # env_id → threshold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================================================================
# Dataset
# ===================================================================
class PBPhysicsDataset(Dataset):
    """
    PB dataset that also returns env_id (VED regime) per image.
    Annotations are Pascal-VOC XML bounding boxes → converted to binary masks.
    """

    def __init__(self, image_dir, label_dir, img_size=512, augment=False):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size  = img_size
        self.augment   = augment

        self.img_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        # Pre-parse metadata
        self.meta = [parse_filename(f) for f in self.img_files]
        print(f"[PBPhysics] found {len(self.img_files)} images")
        self._print_env_dist()

    def _print_env_dist(self):
        from collections import Counter
        c = Counter(m["regime"] for m in self.meta)
        print(f"  Regime distribution: {dict(c)}")

    def __len__(self):
        return len(self.img_files)

    def _load_mask(self, img_filename, orig_h, orig_w):
        """Convert Pascal-VOC XML bounding boxes to binary mask."""
        xml_stem = Path(img_filename).stem + ".xml"
        xml_path = self.label_dir / xml_stem
        mask     = Image.new('L', (orig_w, orig_h), 0)
        if not xml_path.exists():
            return mask
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            draw = ImageDraw.Draw(mask)
            for obj in root.findall('object'):
                bb   = obj.find('bndbox')
                xmin = int(float(bb.find('xmin').text))
                ymin = int(float(bb.find('ymin').text))
                xmax = int(float(bb.find('xmax').text))
                ymax = int(float(bb.find('ymax').text))
                draw.rectangle([xmin, ymin, xmax, ymax], fill=255)
        except Exception:
            pass
        return mask

    def _physics_augment(self, image: Image.Image) -> Image.Image:
        """
        Physics-aware augmentation for PB images:
        Add synthetic horizontal stripe noise to simulate recoater artifacts.
        This teaches the model that stripes are NOT defects.
        """
        arr  = np.array(image).astype(np.float32)
        h, w = arr.shape[:2]
        # Random horizontal stripe
        if random.random() > 0.7:
            n_stripes = random.randint(1, 3)
            for _ in range(n_stripes):
                y0 = random.randint(0, h - 5)
                y1 = y0 + random.randint(2, 6)
                intensity = random.uniform(-20, 20)
                arr[y0:y1, :] = np.clip(arr[y0:y1, :] + intensity, 0, 255)
        # Random vignette / illumination gradient
        if random.random() > 0.6:
            gradient = np.linspace(0.85, 1.15, w)[np.newaxis, :, np.newaxis]  # (1, W, 1) broadcasts with (H, W, 3)
            arr = np.clip(arr * gradient, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    def __getitem__(self, idx):
        fname  = self.img_files[idx]
        meta   = self.meta[idx]
        env_id = meta["env_id"]
        ved    = meta["ved"]

        image = Image.open(self.image_dir / fname).convert('RGB')
        orig_w, orig_h = image.size
        mask  = self._load_mask(fname, orig_h, orig_w)

        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask  = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image); mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image); mask = TF.vflip(mask)
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle); mask = TF.rotate(mask, angle)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            # Physics-aware stripe augmentation
            image = self._physics_augment(image)

        image = TF.to_tensor(image)
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])
        mask  = torch.from_numpy(np.array(mask)).float()
        mask  = (mask > 127).float().unsqueeze(0)

        return image, mask, torch.tensor(env_id, dtype=torch.long), torch.tensor(ved, dtype=torch.float32)


# ===================================================================
# UNet  (same architecture as sweep, + dropout in bottleneck)
# ===================================================================
def double_conv(in_ch, out_ch, dropout=0.0):
    layers = [
        nn.Conv2d(in_ch,  out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, bottleneck_dropout=0.3):
        super().__init__()
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64,  128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(512, 1024, dropout=bottleneck_dropout)
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = double_conv(1024, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = double_conv(512, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = double_conv(256, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = double_conv(128, 64)
        self.out  = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


# ===================================================================
# Loss functions
# ===================================================================
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce  = self.bce(pred, target)
        p    = torch.sigmoid(pred)
        sm   = 1e-6
        inter = (p * target).sum(dim=(2, 3))
        dice  = 1 - (2 * inter + sm) / (p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + sm)
        return bce + dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma

    def forward(self, pred, target):
        bce   = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt    = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


def irm_penalty(losses_per_env: list) -> torch.Tensor:
    """
    IRM stability penalty:  Σ_e (L_e − L̄)²
    Encourages uniform loss across VED environments.
    """
    if len(losses_per_env) < 2:
        return torch.tensor(0.0)
    stacked = torch.stack(losses_per_env)
    return ((stacked - stacked.mean()) ** 2).sum()


# ===================================================================
# Metrics
# ===================================================================
def compute_metrics(pred_logits, target, tau=0.5):
    pred = (torch.sigmoid(pred_logits) >= tau).float()
    sm   = 1e-6
    tp   = (pred * target).sum(dim=(2, 3))
    fp   = (pred * (1 - target)).sum(dim=(2, 3))
    fn   = ((1 - pred) * target).sum(dim=(2, 3))
    dice = (2 * tp + sm) / (2 * tp + fp + fn + sm)
    iou  = (tp + sm) / (tp + fp + fn + sm)
    return dice.mean().item(), iou.mean().item()


# ===================================================================
# VED-conditioned threshold tuning
# ===================================================================
def tune_thresholds(model, val_loader, env_count=NUM_ENVS, device=device):
    """
    For each VED environment, sweep τ ∈ [0.2, 0.8] and pick the τ
    that maximises Dice on the val set.
    Returns dict {env_id: best_tau}.
    """
    tau_grid  = np.arange(0.2, 0.81, 0.05)
    best_tau  = {e: 0.5 for e in range(env_count)}
    best_dice = {e: 0.0 for e in range(env_count)}

    # Accumulate per-env predictions
    env_preds  = {e: [] for e in range(env_count)}
    env_masks  = {e: [] for e in range(env_count)}

    model.eval()
    with torch.no_grad():
        for images, masks, env_ids, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            for b in range(images.shape[0]):
                e = env_ids[b].item()
                env_preds[e].append(logits[b:b+1].cpu())
                env_masks[e].append(masks[b:b+1])

    for e in range(env_count):
        if not env_preds[e]:
            continue
        preds_cat = torch.cat(env_preds[e], dim=0)
        masks_cat = torch.cat(env_masks[e], dim=0)
        for tau in tau_grid:
            d, _ = compute_metrics(preds_cat, masks_cat, tau=tau)
            if d > best_dice[e]:
                best_dice[e] = d
                best_tau[e]  = round(float(tau), 2)

    print("\n  [Threshold tuning]")
    regime_names = {0: "low", 1: "stable", 2: "high"}
    for e in range(env_count):
        print(f"    env={e} ({regime_names[e]:7s})  best_tau={best_tau[e]:.2f}  Dice={best_dice[e]:.4f}")
    return best_tau


# ===================================================================
# Training
# ===================================================================
def train_one_epoch(model, train_loader, criterion, optimizer, irm_lambda):
    model.train()
    total_loss   = 0.0
    env_losses   = {e: [] for e in range(NUM_ENVS)}

    for images, masks, env_ids, _ in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)

        # Per-sample seg loss
        batch_loss = 0.0
        env_batch  = {e: [] for e in range(NUM_ENVS)}

        for b in range(images.shape[0]):
            loss_b = criterion(logits[b:b+1], masks[b:b+1])
            e      = env_ids[b].item()
            env_batch[e].append(loss_b)
            batch_loss = batch_loss + loss_b

        batch_loss = batch_loss / images.shape[0]

        # IRM penalty
        env_means = []
        for e in range(NUM_ENVS):
            if env_batch[e]:
                env_means.append(torch.stack(env_batch[e]).mean())
        penalty = irm_penalty(env_means) if len(env_means) >= 2 else torch.tensor(0.0)

        loss = batch_loss + irm_lambda * penalty
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, criterion, tau_dict):
    model.eval()
    val_loss     = 0.0
    env_dice     = {e: [] for e in range(NUM_ENVS)}
    env_iou      = {e: [] for e in range(NUM_ENVS)}

    for images, masks, env_ids, _ in val_loader:
        images, masks = images.to(device), masks.to(device)
        logits        = model(images)
        val_loss     += criterion(logits, masks).item()

        for b in range(images.shape[0]):
            e   = env_ids[b].item()
            tau = tau_dict.get(e, 0.5)
            d, iou = compute_metrics(logits[b:b+1], masks[b:b+1], tau=tau)
            env_dice[e].append(d)
            env_iou[e].append(iou)

    val_loss /= len(val_loader)
    avg_dice = np.mean([v for vals in env_dice.values() for v in vals])
    avg_iou  = np.mean([v for vals in env_iou.values()  for v in vals])

    per_env_dice = {e: np.mean(v) if v else 0.0 for e, v in env_dice.items()}
    return val_loss, avg_dice, avg_iou, per_env_dice


# ===================================================================
# Main
# ===================================================================
if __name__ == '__main__':
    print(f"Device: {device}")
    print(f"IRM lambda: {IRM_LAMBDA}\n")

    # Dataset
    full_ds = PBPhysicsDataset(IMAGE_DIR, LABEL_DIR,
                               img_size=IMG_SIZE, augment=False)
    n       = len(full_ds)
    n_val   = int(n * VAL_SPLIT)
    n_train = n - n_val
    indices = list(range(n))
    random.shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)

    # Augmentation wrapper for train set
    aug_ds = PBPhysicsDataset(IMAGE_DIR, LABEL_DIR,
                              img_size=IMG_SIZE, augment=True)
    train_ds = Subset(aug_ds, train_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {n_train}  Val: {n_val}\n")

    # Model
    model     = UNet(bottleneck_dropout=0.3).to(device)
    criterion = BCEDiceLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    tau_dict         = dict(TAU_INIT)
    best_val_loss    = float('inf')
    patience_counter = 0
    history          = []
    regime_names     = {0: "low", 1: "stable", 2: "high"}

    print(f"{'='*70}")
    print(f"  Physics-Guided IRM Training  |  λ_IRM={IRM_LAMBDA}")
    print(f"{'='*70}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, IRM_LAMBDA)
        val_loss, dice, iou, per_env_dice = validate(model, val_loader,
                                                      criterion, tau_dict)
        scheduler.step()

        env_str = "  ".join(
            f"{regime_names[e]}:{per_env_dice[e]:.3f}"
            for e in range(NUM_ENVS)
        )
        print(f"  Epoch {epoch:>3} | train={train_loss:.4f} | val={val_loss:.4f} | "
              f"Dice={dice:.4f} | IoU={iou:.4f} | [{env_str}]")

        row = {"epoch": epoch, "train_loss": round(train_loss, 4),
               "val_loss": round(val_loss, 4),
               "dice": round(dice, 4), "iou": round(iou, 4)}
        for e in range(NUM_ENVS):
            row[f"dice_{regime_names[e]}"] = round(per_env_dice[e], 4)
        history.append(row)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"           --> Best model saved")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}")
            break

        # Re-tune thresholds every 10 epochs
        if epoch % 10 == 0:
            tau_dict = tune_thresholds(model, val_loader)

    # Final threshold tuning on best model
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device,
                                     weights_only=True))
    print("\n  Final threshold tuning on best model...")
    tau_dict = tune_thresholds(model, val_loader)

    # Save results
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    print(f"\nTraining complete. Results → {RESULTS_CSV}")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Final per-VED thresholds: {tau_dict}")