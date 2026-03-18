import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop
import numpy as np
import csv
from file_finder_pb import get_pb_dataloaders

# ========== UNet ==========
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = double_conv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

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
        return self.out_conv(d1)


# ========== Config ==========
IMAGE_DIR  = r"C:\Users\Zimin\Desktop\CURREN~1\Research\829C1~1.202\3AC67~1.AII\4DFBA~1.FIN\VERSIO~1\LPBF_P~1\14996806\ORIGIN~1\ORIGIN~1\PB"
LABEL_DIR  = r"C:\Users\Zimin\Desktop\CURREN~1\Research\829C1~1.202\3AC67~1.AII\4DFBA~1.FIN\VERSIO~1\LPBF_P~1\14996806\ORIGIN~1\ORIGIN~1\PB_label"
IMG_SIZE   = 512
BATCH_SIZE = 4
NUM_EPOCHS = 50
PATIENCE   = 7
RESULTS_CSV = "pb_sweep_results.csv"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========== Loss Functions ==========
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_sig = torch.sigmoid(pred)
        smooth = 1e-6
        inter = (pred_sig * target).sum(dim=(2, 3))
        dice  = 1 - (2 * inter + smooth) / (
            pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
        return bce_loss + dice.mean()

class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        smooth = 1e-6
        inter = (pred_sig * target).sum(dim=(2, 3))
        dice  = 1 - (2 * inter + smooth) / (
            pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
        return dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, pred, target):
        bce   = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt    = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

def get_loss(loss_name):
    return {'BCE': nn.BCEWithLogitsLoss(),
            'Dice': DiceLoss(),
            'Focal': FocalLoss(),
            'BCEDice': BCEDiceLoss()}[loss_name]


# ========== Metrics ==========
def compute_metrics(pred, target, threshold):
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    smooth = 1e-6
    tp  = (pred_bin * target).sum(dim=(2, 3))
    fp  = (pred_bin * (1 - target)).sum(dim=(2, 3))
    fn  = ((1 - pred_bin) * target).sum(dim=(2, 3))
    dice      = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    iou       = (tp + smooth) / (tp + fp + fn + smooth)
    return (dice.mean().item(), precision.mean().item(),
            recall.mean().item(), iou.mean().item())


# ========== Single Experiment ==========
def run_experiment(lr, loss_name, optimizer_name, threshold):
    train_loader, val_loader = get_pb_dataloaders(
        IMAGE_DIR, LABEL_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    model     = UNet().to(device)
    criterion = get_loss(loss_name)

    if optimizer_name == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=lr)

    best_val_loss    = float('inf')
    patience_counter = 0
    best_metrics     = None

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0.0
        all_dice, all_prec, all_rec, all_iou = [], [], [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs   = model(images)
                val_loss += criterion(outputs, masks).item()
                d, p, r, iou = compute_metrics(outputs, masks, threshold)
                all_dice.append(d); all_prec.append(p)
                all_rec.append(r);  all_iou.append(iou)

        val_loss /= len(val_loader)
        metrics   = (np.mean(all_dice), np.mean(all_prec),
                     np.mean(all_rec),  np.mean(all_iou))

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_metrics     = metrics
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

        print(f"  Epoch {epoch+1:>2} | val_loss: {val_loss:.4f} | Dice: {metrics[0]:.4f}")

    return best_val_loss, best_metrics


# ========== Experiment Grid ==========
# Same structure as OT sweep for easy comparison
# (lr, loss, optimizer, threshold)
experiments = [
    (1e-4, 'BCEDice', 'Adam',    0.5),   # Exp1: baseline (same config as train_pb.py)
    (5e-4, 'BCEDice', 'Adam',    0.5),   # Exp2: higher lr
    (1e-3, 'BCEDice', 'Adam',    0.5),   # Exp3: even higher lr
    (1e-4, 'BCE',     'Adam',    0.5),   # Exp4: BCE only
    (1e-4, 'Dice',    'Adam',    0.5),   # Exp5: Dice only
    (1e-4, 'Focal',   'Adam',    0.5),   # Exp6: Focal loss (good for class imbalance)
    (1e-4, 'BCEDice', 'SGD',     0.5),   # Exp7: SGD optimizer
    (1e-4, 'BCEDice', 'RMSprop', 0.5),   # Exp8: RMSprop optimizer
    (1e-4, 'BCEDice', 'Adam',    0.3),   # Exp9: lower threshold
    (1e-4, 'BCEDice', 'Adam',    0.7),   # Exp10: higher threshold
]


# ========== Main ==========
if __name__ == '__main__':
    print(f"Device: {device}")
    print(f"Total experiments: {len(experiments)}\n")

    results = []

    for i, (lr, loss_name, opt_name, threshold) in enumerate(experiments, start=1):
        exp_label = f"Exp{i}: lr={lr}, loss={loss_name}, opt={opt_name}, tau={threshold}"
        print(f"\n{'='*65}")
        print(f"[PB] {exp_label}")
        print(f"{'='*65}")

        best_val_loss, metrics = run_experiment(lr, loss_name, opt_name, threshold)
        dice, prec, rec, iou  = metrics

        results.append({
            'exp':       i,
            'lr':        lr,
            'loss':      loss_name,
            'optimizer': opt_name,
            'threshold': threshold,
            'val_loss':  round(best_val_loss, 4),
            'dice':      round(dice,  4),
            'precision': round(prec,  4),
            'recall':    round(rec,   4),
            'iou':       round(iou,   4),
        })

        # Save after every experiment in case of crash
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"  --> Saved progress to {RESULTS_CSV}")

    # Final summary table
    print(f"\n{'='*85}")
    print("[PB] SWEEP FINAL RESULTS")
    print(f"{'='*85}")
    header = f"{'Exp':<5} {'LR':<8} {'Loss':<10} {'Opt':<10} {'Tau':<6} {'ValLoss':<9} {'Dice':<7} {'Prec':<7} {'Recall':<8} {'IoU':<7}"
    print(header)
    print("-" * 85)
    for r in results:
        print(f"{r['exp']:<5} {r['lr']:<8} {r['loss']:<10} {r['optimizer']:<10} "
              f"{r['threshold']:<6} {r['val_loss']:<9} {r['dice']:<7} "
              f"{r['precision']:<7} {r['recall']:<8} {r['iou']:<7}")

    # Highlight best experiment by Dice
    best = max(results, key=lambda x: x['dice'])
    print(f"\nBest Dice: Exp{best['exp']} "
          f"(lr={best['lr']}, loss={best['loss']}, "
          f"opt={best['optimizer']}, tau={best['threshold']}) "
          f"→ Dice={best['dice']}, IoU={best['iou']}")

    print(f"\nAll results saved to {RESULTS_CSV}")
