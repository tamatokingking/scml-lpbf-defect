import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
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
        super(UNet, self).__init__()
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
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# ========== Config (PB) ==========
IMAGE_DIR  = r"C:\Users\Zimin\Desktop\CURREN~1\Research\829C1~1.202\3AC67~1.AII\4DFBA~1.FIN\VERSIO~1\LPBF_P~1\14996806\ORIGIN~1\ORIGIN~1\PB"
LABEL_DIR  = r"C:\Users\Zimin\Desktop\CURREN~1\Research\829C1~1.202\3AC67~1.AII\4DFBA~1.FIN\VERSIO~1\LPBF_P~1\14996806\ORIGIN~1\ORIGIN~1\PB_label"
MODEL_PATH = "pb_best_model.pth"
THRESHOLD  = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========== Helpers ==========
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)

def load_model():
    model = UNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


# ========== 1. Training curves ==========
def plot_training_curves():
    train_losses = np.load("pb_train_losses.npy")
    val_losses   = np.load("pb_val_losses.npy")
    train_dices  = np.load("pb_train_dices.npy")
    val_dices    = np.load("pb_val_dices.npy")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses,   label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('[PB] Training & Validation Loss')
    ax1.legend()

    ax2.plot(epochs, train_dices, label='Train Dice')
    ax2.plot(epochs, val_dices,   label='Val Dice')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice Score')
    ax2.set_title('[PB] Training & Validation Dice')
    ax2.legend()

    plt.tight_layout()
    plt.savefig("pb_training_curves.png", dpi=100, bbox_inches='tight')
    print("Saved pb_training_curves.png")
    plt.show()


# ========== 2. Prediction visualizations ==========
def visualize_predictions(num_samples=8):
    _, val_loader = get_pb_dataloaders(IMAGE_DIR, LABEL_DIR, img_size=512, batch_size=1)
    model = load_model()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    axes[0, 0].set_title("Input Image",       fontsize=14)
    axes[0, 1].set_title("Ground Truth Mask", fontsize=14)
    axes[0, 2].set_title("Predicted Mask",    fontsize=14)

    count = 0
    with torch.no_grad():
        for images, masks in val_loader:
            if count >= num_samples:
                break
            images = images.to(device)
            output = model(images)
            pred   = (torch.sigmoid(output) >= THRESHOLD).float()

            img       = denormalize(images[0].cpu()).permute(1, 2, 0).numpy()
            mask      = masks[0, 0].numpy()
            pred_mask = pred[0, 0].cpu().numpy()

            axes[count, 0].imshow(img);       axes[count, 0].axis('off')
            axes[count, 1].imshow(mask,      cmap='Reds'); axes[count, 1].axis('off')
            axes[count, 2].imshow(pred_mask, cmap='Reds'); axes[count, 2].axis('off')
            count += 1

    plt.tight_layout()
    plt.savefig("pb_predictions.png", dpi=100, bbox_inches='tight')
    print("Saved pb_predictions.png")
    plt.show()


# ========== 3. Collect all val predictions ==========
def collect_predictions():
    _, val_loader = get_pb_dataloaders(IMAGE_DIR, LABEL_DIR, img_size=512, batch_size=4)
    model = load_model()

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs   = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy().flatten())
            all_preds.append((probs >= THRESHOLD).float().cpu().numpy().flatten())
            all_labels.append(masks.numpy().flatten())

    return (np.concatenate(all_probs),
            np.concatenate(all_preds),
            np.concatenate(all_labels))


# ========== 4. Pixel-level confusion matrix ==========
def plot_pixel_confusion_matrix(all_preds, all_labels):
    cm   = confusion_matrix(all_labels.astype(int), all_preds.astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Defect", "Defect"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title("[PB] Pixel-Level Confusion Matrix")
    plt.tight_layout()
    plt.savefig("pb_pixel_confusion_matrix.png", dpi=100)
    print("Saved pb_pixel_confusion_matrix.png")
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    print(f"\n[PB] Pixel-Level Metrics:")
    print(f"  Precision: {tp/(tp+fp+1e-6):.4f}")
    print(f"  Recall:    {tp/(tp+fn+1e-6):.4f}")
    print(f"  Dice:      {2*tp/(2*tp+fp+fn+1e-6):.4f}")
    print(f"  IoU:       {tp/(tp+fp+fn+1e-6):.4f}")


# ========== 5. Image-level confusion matrix ==========
def plot_image_level_confusion_matrix():
    from scipy import ndimage
    _, val_loader = get_pb_dataloaders(IMAGE_DIR, LABEL_DIR, img_size=512, batch_size=1)
    model = load_model()

    img_preds, img_labels = [], []
    A_min = 50

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            outputs  = model(images)
            pred_mask = (torch.sigmoid(outputs) >= THRESHOLD).float().cpu().numpy()[0, 0]
            gt_mask   = masks.numpy()[0, 0]

            labeled, nf = ndimage.label(pred_mask)
            max_area = max(ndimage.sum(pred_mask, labeled, range(1, nf+1))) if nf > 0 else 0

            labeled_gt, nf_gt = ndimage.label(gt_mask)
            max_area_gt = max(ndimage.sum(gt_mask, labeled_gt, range(1, nf_gt+1))) if nf_gt > 0 else 0

            img_preds.append(1 if max_area >= A_min else 0)
            img_labels.append(1 if max_area_gt >= A_min else 0)

    cm   = confusion_matrix(img_labels, img_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Alarm", "Alarm"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Oranges')
    ax.set_title("[PB] Image-Level Confusion Matrix")
    plt.tight_layout()
    plt.savefig("pb_image_confusion_matrix.png", dpi=100)
    print("Saved pb_image_confusion_matrix.png")
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    print(f"\n[PB] Image-Level Metrics:")
    print(f"  TP (correct alarm):  {tp}")
    print(f"  TN (correct pass):   {tn}")
    print(f"  FP (false alarm):    {fp}")
    print(f"  FN (missed defect):  {fn}")
    print(f"  False Alarm Rate:    {fp/(fp+tn+1e-6):.4f}")


# ========== 6. ROC and PR curves ==========
def plot_roc_pr_curves(all_probs, all_labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(all_labels.astype(int), all_probs)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    ax1.plot([0,1],[0,1],'k--', label='Random')
    ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
    ax1.set_title('[PB] ROC Curve (Pixel Level)'); ax1.legend()

    precision, recall, _ = precision_recall_curve(all_labels.astype(int), all_probs)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, label=f'AUC = {pr_auc:.4f}')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.set_title('[PB] Precision-Recall Curve (Pixel Level)'); ax2.legend()

    plt.tight_layout()
    plt.savefig("pb_roc_pr_curves.png", dpi=100)
    print("Saved pb_roc_pr_curves.png")
    plt.show()

    print(f"\n[PB] ROC AUC: {roc_auc:.4f}")
    print(f"[PB] PR  AUC: {pr_auc:.4f}")


# ========== Main ==========
if __name__ == '__main__':
    print("=== [PB] Training Curves ===")
    plot_training_curves()

    print("\n=== [PB] Prediction Visualization ===")
    visualize_predictions(num_samples=8)

    print("\n=== [PB] Collecting Predictions ===")
    all_probs, all_preds, all_labels = collect_predictions()

    print("\n=== [PB] Pixel-Level Confusion Matrix ===")
    plot_pixel_confusion_matrix(all_preds, all_labels)

    print("\n=== [PB] ROC and PR Curves ===")
    plot_roc_pr_curves(all_probs, all_labels)

    print("\n=== [PB] Image-Level Confusion Matrix ===")
    plot_image_level_confusion_matrix()
