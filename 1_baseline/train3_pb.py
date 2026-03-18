import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from file_finder_pb import get_pb_dataloaders


# ========== U-Net ==========
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
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
IMAGE_DIR = r"C:\Users\Zimin\Desktop\CURREN~1\Research\829C1~1.202\3AC67~1.AII\4DFBA~1.FIN\VERSIO~1\LPBF_P~1\14996806\ORIGIN~1\ORIGIN~1\PB"
LABEL_DIR = r"C:\Users\Zimin\Desktop\CURREN~1\Research\829C1~1.202\3AC67~1.AII\4DFBA~1.FIN\VERSIO~1\LPBF_P~1\14996806\ORIGIN~1\ORIGIN~1\PB_label"
IMG_SIZE  = 512
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7

# PB-specific output files (separate from OT results)
SAVE_PATH       = "pb_best_model.pth"
SAVE_TRAIN_LOSS = "pb_train_losses.npy"
SAVE_VAL_LOSS   = "pb_val_losses.npy"
SAVE_TRAIN_DICE = "pb_train_dices.npy"
SAVE_VAL_DICE   = "pb_val_dices.npy"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ========== Loss ==========
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + smooth) / (
            pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
        )
        return bce_loss + dice_loss.mean()


# ========== Metrics ==========
def compute_dice(pred, target, threshold=0.5):
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    smooth = 1e-6
    intersection = (pred_bin * target).sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (
        pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
    )
    return dice.mean().item()


# ========== Train one epoch ==========
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_dice = 0, 0
    for batch_idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += compute_dice(outputs, masks)
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")
    return total_loss / len(loader), total_dice / len(loader)


# ========== Validate ==========
def validate(model, loader, criterion):
    model.eval()
    total_loss, total_dice = 0, 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, masks).item()
            total_dice += compute_dice(outputs, masks)
    return total_loss / len(loader), total_dice / len(loader)


# ========== Main training loop ==========
def train():
    train_loader, val_loader = get_pb_dataloaders(
        IMAGE_DIR, LABEL_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = BCEDiceLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses, train_dices, val_dices = [], [], [], []

    for epoch in range(NUM_EPOCHS):
        print(f"\n[PB] Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_dice = validate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)

        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Dice: {val_dice:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f">>> Saved best PB model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping! Best val_loss: {best_val_loss:.4f}")
            break

    np.save(SAVE_TRAIN_LOSS, np.array(train_losses))
    np.save(SAVE_VAL_LOSS,   np.array(val_losses))
    np.save(SAVE_TRAIN_DICE, np.array(train_dices))
    np.save(SAVE_VAL_DICE,   np.array(val_dices))
    print("\nPB training complete!")
    print(f"Model saved to: {SAVE_PATH}")


if __name__ == '__main__':
    train()
