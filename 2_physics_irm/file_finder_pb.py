"""
file_finder_pb.py
=================
PyTorch DataLoader factory for the LPBF Powder Bed (PB) dataset.

Dataset structure expected:
    <IMAGE_DIR>/
        set1A_0001.jpg
        set1A_0002.jpg
        ...
    <LABEL_DIR>/
        set1A_0001.xml   (Pascal VOC bounding boxes)
        set1A_0002.xml
        ...

Labels are Pascal VOC XML bounding-box annotations, which are converted
to filled binary masks at load time.

Usage:
    from file_finder_pb import get_pb_dataloaders
    train_loader, val_loader = get_pb_dataloaders(
        image_dir, label_dir, img_size=512, batch_size=4
    )
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PBDataset(Dataset):
    """
    LPBF Powder Bed image dataset with Pascal-VOC XML annotations.

    Returns
    -------
    image : torch.Tensor  (3, H, W)  normalised with ImageNet stats
    mask  : torch.Tensor  (1, H, W)  binary {0, 1}
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, image_dir: str, label_dir: str,
                 img_size: int = 512, augment: bool = False):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size  = img_size
        self.augment   = augment

        self.img_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if len(self.img_files) == 0:
            raise FileNotFoundError(
                f"No images found in {image_dir}. "
                "Check that IMAGE_DIR is correct."
            )

        print(f"[PBDataset] {len(self.img_files)} images  "
              f"(img_size={img_size}, augment={augment})")

    def __len__(self) -> int:
        return len(self.img_files)

    def _load_mask(self, img_filename: str, orig_h: int, orig_w: int) -> Image.Image:
        """Convert Pascal-VOC XML bounding boxes → filled binary PIL mask."""
        xml_stem = Path(img_filename).stem + ".xml"
        xml_path = self.label_dir / xml_stem
        mask     = Image.new('L', (orig_w, orig_h), 0)

        if not xml_path.exists():
            return mask   # no annotation → all-background mask

        try:
            root = ET.parse(xml_path).getroot()
            draw = ImageDraw.Draw(mask)
            for obj in root.findall('object'):
                bb   = obj.find('bndbox')
                xmin = int(float(bb.find('xmin').text))
                ymin = int(float(bb.find('ymin').text))
                xmax = int(float(bb.find('xmax').text))
                ymax = int(float(bb.find('ymax').text))
                draw.rectangle([xmin, ymin, xmax, ymax], fill=255)
        except Exception as e:
            print(f"  [WARNING] Failed to parse {xml_path}: {e}")

        return mask

    def _augment(self, image: Image.Image, mask: Image.Image):
        """Standard geometric augmentations (applied consistently to both)."""
        import random
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)
        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle)
        mask  = TF.rotate(mask,  angle)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        return image, mask

    def __getitem__(self, idx: int):
        fname  = self.img_files[idx]
        image  = Image.open(self.image_dir / fname).convert('RGB')
        orig_w, orig_h = image.size
        mask   = self._load_mask(fname, orig_h, orig_w)

        image  = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask   = mask.resize( (self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        image  = TF.to_tensor(image)
        image  = TF.normalize(image, mean=self.MEAN, std=self.STD)

        mask_t = torch.from_numpy(np.array(mask)).float()
        mask_t = (mask_t > 127).float().unsqueeze(0)   # (1, H, W) binary

        return image, mask_t


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def get_pb_dataloaders(
    image_dir: str,
    label_dir: str,
    img_size:  int = 512,
    batch_size: int = 4,
    val_split:  float = 0.2,
    num_workers: int = 2,
    seed: int = 42,
):
    """
    Build train and validation DataLoaders for the PB dataset.

    Parameters
    ----------
    image_dir   : path to folder with .jpg / .png images
    label_dir   : path to folder with .xml Pascal VOC annotations
    img_size    : resize both sides to this value (default 512)
    batch_size  : samples per batch
    val_split   : fraction held out for validation
    num_workers : DataLoader worker threads
    seed        : random seed for the train/val split

    Returns
    -------
    train_loader, val_loader : torch.utils.data.DataLoader
    """
    full_ds = PBDataset(image_dir, label_dir, img_size=img_size, augment=False)

    n_val   = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=generator)

    # Wrap train subset with augmentation
    train_ds_aug = PBDataset(image_dir, label_dir, img_size=img_size, augment=True)
    from torch.utils.data import Subset
    train_ds = Subset(train_ds_aug, train_ds.indices)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"[DataLoader] train={n_train}  val={n_val}  "
          f"batch={batch_size}  img_size={img_size}")
    return train_loader, val_loader
