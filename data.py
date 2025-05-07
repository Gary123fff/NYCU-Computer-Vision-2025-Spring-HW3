"""Dataset and visualization utilities for instance segmentation using PyTorch."""

import os
import numpy as np
import cv2  # pylint: disable=import-error
import torch
from torch.utils.data import Dataset
import skimage.io as skio
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as coco_mask

# Color mapping for classes (for visualization)
CLASS_COLOR_MAP = {
    1: (255, 50, 50),
    2: (50, 255, 50),
    3: (50, 50, 255),
    4: (255, 255, 80),
}


def decode_maskobj(mask_obj):
    """Decode a COCO-format RLE mask object into a binary mask."""
    return coco_mask.decode(mask_obj)


def encode_mask(binary_mask):
    """Encode a binary mask into COCO RLE format."""
    mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = coco_mask.encode(mask)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def read_maskfile(path):
    """Read a mask image from the given file path."""
    return skio.imread(path)


def visualize_random_samples(dataset, index=0):
    """Visualize the image and color overlay mask at the specified index in the dataset."""
    image_tensor, target = dataset[index]
    image_array = image_tensor.permute(1, 2, 0).numpy()
    image_array = (image_array * 255).astype(np.uint8)

    overlay = np.zeros_like(image_array)
    all_masks = target["masks"].numpy()
    all_labels = target["labels"].numpy()

    for idx, mask in enumerate(all_masks):
        class_id = int(all_labels[idx])
        color = CLASS_COLOR_MAP.get(class_id, (200, 200, 200))
        mask_bool = mask.astype(bool)
        for c in range(3):
            overlay[:, :, c][mask_bool] = color[c]

    plt.subplot(1, 2, 1)
    plt.imshow(image_array)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Colored Masks")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def train_transform():
    """Return the training augmentation pipeline."""
    return A.Compose([
        A.Resize(height=400, width=400),
        A.OneOf([
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=0.05, p=0.5)
        ], p=0.7),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def val_transform():
    """Return the validation augmentation pipeline."""
    return A.Compose([
        A.Resize(400, 400),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


class InstanceSegDataset(Dataset):
    """Custom PyTorch dataset for instance segmentation with mask and box annotations."""

    def __init__(self, root_dir, img_dirs, transforms=None):
        """
        Args:
            root_dir (str): Base directory for the dataset.
            img_dirs (List[str]): List of subfolders containing samples.
            transforms (callable, optional): Albumentations transformation pipeline.
        """
        self.root = root_dir
        self.image_folders = sorted(img_dirs)
        self.tfms = transforms

    def __len__(self):
        return len(self.image_folders)

    def __getitem__(self, idx):
        """
        Load image and corresponding instance masks, boxes, and labels.

        Returns:
            Tuple[Tensor, Dict]: (image_tensor, target_dict)
        """
        folder = self.image_folders[idx]
        dir_path = os.path.join(self.root, folder)
        img_path = os.path.join(dir_path, 'image.tif')

        img = cv2.imread(
            img_path, cv2.IMREAD_COLOR)  # pylint: disable=no-member
        img = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB)    # pylint: disable=no-member

        masks, boxes, labels = [], [], []

        for class_idx in range(1, 5):
            mask_path = os.path.join(dir_path, f"class{class_idx}.tif")
            if not os.path.exists(mask_path):
                continue

            raw_mask = skio.imread(mask_path)
            for obj_val in np.unique(raw_mask):
                if obj_val == 0:
                    continue

                bin_mask = (raw_mask == obj_val).astype(np.uint8)
                y_loc, x_loc = np.where(bin_mask)
                if len(y_loc) < 1 or len(x_loc) < 1:
                    continue

                x1, x2 = x_loc.min(), x_loc.max()
                y1, y2 = y_loc.min(), y_loc.max()
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(class_idx)
                masks.append(bin_mask)

        if self.tfms:
            result = self.tfms(image=img, masks=masks,
                               bboxes=boxes, labels=labels)
            img = result["image"]
            masks = result["masks"]
            boxes = result["bboxes"]
            labels = result["labels"]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.stack([
                m.clone().detach().to(torch.uint8) if isinstance(m, torch.Tensor) else torch.tensor(m, dtype=torch.uint8)
                for m in masks
            ])
        }

        return img, target
