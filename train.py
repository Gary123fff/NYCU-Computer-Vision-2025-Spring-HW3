"""Training script for instance segmentation using Swin Transformer + FPN."""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import InstanceSegDataset, train_transform
from model import count_parameters, swin_model_fpn
from evaluate import evaluate_map
from plot_curve import plot_loss_and_map

# Constants (UPPER_CASE naming style)
NUM_CLASSES = 5
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4
BATCH_SIZE = 4
MODEL_SAVE_PATH = "swin_model_fpn+head.pth"

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    """Custom collate function to handle batches of variable-sized targets."""
    return tuple(zip(*batch))


def train():
    """Main training loop."""
    set_seed(42)
    data_path = "./dataset/train"
    all_dirs = sorted(os.listdir(data_path))

    dataset = InstanceSegDataset(
        root_dir=data_path, img_dirs=all_dirs, transforms=train_transform())
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, collate_fn=collate_fn),
        "val": DataLoader(val_dataset, batch_size=1, shuffle=False,
                          num_workers=2, collate_fn=collate_fn)
    }

    model = swin_model_fpn(NUM_CLASSES, DEVICE).to(DEVICE)

    total_params = count_parameters(model)
    print(f"Trainable parameters: {total_params / 1e6:.2f}M")
    if total_params < 200_000_000:
        print("Your model meets the size requirement (< 200M).")
    else:
        print("Your model exceeds the 200M parameter limit.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=5e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-6)

    train_losses = []
    val_maps = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            dataloaders["train"], desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", ncols=100)
        for images, targets in progress_bar:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()}
                       for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            progress_bar.set_postfix(loss=losses.item())

        lr_scheduler.step()
        avg_loss = epoch_loss / len(dataloaders["train"])
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}")

        val_map = evaluate_map(model, dataloaders["val"], DEVICE)
        val_maps.append(val_map)
        print(f"Validation mAP: {val_map:.4f}")

        plot_loss_and_map(train_losses, val_maps, epoch + 1)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("Training completed!")
