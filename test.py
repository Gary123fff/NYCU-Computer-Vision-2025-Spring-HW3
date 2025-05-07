import torch
import os
import json
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as maskUtils
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# Constants
LABEL_COLORS = {
    1: (0, 255, 0),   # Green
    2: (0, 0, 255),   # Red
    3: (255, 0, 0),   # Blue
    4: (255, 255, 0),  # Yellow
}


def visualize_predictions(test_loader, model, device, test_transform):
    """
    Visualizes the predictions by displaying the image with its predicted masks.
    """
    sample_indices = random.sample(range(len(test_loader.dataset)), 5)
    print(f"Visualizing predictions for image indices: {sample_indices}")

    fig, axes = plt.subplots(5, 1, figsize=(6, 25))

    for i, idx in enumerate(sample_indices):
        image, image_id, height, width = test_loader.dataset[idx]
        aug = test_transform(image=image)
        img_tensor = aug['image'].unsqueeze(0).to(device)
        output = model(img_tensor)[0]

        black_background = np.zeros_like(image)

        if len(output["masks"]) > 0:
            masks = output["masks"].squeeze(1).detach().cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for mask, label in zip(masks, labels):
                binary_mask = (mask > 0.5).astype(np.uint8)
                binary_mask = cv2.resize(
                    binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

                color = LABEL_COLORS.get(
                    label, (255, 255, 255))  # Default to white
                colored = np.zeros_like(image)
                colored[:, :, 0] = binary_mask * color[0]  # B
                colored[:, :, 1] = binary_mask * color[1]  # G
                colored[:, :, 2] = binary_mask * color[2]  # R

                black_background = cv2.addWeighted(
                    black_background, 1.0, colored, 1.0, 0)

        axes[i].imshow(black_background)
        axes[i].set_title(f"Predicted Mask for Image ID: {image_id}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# Constants for dataset paths
DATA_PATH = "./dataset"
SAVED_PATH = "./"
BATCH_SIZE = 4
NUM_CLASSES = 2


class TestDataset(Dataset):
    """
    Custom dataset class to load images and their annotations.
    """

    def __init__(self, img_dir, json_file):
        """
        Args:
            img_dir (str): Path to the directory containing images.
            json_file (str): Path to the JSON file with image annotations.
        """
        self.img_dir = img_dir
        with open(json_file, 'r', encoding='utf-8') as f:
            self.img_info = json.load(f)
        self.img_info.sort(key=lambda x: x["id"])

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the image to load.

        Returns:
            image (ndarray): Loaded image.
            image_id (int): Image ID.
            height (int): Height of the image.
            width (int): Width of the image.
        """
        info = self.img_info[index]
        file_name = info["file_name"]
        image_id = info["id"]
        height = info["height"]
        width = info["width"]

        image_path = os.path.join(self.img_dir, file_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, image_id, height, width


def collate_fn(batch):
    """
    Collates batch data.
    """
    return tuple(zip(*batch))


def load_data(data_path, batch_size):
    """
    Loads test dataset.

    Args:
        data_path (str): Path to dataset.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: DataLoader object for testing.
    """
    dataset = TestDataset(
        img_dir=os.path.join(data_path, "test_release"),
        json_file=os.path.join(data_path, "test_image_name_to_ids.json")
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    return loader


@torch.no_grad()
def test_model(device, model):
    """
    Function to load the model, run inference, and visualize predictions.
    """
    model_path = os.path.join(SAVED_PATH, "swin_model_fpn+head.pth")
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")

    test_transform = A.Compose([
        A.Resize(400, 400),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_loader = load_data(DATA_PATH, BATCH_SIZE)
    results = []

    for images, image_ids, heights, widths in tqdm(test_loader, desc="Inference"):
        for img, image_id, height, width in zip(images, image_ids, heights, widths):
            aug = test_transform(image=img)
            img_tensor = aug['image'].unsqueeze(0).to(device)

            output = model(img_tensor)[0]

            boxes = output['boxes'].cpu()
            scores = output['scores'].cpu()
            labels = output['labels'].cpu()
            masks = output['masks'].squeeze(1).cpu()  # (N, H, W)

            scale = 384. / max(height, width)
            pad_h = max(0, 384 - int(height * scale))
            pad_w = max(0, 384 - int(width * scale))
            pad_top = pad_h // 2
            pad_left = pad_w // 2

            for i in range(len(scores)):
                x_min, y_min, x_max, y_max = boxes[i]

                x_min = (x_min - pad_left) / scale
                x_max = (x_max - pad_left) / scale
                y_min = (y_min - pad_top) / scale
                y_max = (y_max - pad_top) / scale

                x_min = np.clip(x_min.item(), 0, width)
                x_max = np.clip(x_max.item(), 0, width)
                y_min = np.clip(y_min.item(), 0, height)
                y_max = np.clip(y_max.item(), 0, height)

                width_bbox = x_max - x_min
                height_bbox = y_max - y_min

                mask = (masks[i] > 0.5).numpy().astype(np.uint8)

                mask = mask[
                    pad_top: 384 - (pad_h - pad_top),
                    pad_left: 384 - (pad_w - pad_left)
                ]

                mask = cv2.resize(mask, (width, height),
                                  interpolation=cv2.INTER_NEAREST)

                rle = maskUtils.encode(np.asfortranarray(mask))
                rle['counts'] = rle['counts'].decode('utf-8')

                results.append({
                    "image_id": int(image_id),
                    "bbox": [float(x_min), float(y_min), float(width_bbox), float(height_bbox)],
                    "score": float(scores[i]),
                    "category_id": int(labels[i]),
                    "segmentation": {
                        "size": [height, width],
                        "counts": rle['counts']
                    }
                })

    os.makedirs(SAVED_PATH, exist_ok=True)
    pred_json_path = os.path.join(SAVED_PATH, "test-results.json")
    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"Segmentation predictions saved to {pred_json_path}")

    visualize_predictions(test_loader, model, device, test_transform)
