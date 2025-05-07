"""
Evaluation module using COCO-style mAP for instance segmentation with mask R-CNN.
"""

import json
import tempfile

import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from tqdm import tqdm

NUM_CLASSES = 5


def evaluate_map(model, data_loader, device):
    """
    Evaluate the given model using COCO-style mean average precision (mAP) on the dataset.

    Args:
        model (torch.nn.Module): The instance segmentation model.
        data_loader (torch.utils.data.DataLoader): Data loader with images and annotations.
        device (torch.device): The computation device (e.g., 'cuda' or 'cpu').

    Returns:
        float: mAP@[IoU=0.50:0.95] score.
    """
    model.eval()
    results = []
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": str(i)} for i in range(1, NUM_CLASSES)]
    }

    ann_id = 1
    image_id = 1

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        image = images[0].to(device)
        target = {k: v.to(device) for k, v in targets[0].items()}

        with torch.no_grad():
            outputs = model([image])[0]

        h, w = image.shape[1:]
        coco_gt["images"].append({
            "id": image_id,
            "width": w,
            "height": h
        })

        for i, box in enumerate(target["boxes"]):
            gt_mask = target["masks"][i].cpu().numpy().astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(gt_mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(target["labels"][i]),
                "segmentation": rle,
                "bbox": list(map(float, torchvision.ops.box_convert(
                    box.cpu(), in_fmt='xyxy', out_fmt='xywh'))),
                "iscrowd": 0,
                "area": float(gt_mask.sum())
            })
            ann_id += 1

        for i, score in enumerate(outputs["scores"]):
            if score < 0.05:
                continue
            pred_mask = outputs["masks"][i][0].cpu().numpy() > 0.5
            rle = mask_utils.encode(
                np.asfortranarray(pred_mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            results.append({
                "image_id": image_id,
                "category_id": int(outputs["labels"][i]),
                "segmentation": rle,
                "score": float(score),
                "bbox": list(map(float, torchvision.ops.box_convert(
                    outputs["boxes"][i].cpu(), in_fmt='xyxy', out_fmt='xywh')))
            })

        image_id += 1

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f_gt, \
            tempfile.NamedTemporaryFile(mode='w+', delete=False) as f_dt:
        json.dump(coco_gt, f_gt)
        json.dump(results, f_dt)
        f_gt.flush()
        f_dt.flush()

        coco = COCO(f_gt.name)
        coco_dt = coco.loadRes(f_dt.name)
        coco_eval = COCOeval(coco, coco_dt, iouType='segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    return coco_eval.stats[0]
