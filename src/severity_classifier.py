import os
import json
import math
import random
import warnings
from pathlib import Path

import rasterio
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

try:
    import timm
except ImportError as exc:
    raise ImportError("Please install timm before running this script.") from exc

warnings.filterwarnings("ignore")

from src.config import IMAGE_SIZE_CNN, RANDOM_SEED, DEVICE, PADDING_RATIO, CLASS_NAMES, CLASS_TO_IDX, MERGE_MAPPING, IGNORED_CLASSES, CLASSIFIER_WEIGHTS_PATH
from src.utils import seed_everything

seed_everything(RANDOM_SEED)

# Helper Functions
def load_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_xy_points(wkt_polygon: str):
    polygon_text = wkt_polygon.replace("POLYGON ((", "").replace("))", "")
    points = []
    for pair in polygon_text.split(","):
        x_str, y_str = pair.strip().split()
        points.append((float(x_str), float(y_str)))
    return points


def polygon_to_bbox(points, width: int, height: int, padding_ratio: float = 0.10):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    box_w = max(1.0, x_max - x_min)
    box_h = max(1.0, y_max - y_min)
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    x1 = max(0, int(math.floor(x_min - pad_x)))
    y1 = max(0, int(math.floor(y_min - pad_y)))
    x2 = min(width, int(math.ceil(x_max + pad_x)))
    y2 = min(height, int(math.ceil(y_max + pad_y)))
    return x1, y1, x2, y2


def extract_building_samples_from_post_label(post_label_path: Path, post_image_path: Path):
    record = load_json(post_label_path)
    metadata = record.get("metadata", {})
    features = record.get("features", {})
    xy_features = features.get("xy", [])

    width = int(metadata.get("width", 1024))
    height = int(metadata.get("height", 1024))
    image_id = metadata.get("img_name", Path(post_image_path).name).replace("_post_disaster.png", "")
    disaster = metadata.get("disaster", "unknown")
    disaster_type = metadata.get("disaster_type", "unknown")

    rows = []
    for feature in xy_features:
        props = feature.get("properties", {})
        if props.get("feature_type") != "building":
            continue

        subtype = props.get("subtype")
        if subtype in IGNORED_CLASSES or subtype is None:
            continue

        subtype = MERGE_MAPPING[subtype]

        if subtype not in CLASS_TO_IDX:
            continue

        points = extract_xy_points(feature["wkt"])
        x1, y1, x2, y2 = polygon_to_bbox(
            points,
            width=width,
            height=height,
            padding_ratio=PADDING_RATIO,
        )
        if x2 <= x1 or y2 <= y1:
            continue

        rows.append(
            {
                "image_id": image_id,
                "image_path": str(post_image_path),
                "label_path": str(post_label_path),
                "building_uid": props.get("uid"),
                "label_name": subtype,
                "label_idx": CLASS_TO_IDX[subtype],
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "polygon_points": points,
                "disaster": disaster,
                "disaster_type": disaster_type,
            }
        )
    return pd.DataFrame(rows)


eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE_CNN, IMAGE_SIZE_CNN)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class BuildingSeverityDataset:
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        with rasterio.open(row["image_path"]) as src:
            image = src.read([1, 2, 3]).transpose(1, 2, 0)

        x1 = int(row["bbox_x1"])
        y1 = int(row["bbox_y1"])
        x2 = int(row["bbox_x2"])
        y2 = int(row["bbox_y2"])

        crop = image[y1:y2, x1:x2]
        crop = Image.fromarray(crop.astype(np.uint8))

        if self.transform:
            crop_tensor = self.transform(crop)
        else:
            crop_tensor = transforms.ToTensor()(crop)

        label = int(row["label_idx"])

        metadata = {
            "image_id": row["image_id"],
            "building_uid": row["building_uid"],
            "label_name": row["label_name"],
            "bbox": [x1, y1, x2, y2],
        }

        return crop_tensor, label, metadata

# Model
def build_model(num_classes: int):
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    return model


def load_model(weights_path, device):
    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Inference
def softmax_confidence(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=1)
    confidences, preds = probs.max(dim=1)
    return probs, preds, confidences


def build_prediction_dataframe(y_true, logits_tensor, metadata_list, split_name: str):
    probs, preds, confidences = softmax_confidence(logits_tensor)

    rows = []

    for idx, meta in enumerate(metadata_list):
        rows.append(
            {
                "split": split_name,
                "image_id": meta["image_id"],
                "building_uid": meta["building_uid"],
                "true_label": CLASS_NAMES[y_true[idx]],
                "predicted_label": CLASS_NAMES[int(preds[idx].item())],
                "confidence": float(confidences[idx].item()),
                "bbox_x1": meta["bbox"][0],
                "bbox_y1": meta["bbox"][1],
                "bbox_x2": meta["bbox"][2],
                "bbox_y2": meta["bbox"][3],
            }
        )

    return pd.DataFrame(rows)


@torch.no_grad()
def predict_buildings(dataset, model, device, split_name="inference"):
    model.eval()

    y_true = []
    logits_list = []
    metadata_list = []

    for idx in range(len(dataset)):
        crop_tensor, label, metadata = dataset[idx]
        image = crop_tensor.unsqueeze(0).to(device)
        logits = model(image)

        y_true.append(label)
        logits_list.append(logits.cpu())
        metadata_list.append(
            {
                "image_id": metadata["image_id"],
                "building_uid": metadata["building_uid"],
                "label_name": metadata["label_name"],
                "bbox": [
                    int(metadata["bbox"][0]),
                    int(metadata["bbox"][1]),
                    int(metadata["bbox"][2]),
                    int(metadata["bbox"][3]),
                ],
            }
        )

    logits_tensor = torch.cat(logits_list, dim=0)
    return build_prediction_dataframe(y_true, logits_tensor, metadata_list, split_name)


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    WEIGHTS_PATH = "CLASSIFIER_WEIGHTS_PATH"
    POST_IMAGE_PATH = ""
    POST_LABEL_PATH = ""
    OUTPUT_CSV_PATH = "severity_predictions.csv"

    if not WEIGHTS_PATH or not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")
    if not POST_IMAGE_PATH or not os.path.exists(POST_IMAGE_PATH):
        raise FileNotFoundError(f"Post-disaster image not found: {POST_IMAGE_PATH}")
    if not POST_LABEL_PATH or not os.path.exists(POST_LABEL_PATH):
        raise FileNotFoundError(f"Post-disaster label not found: {POST_LABEL_PATH}")

    sample_df = extract_building_samples_from_post_label(
        Path(POST_LABEL_PATH),
        Path(POST_IMAGE_PATH),
    )

    if len(sample_df) == 0:
        raise ValueError("No valid building samples were extracted from the provided post label.")

    dataset = BuildingSeverityDataset(sample_df, transform=eval_transform)
    model = load_model(WEIGHTS_PATH, DEVICE)
    pred_df = predict_buildings(dataset, model, DEVICE, split_name="inference")
    pred_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Saved predictions to: {OUTPUT_CSV_PATH}")
