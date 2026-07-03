"""
Common utility functions used across the project.
"""

### SIAMESE
import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

### RAG
from __future__ import annotations
from typing import Any
import pandas as pd
from .config import EVENT_TO_DOC_FOLDER

def infer_event_name(image_id: str) -> str:
    """Infer the event name from an image id."""
    parts = image_id.split("_")
    if len(parts) < 2:
        return image_id
    return "_".join(parts[:-1]).replace("_", "-")


def infer_doc_folder(image_id: str) -> str:
    """Map an image id to the relevant document category."""
    event_name = infer_event_name(image_id)
    return EVENT_TO_DOC_FOLDER.get(event_name, "Building Assessment")


def extract_incident_info(image_id: str) -> tuple[str, str]:
    """Extract disaster type and event description from image id."""
    incident = image_id.split("_")[0]
    parts = incident.split("-")
    disaster_type = parts[0].replace("_", " ").title()
    event_name = " ".join(parts[1:]).replace("_", " ").title()

    if event_name:
        full_event = f"{disaster_type} {event_name}"
    else:
        full_event = disaster_type

    return disaster_type, full_event


def describe_change_ratio(value: float) -> str:
    """Describe the changed-area ratio using the notebook thresholds."""
    if value < 0.005:
        return (
            "Only limited scene-wide physical changes were detected, "
            "indicating that visible damage is relatively localized."
        )
    if value < 0.02:
        return "Localized structural changes were detected across the scene."
    if value < 0.05:
        return "Moderate structural changes were observed across multiple areas."
    return "Extensive structural changes were observed throughout the scene."


def describe_region(region: str) -> str:
    """Convert region keys into more natural wording."""
    mapping = {
        "upper-left": "northwestern portion",
        "upper-right": "northeastern portion",
        "lower-left": "southwestern portion",
        "lower-right": "southeastern portion",
        "center": "central region",
    }
    return mapping.get(region, region)


def get_priority_label(row: pd.Series) -> str:
    """Compute the report priority label exactly as in the notebook."""
    destroyed = row["pred_destroyed"]
    affected = row["pred_affected_ratio"]

    if destroyed >= 5 or affected >= 0.80:
        return "HIGH"
    if affected >= 0.40:
        return "MODERATE"
    return "LOW"


def build_impact_summary(row: pd.Series) -> str:
    """Build the notebook's impact summary text."""
    affected_pct = row["pred_affected_ratio"] * 100
    destroyed = int(row["pred_destroyed"])

    if destroyed == 0:
        return (
            f"Approximately {affected_pct:.1f}% of identified buildings show visible "
            "structural damage. Most affected buildings appear potentially repairable "
            "based on automated assessment."
        )

    if affected_pct == 100:
        return (
            f"{affected_pct:.1f}% of identified buildings were affected, including "
            f"{destroyed} building(s) classified as destroyed. The scene indicates "
            "significant disruption requiring coordinated inspection and recovery efforts."
        )

    return (
        f"Approximately {affected_pct:.1f}% of identified buildings were affected, "
        f"including {destroyed} building(s) classified as destroyed. The scene indicates "
        "significant disruption requiring coordinated inspection and recovery efforts."
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize prediction DataFrame column names using the notebook rules."""
    if df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]

    rename_dict: dict[str, str] = {}
    for col in df.columns:
        if col in ["image_id", "image_name", "img_id", "img_name", "imageid", "image-id", "image id"]:
            rename_dict[col] = "image_id"
        elif col in ["building_uid", "building_id", "buildinguid", "building-id", "building id", "uid"]:
            rename_dict[col] = "building_uid"
        elif col in ["predicted_label", "pred_label", "prediction", "label", "predicted"]:
            rename_dict[col] = "predicted_label"
        elif col in ["bbox_x1", "x1", "bbox-x1", "bbox x1"]:
            rename_dict[col] = "bbox_x1"
        elif col in ["bbox_y1", "y1", "bbox-y1", "bbox y1"]:
            rename_dict[col] = "bbox_y1"
        elif col in ["bbox_x2", "x2", "bbox-x2", "bbox x2"]:
            rename_dict[col] = "bbox_x2"
        elif col in ["bbox_y2", "y2", "bbox-y2", "bbox y2"]:
            rename_dict[col] = "bbox_y2"

    df = df.rename(columns=rename_dict)

    if "image_id" not in df.columns:
        for col in df.columns:
            if "image" in col or "img" in col or col == "id":
                df = df.rename(columns={col: "image_id"})
                break

    return df


def get_region_name(row_idx: int, col_idx: int, rows: int = 3, cols: int = 3) -> str:
    """Return grid region names using the notebook logic."""
    vertical = ["upper", "central", "lower"][row_idx] if rows == 3 else ["upper", "lower"][row_idx]
    horizontal = ["left", "center", "right"][col_idx] if cols == 3 else ["left", "right"][col_idx]
    if vertical == "central" and horizontal == "center":
        return "central region"
    return f"{vertical}-{horizontal} region"


def safe_row_get(row: pd.Series, key: str, default: Any) -> Any:
    """Series-safe wrapper used during score calculations."""
    if key in row:
        return row[key]
    return default