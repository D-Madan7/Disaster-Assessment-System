from __future__ import annotations
from typing import Any
import pandas as pd

from .config import ENABLE_RAG_RETRIEVAL, TOP_K_DOC_CHUNKS, USE_TRUE_LABELS_FOR_NOTEBOOK_ANALYSIS
from .report_generator import generate_report_text
from .retrieval import RetrievalEngine, build_scene_query, retrieve_doc_chunks
from .utils import infer_doc_folder, infer_event_name, safe_row_get, standardize_columns


def summarize_buildings_per_image(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize building-level predictions per image exactly as in the notebook."""
    if pred_df.empty:
        return pd.DataFrame(
            columns=[
                "image_id",
                "event_name",
                "doc_folder",
                "total_buildings",
                "pred_no_damage",
                "pred_damaged",
                "pred_destroyed",
                "pred_affected_total",
                "pred_affected_ratio",
                "pred_destroyed_ratio",
                "avg_confidence",
            ]
        )

    pred_df = standardize_columns(pred_df)

    if "image_id" not in pred_df.columns:
        pred_df = pred_df.copy()
        pred_df["image_id"] = "unknown_scene"

    rows = []
    for image_id, group in pred_df.groupby("image_id"):
        total = len(group)
        count_no_damage = int((group["predicted_label"] == "no-damage").sum())
        count_damaged = int((group["predicted_label"] == "damaged").sum())
        count_destroyed = int((group["predicted_label"] == "destroyed").sum())
        damaged_total = count_damaged + count_destroyed
        damaged_ratio = damaged_total / total if total else 0.0
        destroyed_ratio = count_destroyed / total if total else 0.0
        avg_conf = float(group["confidence"].mean()) if "confidence" in group.columns else float("nan")

        row: dict[str, Any] = {
            "image_id": image_id,
            "event_name": infer_event_name(image_id),
            "doc_folder": infer_doc_folder(image_id),
            "total_buildings": total,
            "pred_no_damage": count_no_damage,
            "pred_damaged": count_damaged,
            "pred_destroyed": count_destroyed,
            "pred_affected_total": damaged_total,
            "pred_affected_ratio": damaged_ratio,
            "pred_destroyed_ratio": destroyed_ratio,
            "avg_confidence": avg_conf,
        }

        if USE_TRUE_LABELS_FOR_NOTEBOOK_ANALYSIS and "true_label" in group.columns:
            true_no_damage = int((group["true_label"] == "no-damage").sum())
            true_damaged = int((group["true_label"] == "damaged").sum())
            true_destroyed = int((group["true_label"] == "destroyed").sum())
            row.update(
                {
                    "true_no_damage": true_no_damage,
                    "true_damaged": true_damaged,
                    "true_destroyed": true_destroyed,
                }
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)


def compute_impact_score(row: pd.Series) -> float:
    """Compute the notebook's interpretable impact score."""
    score = 0.0
    score += 0.45 * safe_row_get(row, "pred_destroyed_ratio", 0.0)
    score += 0.35 * safe_row_get(row, "pred_affected_ratio", 0.0)
    score += 0.20 * safe_row_get(row, "changed_area_ratio", 0.0)
    return score


def impact_label(score: float) -> str:
    """Map impact score to the notebook priority label."""
    if score >= 0.35:
        return "high"
    if score >= 0.18:
        return "medium"
    return "low"


def fuse_scene_and_mask(scene_df: pd.DataFrame, mask_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Fuse scene summaries with change-detection summaries."""
    fused = scene_df.merge(
        mask_df[["image_id", "changed_area_ratio", "dominant_region", "mask_path"]],
        on="image_id",
        how="left",
    )
    fused["split"] = split_name
    fused["impact_score"] = fused.apply(compute_impact_score, axis=1)
    fused["priority_level"] = fused["impact_score"].apply(impact_label)
    return fused.sort_values(["impact_score", "image_id"], ascending=[False, True]).reset_index(drop=True)


def generate_reports_for_split(
    fused_df: pd.DataFrame,
    retrieval_engine: RetrievalEngine,
    top_k: int = TOP_K_DOC_CHUNKS,
) -> pd.DataFrame:
    """Generate final reports for a fused split DataFrame."""
    if fused_df.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "image_id",
                "event_name",
                "doc_folder",
                "priority_level",
                "impact_score",
                "changed_area_ratio",
                "dominant_region",
                "report_text",
                "retrieved_sources",
            ]
        )

    rows = []
    for _, row in fused_df.iterrows():
        query = build_scene_query(row)
        retrieved = retrieve_doc_chunks(query, row["image_id"], retrieval_engine, top_k=top_k)
        report_text = generate_report_text(row, retrieved)
        rows.append(
            {
                "split": row["split"],
                "image_id": row["image_id"],
                "event_name": row["event_name"],
                "doc_folder": row["doc_folder"],
                "priority_level": row["priority_level"],
                "impact_score": row["impact_score"],
                "changed_area_ratio": row["changed_area_ratio"],
                "dominant_region": row["dominant_region"],
                "report_text": report_text,
                "retrieved_sources": sorted(set(item["file_name"] for item in retrieved)),
            }
        )
    return pd.DataFrame(rows)


def run_rag_pipeline(
    change_detection_df: pd.DataFrame,
    severity_df: pd.DataFrame,
    retrieval_engine: RetrievalEngine | None = None,
    top_k_doc_chunks: int = TOP_K_DOC_CHUNKS,
) -> pd.DataFrame:
    """Run the full fusion, retrieval, and reporting pipeline."""
    if retrieval_engine is None:
        retrieval_engine = RetrievalEngine(enable_rag_retrieval=ENABLE_RAG_RETRIEVAL)

    severity_df = standardize_columns(severity_df)
    change_detection_df = standardize_columns(change_detection_df)

    split_series = None
    if "split" in severity_df.columns:
        split_series = severity_df["split"].fillna("inference")
    else:
        split_series = pd.Series(["inference"] * len(severity_df), index=severity_df.index)

    reports = []
    for split_name in split_series.astype(str).unique():
        severity_split_df = severity_df[split_series.astype(str) == split_name].copy()

        if "split" in change_detection_df.columns:
            change_split_df = change_detection_df[change_detection_df["split"].astype(str) == split_name].copy()
        else:
            change_split_df = change_detection_df.copy()
            change_split_df["split"] = split_name

        scene_df = summarize_buildings_per_image(severity_split_df)
        fused_df = fuse_scene_and_mask(scene_df, change_split_df, split_name)
        report_df = generate_reports_for_split(fused_df, retrieval_engine=retrieval_engine, top_k=top_k_doc_chunks,)
        reports.append(report_df)

    if not reports:
        return pd.DataFrame()

    return pd.concat(reports, ignore_index=True)
