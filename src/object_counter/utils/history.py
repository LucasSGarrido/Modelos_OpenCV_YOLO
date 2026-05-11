from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

HISTORY_COLUMNS = [
    "run_at",
    "input_name",
    "media_type",
    "mode",
    "total",
    "classes",
    "confidence",
    "iou",
    "roi_enabled",
    "tracking_backend",
    "fps",
    "processing_seconds",
    "output_path",
    "summary_path",
    "csv_output",
]


def build_history_record(
    result: dict[str, Any],
    media_type: str | None,
    model_config: dict[str, Any],
    video_config: dict[str, Any],
) -> dict[str, Any]:
    mode = result.get("counting_mode", "image")

    if media_type == "image":
        total = result.get("total", 0)
    elif mode == "line":
        total = result.get("total_line_crossings", 0)
    else:
        total = sum(result.get("last_frame_counts", {}).values())

    return {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_name": Path(result.get("input_path", "arquivo")).name,
        "media_type": media_type or "-",
        "mode": mode,
        "total": int(total),
        "classes": " ".join(model_config.get("classes", [])),
        "confidence": float(model_config.get("confidence", 0)),
        "iou": float(model_config.get("iou", 0)),
        "roi_enabled": bool(video_config.get("roi_enabled", False)),
        "tracking_backend": result.get("tracking_backend", ""),
        "fps": float(result.get("average_processing_fps", 0) or 0),
        "processing_seconds": float(
            result.get("processing_seconds", result.get("inference_seconds", 0)) or 0
        ),
        "output_path": result.get("output_path", ""),
        "summary_path": result.get("summary_path", ""),
        "csv_output": result.get("csv_output", ""),
    }


def append_history_csv(path: str | Path, record: dict[str, Any], max_rows: int = 200) -> pd.DataFrame:
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = load_history_csv(history_path)
    new_row = pd.DataFrame([record], columns=HISTORY_COLUMNS)
    history = pd.concat([new_row, history], ignore_index=True)
    history = history[HISTORY_COLUMNS].head(max_rows)
    history.to_csv(history_path, index=False, encoding="utf-8")
    return history


def load_history_csv(path: str | Path) -> pd.DataFrame:
    history_path = Path(path)
    if not history_path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    history = pd.read_csv(history_path)
    for column in HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = ""
    return history[HISTORY_COLUMNS]


def filter_history(
    history: pd.DataFrame,
    media_type: str = "Todos",
    mode: str = "Todos",
    search: str = "",
) -> pd.DataFrame:
    filtered = history.copy()
    for column in ["input_name", "classes", "output_path", "media_type", "mode"]:
        if column not in filtered.columns:
            filtered[column] = ""

    if media_type != "Todos":
        filtered = filtered[filtered["media_type"] == media_type]

    if mode != "Todos":
        filtered = filtered[filtered["mode"] == mode]

    if search.strip():
        query = search.strip().lower()
        filtered = filtered[
            filtered["input_name"].astype(str).str.lower().str.contains(query)
            | filtered["classes"].astype(str).str.lower().str.contains(query)
            | filtered["output_path"].astype(str).str.lower().str.contains(query)
        ]

    return filtered
