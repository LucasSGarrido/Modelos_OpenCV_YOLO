from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from object_counter.utils.io import ensure_parent_dir, write_json


@dataclass(frozen=True)
class TaskAnnotationRow:
    sample_id: str
    task: str
    input_path: str
    summary_path: str
    media_type: str
    metric_name: str
    expected_value: float
    tolerance: float = 0.0
    notes: str = ""
    condition_tags: str = ""
    manual_error_type: str = ""


@dataclass(frozen=True)
class TaskEvaluationReport:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def load_task_annotations(path: str | Path) -> list[TaskAnnotationRow]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [
            TaskAnnotationRow(
                sample_id=row["sample_id"],
                task=row["task"],
                input_path=row["input_path"],
                summary_path=row["summary_path"],
                media_type=row["media_type"],
                metric_name=row["metric_name"],
                expected_value=float(row["expected_value"]),
                tolerance=float(row.get("tolerance") or 0.0),
                notes=row.get("notes", ""),
                condition_tags=row.get("condition_tags", ""),
                manual_error_type=row.get("manual_error_type", ""),
            )
            for row in reader
        ]


def evaluate_task_annotations(path: str | Path) -> TaskEvaluationReport:
    annotations = load_task_annotations(path)
    summaries = _load_summaries(annotations)
    rows: list[dict[str, Any]] = []

    for annotation in annotations:
        summary = summaries[annotation.summary_path]
        predicted_value = predicted_value_from_summary(
            summary=summary,
            task=annotation.task,
            media_type=annotation.media_type,
            metric_name=annotation.metric_name,
        )
        absolute_error = abs(annotation.expected_value - predicted_value)
        within_tolerance = absolute_error <= annotation.tolerance
        rows.append(
            {
                "sample_id": annotation.sample_id,
                "task": annotation.task,
                "input_path": annotation.input_path,
                "summary_path": annotation.summary_path,
                "media_type": annotation.media_type,
                "metric_name": annotation.metric_name,
                "expected_value": annotation.expected_value,
                "predicted_value": predicted_value,
                "absolute_error": absolute_error,
                "relative_error_pct": _relative_error_pct(
                    annotation.expected_value,
                    predicted_value,
                ),
                "tolerance": annotation.tolerance,
                "within_tolerance": within_tolerance,
                "status": "ok" if within_tolerance else "revisar",
                "suggested_issue": _suggest_metric_issue(annotation, predicted_value),
                "condition_tags": annotation.condition_tags,
                "manual_error_type": annotation.manual_error_type,
                "notes": annotation.notes,
            }
        )

    return TaskEvaluationReport(rows=rows, summary=summarize_task_evaluation(rows))


def predicted_value_from_summary(
    summary: dict[str, Any],
    task: str,
    media_type: str,
    metric_name: str,
) -> float:
    task = task.lower()
    metric_name = metric_name.lower()
    media_type = media_type.lower()

    if task == "segmentation":
        return _segmentation_value(summary, media_type, metric_name)
    if task == "pose":
        return _pose_value(summary, media_type, metric_name)

    raise ValueError(f"Tarefa não suportada para avaliação manual: {task}")


def summarize_task_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_error = sum(float(row["absolute_error"]) for row in rows)
    row_count = len(rows)
    sample_count = len({str(row["sample_id"]) for row in rows})
    exact_rows = sum(1 for row in rows if bool(row["within_tolerance"]))
    by_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        task = str(row["task"])
        bucket = by_task.setdefault(task, {"rows": 0, "total_absolute_error": 0.0, "ok_rows": 0})
        bucket["rows"] += 1
        bucket["total_absolute_error"] += float(row["absolute_error"])
        bucket["ok_rows"] += int(bool(row["within_tolerance"]))

    for bucket in by_task.values():
        bucket["mean_absolute_error"] = round(
            bucket["total_absolute_error"] / bucket["rows"],
            4,
        )
        bucket["within_tolerance_rate"] = round(bucket["ok_rows"] / bucket["rows"], 4)
        bucket["total_absolute_error"] = round(bucket["total_absolute_error"], 4)

    return {
        "sample_count": sample_count,
        "metric_rows": row_count,
        "total_absolute_error": round(total_error, 4),
        "mean_absolute_error": round(total_error / row_count, 4) if row_count else 0.0,
        "within_tolerance_rate": round(exact_rows / row_count, 4) if row_count else 0.0,
        "by_task": by_task,
    }


def task_error_analysis_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    analysis = []
    for row in rows:
        expected = float(row["expected_value"])
        predicted = float(row["predicted_value"])
        if bool(row["within_tolerance"]):
            direction = "ok"
        elif predicted > expected:
            direction = "superestimado"
        else:
            direction = "subestimado"

        analysis.append({**row, "error_direction": direction})
    return analysis


def write_task_evaluation_report(
    report: TaskEvaluationReport,
    csv_output: str | Path,
    summary_output: str | Path,
    error_output: str | Path | None = None,
) -> None:
    ensure_parent_dir(csv_output)
    fieldnames = [
        "sample_id",
        "task",
        "input_path",
        "summary_path",
        "media_type",
        "metric_name",
        "expected_value",
        "predicted_value",
        "absolute_error",
        "relative_error_pct",
        "tolerance",
        "within_tolerance",
        "status",
        "suggested_issue",
        "condition_tags",
        "manual_error_type",
        "notes",
    ]
    with Path(csv_output).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report.rows)

    write_json(report.summary, summary_output)

    if error_output:
        _write_task_error_analysis(task_error_analysis_rows(report.rows), error_output)


def _segmentation_value(summary: dict[str, Any], media_type: str, metric_name: str) -> float:
    if metric_name == "instance_total":
        return float(summary.get("max_frame_total", 0) if media_type == "video" else summary.get("total", 0))
    if metric_name.startswith("class:"):
        label = metric_name.split(":", 1)[1]
        counts = summary.get("max_counts_by_class", {}) if media_type == "video" else summary.get("counts", {})
        return float(counts.get(label, 0))
    if metric_name == "mask_area_total":
        return float(
            summary.get("max_frame_mask_area", 0.0)
            if media_type == "video"
            else summary.get("total_mask_area", 0.0)
        )
    if metric_name == "area_ratio":
        if media_type == "video":
            return float(summary.get("max_frame_area_ratio", 0.0))
        return float(summary.get("area_metrics", {}).get("mask_area_ratio", 0.0))
    if metric_name == "largest_mask_area":
        if media_type == "video":
            return float(summary.get("largest_mask_area", 0.0))
        return float(summary.get("area_metrics", {}).get("largest_mask_area", 0.0))
    if metric_name.startswith("area_by_class:"):
        label = metric_name.split(":", 1)[1]
        areas = (
            summary.get("max_area_by_class", {})
            if media_type == "video"
            else summary.get("area_metrics", {}).get("area_by_class", {})
        )
        return float(areas.get(label, 0.0))

    raise ValueError(f"Métrica de segmentação não suportada: {metric_name}")


def _pose_value(summary: dict[str, Any], media_type: str, metric_name: str) -> float:
    if metric_name == "person_total":
        return float(summary.get("max_people", 0) if media_type == "video" else summary.get("total", 0))
    if metric_name == "visible_keypoints":
        return float(
            summary.get("max_visible_keypoints", 0)
            if media_type == "video"
            else summary.get("visible_keypoints", 0)
        )
    if metric_name == "average_keypoint_confidence":
        return float(summary.get("average_keypoint_confidence") or 0.0)

    raise ValueError(f"Métrica de pose não suportada: {metric_name}")


def _load_summaries(annotations: list[TaskAnnotationRow]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        if annotation.summary_path in summaries:
            continue

        with Path(annotation.summary_path).open("r", encoding="utf-8") as file:
            summaries[annotation.summary_path] = json.load(file)

    return summaries


def _relative_error_pct(expected: float, predicted: float) -> float:
    if expected == 0:
        return 0.0 if predicted == 0 else 100.0
    return round(abs(expected - predicted) / abs(expected) * 100, 4)


def _suggest_metric_issue(annotation: TaskAnnotationRow, predicted_value: float) -> str:
    expected = annotation.expected_value
    if abs(expected - predicted_value) <= annotation.tolerance:
        return "sem erro"

    if annotation.manual_error_type:
        return annotation.manual_error_type

    tags = {tag.strip().lower() for tag in annotation.condition_tags.split("|") if tag.strip()}
    if "baixa_iluminacao" in tags:
        return "baixa iluminação"
    if "oclusao" in tags:
        return "oclusão"
    if "objeto_pequeno" in tags:
        return "objeto pequeno"
    if "fora_roi" in tags:
        return "fora da ROI"

    if annotation.metric_name.startswith("area") or "mask_area" in annotation.metric_name:
        return "área de máscara divergente"
    if annotation.task == "pose" and "keypoint" in annotation.metric_name:
        return "keypoints incompletos ou incertos"
    if predicted_value > expected:
        return "superestimativa"
    return "subestimativa"


def _write_task_error_analysis(rows: list[dict[str, Any]], path: str | Path) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "sample_id",
        "task",
        "input_path",
        "summary_path",
        "media_type",
        "metric_name",
        "expected_value",
        "predicted_value",
        "absolute_error",
        "relative_error_pct",
        "tolerance",
        "within_tolerance",
        "status",
        "error_direction",
        "suggested_issue",
        "condition_tags",
        "manual_error_type",
        "notes",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
