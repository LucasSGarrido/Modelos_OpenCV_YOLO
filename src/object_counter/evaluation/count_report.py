from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from object_counter.evaluation.metrics import (
    exact_count_match_rate,
    mean_absolute_count_error,
    mean_absolute_percentage_error,
)
from object_counter.utils.io import ensure_parent_dir, write_json


@dataclass(frozen=True)
class AnnotationRow:
    sample_id: str
    input_path: str
    summary_path: str
    media_type: str
    counting_mode: str
    class_name: str
    expected_count: int
    notes: str = ""
    condition_tags: str = ""
    manual_error_type: str = ""


@dataclass(frozen=True)
class EvaluationReport:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def load_annotations(path: str | Path) -> list[AnnotationRow]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [
            AnnotationRow(
                sample_id=row["sample_id"],
                input_path=row["input_path"],
                summary_path=row["summary_path"],
                media_type=row["media_type"],
                counting_mode=row["counting_mode"],
                class_name=row["class_name"],
                expected_count=int(row["expected_count"]),
                notes=row.get("notes", ""),
                condition_tags=row.get("condition_tags", ""),
                manual_error_type=row.get("manual_error_type", ""),
            )
            for row in reader
        ]


def evaluate_annotations(path: str | Path) -> EvaluationReport:
    annotations = load_annotations(path)
    summaries = _load_summaries(annotations)
    rows: list[dict[str, Any]] = []

    for annotation in annotations:
        summary = summaries[annotation.summary_path]
        predicted_counts = predicted_counts_from_summary(
            summary=summary,
            media_type=annotation.media_type,
            counting_mode=annotation.counting_mode,
        )
        predicted_count = int(predicted_counts.get(annotation.class_name, 0))
        error = abs(annotation.expected_count - predicted_count)
        rows.append(
            {
                "sample_id": annotation.sample_id,
                "input_path": annotation.input_path,
                "summary_path": annotation.summary_path,
                "media_type": annotation.media_type,
                "counting_mode": annotation.counting_mode,
                "class_name": annotation.class_name,
                "expected_count": annotation.expected_count,
                "predicted_count": predicted_count,
                "absolute_error": error,
                "condition_tags": annotation.condition_tags,
                "manual_error_type": annotation.manual_error_type,
                "notes": annotation.notes,
            }
        )

    summary = summarize_evaluation(rows)
    return EvaluationReport(rows=rows, summary=summary)


def predicted_counts_from_summary(
    summary: dict[str, Any],
    media_type: str,
    counting_mode: str,
) -> dict[str, int]:
    if media_type == "image":
        return _normalize_summary_counts(summary.get("counts", {}))

    if counting_mode == "line":
        return _normalize_summary_counts(summary.get("line_counts_by_class", {}))

    if counting_mode == "max_frame":
        return _normalize_summary_counts(summary.get("max_counts_by_class", {}))

    return _normalize_summary_counts(summary.get("last_frame_counts", {}))


def summarize_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_true: dict[str, dict[str, int]] = {}
    grouped_pred: dict[str, dict[str, int]] = {}

    for row in rows:
        sample_id = str(row["sample_id"])
        class_name = str(row["class_name"])
        grouped_true.setdefault(sample_id, {})[class_name] = int(row["expected_count"])
        grouped_pred.setdefault(sample_id, {})[class_name] = int(row["predicted_count"])

    sample_ids = sorted(grouped_true)
    true_samples = [grouped_true[sample_id] for sample_id in sample_ids]
    pred_samples = [grouped_pred.get(sample_id, {}) for sample_id in sample_ids]
    total_absolute_error = sum(int(row["absolute_error"]) for row in rows)

    return {
        "sample_count": len(sample_ids),
        "class_rows": len(rows),
        "total_absolute_error": total_absolute_error,
        "mean_absolute_count_error": round(
            mean_absolute_count_error(true_samples, pred_samples),
            4,
        ),
        "mean_absolute_percentage_error": round(
            mean_absolute_percentage_error(true_samples, pred_samples),
            4,
        ),
        "exact_count_match_rate": round(
            exact_count_match_rate(true_samples, pred_samples),
            4,
        ),
    }


def error_analysis_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    analysis = []
    for row in rows:
        expected = int(row["expected_count"])
        predicted = int(row["predicted_count"])
        false_positive_count = max(0, predicted - expected)
        false_negative_count = max(0, expected - predicted)
        condition_tags = str(row.get("condition_tags", "") or "")
        manual_error_type = str(row.get("manual_error_type", "") or "")
        if false_positive_count:
            status = "falso positivo"
        elif false_negative_count:
            status = "falso negativo"
        else:
            status = "ok"
        suggested_issue = manual_error_type or _suggest_issue_from_tags(condition_tags, status)

        analysis.append(
            {
                **row,
                "false_positive_count": false_positive_count,
                "false_negative_count": false_negative_count,
                "status": status,
                "suggested_issue": suggested_issue,
            }
        )
    return analysis


def write_evaluation_report(
    report: EvaluationReport,
    csv_output: str | Path,
    summary_output: str | Path,
    error_output: str | Path | None = None,
) -> None:
    ensure_parent_dir(csv_output)
    fieldnames = [
        "sample_id",
        "input_path",
        "summary_path",
        "media_type",
        "counting_mode",
        "class_name",
        "expected_count",
        "predicted_count",
        "absolute_error",
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
        _write_error_analysis(error_analysis_rows(report.rows), error_output)


def _write_error_analysis(rows: list[dict[str, Any]], path: str | Path) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "sample_id",
        "input_path",
        "summary_path",
        "media_type",
        "counting_mode",
        "class_name",
        "expected_count",
        "predicted_count",
        "absolute_error",
        "false_positive_count",
        "false_negative_count",
        "status",
        "suggested_issue",
        "condition_tags",
        "manual_error_type",
        "notes",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_summaries(annotations: list[AnnotationRow]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        if annotation.summary_path in summaries:
            continue

        path = Path(annotation.summary_path)
        with path.open("r", encoding="utf-8") as file:
            summaries[annotation.summary_path] = json.load(file)

    return summaries


def _normalize_summary_counts(counts: dict[str, Any]) -> dict[str, int]:
    return {str(label): int(value) for label, value in counts.items()}


def _suggest_issue_from_tags(condition_tags: str, status: str) -> str:
    tags = {tag.strip().lower() for tag in condition_tags.split("|") if tag.strip()}
    if "baixa_iluminacao" in tags:
        return "baixa iluminação"
    if "oclusao" in tags:
        return "oclusão"
    if "objeto_pequeno" in tags:
        return "objeto pequeno"
    if "classe_confusa" in tags:
        return "classe confundida"
    if "fora_roi" in tags:
        return "fora da ROI"
    if status == "falso positivo":
        return "detecção excedente"
    if status == "falso negativo":
        return "detecção perdida"
    return "sem erro"
