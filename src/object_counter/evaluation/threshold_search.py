from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from object_counter.evaluation.count_report import (
    AnnotationRow,
    predicted_counts_from_summary,
    summarize_evaluation,
)


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    input_path: str
    media_type: str
    counting_mode: str


@dataclass(frozen=True)
class ThresholdConfig:
    confidence: float
    iou: float

    @property
    def slug(self) -> str:
        confidence = str(self.confidence).replace(".", "p")
        iou = str(self.iou).replace(".", "p")
        return f"conf_{confidence}_iou_{iou}"


def classes_from_annotations(annotations: Iterable[AnnotationRow]) -> list[str]:
    return sorted({annotation.class_name for annotation in annotations})


def unique_samples(annotations: Iterable[AnnotationRow]) -> list[SampleSpec]:
    samples: dict[str, SampleSpec] = {}
    for annotation in annotations:
        sample = SampleSpec(
            sample_id=annotation.sample_id,
            input_path=annotation.input_path,
            media_type=annotation.media_type,
            counting_mode=annotation.counting_mode,
        )
        existing = samples.get(annotation.sample_id)
        if existing and existing != sample:
            raise ValueError(
                f"A amostra {annotation.sample_id!r} possui metadados conflitantes."
            )
        samples[annotation.sample_id] = sample

    return sorted(samples.values(), key=lambda sample: sample.sample_id)


def build_threshold_grid(
    confidence_values: Iterable[float],
    iou_values: Iterable[float],
) -> list[ThresholdConfig]:
    configs = [
        ThresholdConfig(confidence=round(float(confidence), 4), iou=round(float(iou), 4))
        for confidence in confidence_values
        for iou in iou_values
    ]
    if not configs:
        raise ValueError("A grade de confidence/IOU não pode ficar vazia.")
    return configs


def evaluation_rows_from_summaries(
    annotations: Iterable[AnnotationRow],
    summaries_by_sample: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for annotation in annotations:
        summary = summaries_by_sample[annotation.sample_id]
        predicted_counts = predicted_counts_from_summary(
            summary=summary,
            media_type=annotation.media_type,
            counting_mode=annotation.counting_mode,
        )
        predicted_count = int(predicted_counts.get(annotation.class_name, 0))
        absolute_error = abs(annotation.expected_count - predicted_count)
        rows.append(
            {
                "sample_id": annotation.sample_id,
                "input_path": annotation.input_path,
                "media_type": annotation.media_type,
                "counting_mode": annotation.counting_mode,
                "class_name": annotation.class_name,
                "expected_count": annotation.expected_count,
                "predicted_count": predicted_count,
                "absolute_error": absolute_error,
                "condition_tags": annotation.condition_tags,
                "manual_error_type": annotation.manual_error_type,
                "notes": annotation.notes,
            }
        )
    return rows


def score_threshold_config(
    config: ThresholdConfig,
    evaluation_rows: list[dict[str, Any]],
    processing_seconds: float,
) -> dict[str, Any]:
    summary = summarize_evaluation(evaluation_rows)
    false_positive_count = 0
    false_negative_count = 0
    for row in evaluation_rows:
        expected = int(row["expected_count"])
        predicted = int(row["predicted_count"])
        false_positive_count += max(0, predicted - expected)
        false_negative_count += max(0, expected - predicted)

    return {
        "confidence": config.confidence,
        "iou": config.iou,
        "sample_count": summary["sample_count"],
        "class_rows": summary["class_rows"],
        "total_absolute_error": summary["total_absolute_error"],
        "mean_absolute_count_error": summary["mean_absolute_count_error"],
        "mean_absolute_percentage_error": summary["mean_absolute_percentage_error"],
        "exact_count_match_rate": summary["exact_count_match_rate"],
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "processing_seconds": round(processing_seconds, 4),
    }


def sort_threshold_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["mean_absolute_count_error"]),
            int(row["total_absolute_error"]),
            int(row["false_positive_count"]) + int(row["false_negative_count"]),
            -float(row["exact_count_match_rate"]),
            float(row["processing_seconds"]),
        ),
    )
