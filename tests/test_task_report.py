from __future__ import annotations

import json
from pathlib import Path

from object_counter.evaluation.task_report import (
    evaluate_task_annotations,
    predicted_value_from_summary,
    task_error_analysis_rows,
)
from object_counter.evaluation.vision_diagnostics import (
    pose_diagnostic_rows,
    segmentation_diagnostic_rows,
)


def test_predicted_value_from_summary_supports_segmentation_metrics() -> None:
    summary = {
        "counts": {"person": 2},
        "total": 2,
        "total_mask_area": 120.0,
        "area_metrics": {"mask_area_ratio": 0.25, "area_by_class": {"person": 120.0}},
    }

    assert predicted_value_from_summary(summary, "segmentation", "image", "instance_total") == 2
    assert predicted_value_from_summary(summary, "segmentation", "image", "class:person") == 2
    assert predicted_value_from_summary(summary, "segmentation", "image", "area_by_class:person") == 120


def test_evaluate_task_annotations_writes_rows_from_csv(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({"model_task": "keypoint_detection", "total": 1, "visible_keypoints": 12}),
        encoding="utf-8",
    )
    annotations_path = tmp_path / "annotations.csv"
    annotations_path.write_text(
        "\n".join(
            [
                "sample_id,task,input_path,summary_path,media_type,metric_name,expected_value,tolerance,notes,condition_tags,manual_error_type",
                f"sample,pose,input.jpg,{summary_path},image,person_total,1,0,,,",
                f"sample,pose,input.jpg,{summary_path},image,visible_keypoints,10,1,,,",
            ]
        ),
        encoding="utf-8",
    )

    report = evaluate_task_annotations(annotations_path)

    assert report.summary["metric_rows"] == 2
    assert report.summary["within_tolerance_rate"] == 0.5
    assert report.rows[1]["status"] == "revisar"
    assert task_error_analysis_rows(report.rows)[1]["error_direction"] == "superestimado"


def test_diagnostics_return_actionable_rows() -> None:
    segmentation_rows = segmentation_diagnostic_rows(
        {"model_task": "instance_segmentation", "total": 0, "area_metrics": {}}
    )
    pose_rows = pose_diagnostic_rows(
        {
            "model_task": "keypoint_detection",
            "total": 1,
            "visible_keypoints": 3,
            "average_keypoint_confidence": 0.2,
        }
    )

    assert segmentation_rows[0]["indicador"] == "sem_mascaras"
    assert any(row["indicador"] == "poucos_keypoints" for row in pose_rows)
    assert any(row["indicador"] == "confianca_baixa" for row in pose_rows)
