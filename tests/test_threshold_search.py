import pytest

from object_counter.evaluation.count_report import AnnotationRow
from object_counter.evaluation.threshold_search import (
    ThresholdConfig,
    build_threshold_grid,
    classes_from_annotations,
    evaluation_rows_from_summaries,
    score_threshold_config,
    sort_threshold_scores,
    unique_samples,
)


def test_classes_from_annotations_are_sorted_unique() -> None:
    annotations = [
        _annotation(class_name="person"),
        _annotation(class_name="car"),
        _annotation(class_name="person"),
    ]

    assert classes_from_annotations(annotations) == ["car", "person"]


def test_unique_samples_rejects_conflicting_metadata() -> None:
    annotations = [
        _annotation(sample_id="a", input_path="a.jpg"),
        _annotation(sample_id="a", input_path="b.jpg"),
    ]

    with pytest.raises(ValueError):
        unique_samples(annotations)


def test_evaluation_rows_from_summaries_compares_expected_and_predicted() -> None:
    annotations = [
        _annotation(sample_id="a", class_name="person", expected_count=2),
        _annotation(sample_id="a", class_name="car", expected_count=1),
    ]
    summaries = {"a": {"counts": {"person": 3}}}

    rows = evaluation_rows_from_summaries(annotations, summaries)

    assert rows[0]["predicted_count"] == 3
    assert rows[0]["absolute_error"] == 1
    assert rows[1]["predicted_count"] == 0
    assert rows[1]["absolute_error"] == 1


def test_score_and_sort_threshold_configs_prioritizes_low_error() -> None:
    good = score_threshold_config(
        ThresholdConfig(confidence=0.4, iou=0.5),
        [_evaluation_row(expected=2, predicted=2)],
        processing_seconds=2.0,
    )
    bad = score_threshold_config(
        ThresholdConfig(confidence=0.2, iou=0.5),
        [_evaluation_row(expected=2, predicted=4)],
        processing_seconds=1.0,
    )

    assert sort_threshold_scores([bad, good])[0]["confidence"] == 0.4


def test_build_threshold_grid_combines_confidence_and_iou() -> None:
    grid = build_threshold_grid([0.3, 0.4], [0.5, 0.6])

    assert [config.slug for config in grid] == [
        "conf_0p3_iou_0p5",
        "conf_0p3_iou_0p6",
        "conf_0p4_iou_0p5",
        "conf_0p4_iou_0p6",
    ]


def _annotation(
    sample_id: str = "a",
    input_path: str = "a.jpg",
    class_name: str = "person",
    expected_count: int = 1,
) -> AnnotationRow:
    return AnnotationRow(
        sample_id=sample_id,
        input_path=input_path,
        summary_path="a.json",
        media_type="image",
        counting_mode="image",
        class_name=class_name,
        expected_count=expected_count,
    )


def _evaluation_row(expected: int, predicted: int) -> dict:
    return {
        "sample_id": "a",
        "class_name": "person",
        "expected_count": expected,
        "predicted_count": predicted,
        "absolute_error": abs(expected - predicted),
    }
