from object_counter.evaluation.count_report import (
    error_analysis_rows,
    predicted_counts_from_summary,
    summarize_evaluation,
)


def test_predicted_counts_from_image_summary() -> None:
    summary = {"counts": {"bus": 1, "person": 3}}

    assert predicted_counts_from_summary(summary, media_type="image", counting_mode="image") == {
        "bus": 1,
        "person": 3,
    }


def test_predicted_counts_from_video_line_summary() -> None:
    summary = {
        "line_counts_by_class": {"bus": 1},
        "last_frame_counts": {"bus": 2},
    }

    assert predicted_counts_from_summary(summary, media_type="video", counting_mode="line") == {
        "bus": 1
    }


def test_summarize_evaluation_uses_sample_level_metrics() -> None:
    rows = [
        {
            "sample_id": "a",
            "class_name": "bus",
            "expected_count": 1,
            "predicted_count": 1,
            "absolute_error": 0,
        },
        {
            "sample_id": "a",
            "class_name": "person",
            "expected_count": 3,
            "predicted_count": 2,
            "absolute_error": 1,
        },
    ]

    assert summarize_evaluation(rows) == {
        "sample_count": 1,
        "class_rows": 2,
        "total_absolute_error": 1,
        "mean_absolute_count_error": 1.0,
        "mean_absolute_percentage_error": 0.25,
        "exact_count_match_rate": 0.0,
    }


def test_error_analysis_rows_labels_false_positive_and_negative() -> None:
    rows = [
        {
            "sample_id": "a",
            "class_name": "car",
            "expected_count": 1,
            "predicted_count": 3,
            "absolute_error": 2,
        },
        {
            "sample_id": "b",
            "class_name": "person",
            "expected_count": 4,
            "predicted_count": 2,
            "absolute_error": 2,
        },
    ]

    analysis = error_analysis_rows(rows)

    assert analysis[0]["false_positive_count"] == 2
    assert analysis[0]["status"] == "falso positivo"
    assert analysis[0]["suggested_issue"] == "detecção excedente"
    assert analysis[1]["false_negative_count"] == 2
    assert analysis[1]["status"] == "falso negativo"


def test_error_analysis_rows_uses_condition_tags_for_issue() -> None:
    rows = [
        {
            "sample_id": "a",
            "class_name": "car",
            "expected_count": 2,
            "predicted_count": 1,
            "absolute_error": 1,
            "condition_tags": "oclusao|objeto_pequeno",
        }
    ]

    assert error_analysis_rows(rows)[0]["suggested_issue"] == "oclusão"
