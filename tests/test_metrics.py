from object_counter.evaluation.metrics import (
    absolute_count_error,
    exact_count_match_rate,
    mean_absolute_count_error,
    mean_absolute_percentage_error,
    per_class_count_errors,
)


def test_absolute_count_error_uses_union_of_classes() -> None:
    assert absolute_count_error({"car": 4, "person": 1}, {"car": 3, "bottle": 2}) == 4


def test_per_class_count_errors() -> None:
    errors = per_class_count_errors({"car": 4}, {"car": 3, "person": 2})

    assert errors == {
        "car": {"real": 4, "previsto": 3, "erro_absoluto": 1},
        "person": {"real": 0, "previsto": 2, "erro_absoluto": 2},
    }


def test_mean_absolute_count_error() -> None:
    y_true = [{"car": 4}, {"person": 2}]
    y_pred = [{"car": 3}, {"person": 4}]

    assert mean_absolute_count_error(y_true, y_pred) == 1.5


def test_mean_absolute_percentage_error_skips_zero_total() -> None:
    y_true = [{"car": 4}, {}]
    y_pred = [{"car": 2}, {"car": 3}]

    assert mean_absolute_percentage_error(y_true, y_pred) == 0.5


def test_exact_count_match_rate() -> None:
    y_true = [{"car": 4}, {"person": 2}]
    y_pred = [{"car": 4}, {"person": 1}]

    assert exact_count_match_rate(y_true, y_pred) == 0.5
