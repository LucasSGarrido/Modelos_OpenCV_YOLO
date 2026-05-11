from object_counter.counting.counts import count_by_label, merge_max_counts, total_count
from object_counter.detection.detector import Detection


def test_count_by_label_returns_sorted_counts() -> None:
    detections = [
        Detection(0, 0, 10, 10, 0.9, 0, "car"),
        Detection(0, 0, 10, 10, 0.8, 0, "person"),
        Detection(0, 0, 10, 10, 0.7, 0, "car"),
    ]

    assert count_by_label(detections) == {"car": 2, "person": 1}


def test_total_count_sums_classes() -> None:
    assert total_count({"car": 2, "person": 1}) == 3


def test_merge_max_counts_keeps_largest_value_per_class() -> None:
    current = {"car": 2, "person": 1}
    new_counts = {"car": 1, "bottle": 3}

    assert merge_max_counts(current, new_counts) == {"bottle": 3, "car": 2, "person": 1}
