from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from object_counter.detection.detector import Detection


def count_by_label(detections: Iterable[Detection]) -> dict[str, int]:
    counter = Counter(detection.label for detection in detections)
    return dict(sorted(counter.items()))


def total_count(counts: Mapping[str, int]) -> int:
    return sum(int(value) for value in counts.values())


def merge_max_counts(current: Mapping[str, int], new_counts: Mapping[str, int]) -> dict[str, int]:
    labels = set(current) | set(new_counts)
    return {label: max(int(current.get(label, 0)), int(new_counts.get(label, 0))) for label in labels}
