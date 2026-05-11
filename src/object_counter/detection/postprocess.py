from __future__ import annotations

from collections.abc import Iterable

from object_counter.detection.detector import Detection


def filter_detections_by_label(
    detections: Iterable[Detection], labels: Iterable[str] | None
) -> list[Detection]:
    if not labels:
        return list(detections)

    allowed = {label.lower() for label in labels}
    return [detection for detection in detections if detection.label.lower() in allowed]


def filter_detections_by_confidence(
    detections: Iterable[Detection], min_confidence: float
) -> list[Detection]:
    return [detection for detection in detections if detection.confidence >= min_confidence]
