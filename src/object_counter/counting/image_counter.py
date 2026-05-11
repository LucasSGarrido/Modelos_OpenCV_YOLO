from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from object_counter.counting.counts import count_by_label, total_count
from object_counter.counting.roi import RegionOfInterest
from object_counter.detection.detector import YoloDetector
from object_counter.utils.io import read_image, save_image, write_json
from object_counter.visualization.draw import draw_detections


@dataclass
class ImageCounterResult:
    input_path: str
    output_path: str
    summary_path: str | None
    counts: dict[str, int]
    total: int
    inference_seconds: float
    detections: list[dict]
    roi_config: dict | None = None

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "summary_path": self.summary_path,
            "counts": self.counts,
            "total": self.total,
            "inference_seconds": round(self.inference_seconds, 6),
            "detections": self.detections,
            "roi_config": self.roi_config,
        }


def process_image(
    input_path: str | Path,
    output_path: str | Path,
    detector: YoloDetector,
    classes: list[str] | None = None,
    summary_output: str | Path | None = None,
    roi: RegionOfInterest | None = None,
) -> ImageCounterResult:
    image = read_image(input_path)

    started_at = perf_counter()
    detections = detector.detect(image, classes=classes)
    if roi:
        detections = roi.filter(detections, image.shape)
    inference_seconds = perf_counter() - started_at

    counts = count_by_label(detections)
    annotated = draw_detections(
        image.copy(),
        detections,
        counts=counts,
        roi_config=roi.as_dict(image.shape) if roi else None,
    )
    save_image(output_path, annotated)

    result = ImageCounterResult(
        input_path=str(input_path),
        output_path=str(output_path),
        summary_path=str(summary_output) if summary_output else None,
        counts=counts,
        total=total_count(counts),
        inference_seconds=inference_seconds,
        detections=[detection.to_dict() for detection in detections],
        roi_config=roi.as_dict() if roi else None,
    )

    if summary_output:
        write_json(result.to_dict(), summary_output)

    return result
