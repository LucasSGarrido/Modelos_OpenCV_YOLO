from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

from object_counter.counting.roi import RegionOfInterest
from object_counter.detection.detector import _class_label
from object_counter.utils.io import ensure_parent_dir, read_image, save_image, write_json
from object_counter.utils.video import create_video_writer, open_video_capture
from object_counter.visualization.segmentation import draw_segmentations


@dataclass(frozen=True)
class SegmentationMask:
    """Single YOLO segmentation result in image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str
    polygon: list[tuple[float, float]]
    mask_area: float

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def bbox_area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["polygon"] = [[round(x, 2), round(y, 2)] for x, y in self.polygon]
        data["mask_area"] = round(self.mask_area, 2)
        data["bbox_area"] = round(self.bbox_area, 2)
        return data


@dataclass
class ImageSegmentationResult:
    input_path: str
    output_path: str
    summary_path: str | None
    counts: dict[str, int]
    total: int
    total_mask_area: float
    area_metrics: dict[str, Any]
    inference_seconds: float
    segments: list[dict]
    roi_config: dict | None = None
    model_task: str = "instance_segmentation"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_task": self.model_task,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "summary_path": self.summary_path,
            "counts": self.counts,
            "total": self.total,
            "total_mask_area": round(self.total_mask_area, 2),
            "area_metrics": self.area_metrics,
            "inference_seconds": round(self.inference_seconds, 6),
            "segments": self.segments,
            "roi_config": self.roi_config,
        }


@dataclass
class VideoSegmentationResult:
    input_path: str
    output_path: str
    summary_path: str | None
    csv_output: str | None
    frames_read: int
    frames_processed: int
    max_counts_by_class: dict[str, int]
    max_area_by_class: dict[str, float]
    last_frame_counts: dict[str, int]
    max_frame_total: int
    last_frame_total: int
    last_frame_mask_area: float
    max_frame_mask_area: float
    max_frame_area_ratio: float
    largest_mask_area: float
    last_frame_area_metrics: dict[str, Any]
    average_processing_fps: float
    processing_seconds: float
    roi_config: dict | None = None
    model_task: str = "instance_segmentation_video"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_task": self.model_task,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "summary_path": self.summary_path,
            "csv_output": self.csv_output,
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "max_counts_by_class": self.max_counts_by_class,
            "max_area_by_class": self.max_area_by_class,
            "last_frame_counts": self.last_frame_counts,
            "max_frame_total": self.max_frame_total,
            "last_frame_total": self.last_frame_total,
            "last_frame_mask_area": round(self.last_frame_mask_area, 2),
            "max_frame_mask_area": round(self.max_frame_mask_area, 2),
            "max_frame_area_ratio": round(self.max_frame_area_ratio, 6),
            "largest_mask_area": round(self.largest_mask_area, 2),
            "last_frame_area_metrics": self.last_frame_area_metrics,
            "average_processing_fps": round(self.average_processing_fps, 4),
            "processing_seconds": round(self.processing_seconds, 4),
            "roi_config": self.roi_config,
        }


class YoloSegmenter:
    """Thin wrapper around Ultralytics YOLO segmentation models."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        iou: float = 0.5,
        device: str | None = None,
        imgsz: int = 640,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            message = (
                "A dependência 'ultralytics' não está instalada. "
                "Instale com: pip install -r requirements.txt"
            )
            raise RuntimeError(message) from exc

        return YOLO(model_path)

    def segment(self, frame: Any, classes: Iterable[str] | None = None) -> list[SegmentationMask]:
        class_filter = {item.lower() for item in classes or []}
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        boxes = list(getattr(result, "boxes", []) or [])
        masks = getattr(result, "masks", None)
        polygons = _mask_polygons(masks)

        segments: list[SegmentationMask] = []
        for index, box in enumerate(boxes):
            class_id = int(box.cls[0].item())
            label = _class_label(names, class_id)

            if class_filter and label.lower() not in class_filter:
                continue

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf[0].item())
            polygon = polygons[index] if index < len(polygons) else []
            segments.append(
                SegmentationMask(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    label=label,
                    polygon=polygon,
                    mask_area=polygon_area(polygon),
                )
            )

        return segments


def process_segmentation_image(
    input_path: str | Path,
    output_path: str | Path,
    segmenter: YoloSegmenter,
    classes: list[str] | None = None,
    summary_output: str | Path | None = None,
    show_boxes: bool = True,
    roi: RegionOfInterest | None = None,
) -> ImageSegmentationResult:
    image = read_image(input_path)

    started_at = perf_counter()
    segments = segmenter.segment(image, classes=classes)
    if roi:
        segments = _filter_segments_by_roi(segments, roi, image.shape)
    inference_seconds = perf_counter() - started_at

    counts = _count_segments(segments)
    area_metrics = segmentation_area_metrics(segments, image.shape)
    annotated = draw_segmentations(
        image.copy(),
        segments,
        counts=counts,
        show_boxes=show_boxes,
        roi_config=roi.as_dict(image.shape) if roi else None,
    )
    save_image(output_path, annotated)

    result = ImageSegmentationResult(
        input_path=str(input_path),
        output_path=str(output_path),
        summary_path=str(summary_output) if summary_output else None,
        counts=counts,
        total=sum(counts.values()),
        total_mask_area=float(area_metrics["total_mask_area"]),
        area_metrics=area_metrics,
        inference_seconds=inference_seconds,
        segments=[segment.to_dict() for segment in segments],
        roi_config=roi.as_dict() if roi else None,
    )

    if summary_output:
        write_json(result.to_dict(), summary_output)

    return result


def process_segmentation_video(
    input_path: str | Path,
    output_path: str | Path,
    segmenter: YoloSegmenter,
    classes: list[str] | None = None,
    csv_output: str | Path | None = None,
    summary_output: str | Path | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    show_boxes: bool = True,
    roi: RegionOfInterest | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> VideoSegmentationResult:
    if frame_stride < 1:
        raise ValueError("frame_stride precisa ser maior ou igual a 1.")

    cap, metadata = open_video_capture(input_path)
    writer = create_video_writer(output_path, metadata)

    rows: list[dict[str, Any]] = []
    frame_index = 0
    frames_processed = 0
    max_counts_by_class: dict[str, int] = {}
    max_area_by_class: dict[str, float] = {}
    last_counts: dict[str, int] = {}
    last_segments: list[SegmentationMask] = []
    last_frame_mask_area = 0.0
    max_frame_total = 0
    max_frame_mask_area = 0.0
    max_frame_area_ratio = 0.0
    largest_mask_area = 0.0
    last_frame_area_metrics: dict[str, Any] = {}
    processing_started = perf_counter()
    processing_time_sum = 0.0
    total_frames = _progress_total(metadata.frame_count, max_frames)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if max_frames is not None and frame_index >= max_frames:
                break

            should_process = frame_index % frame_stride == 0
            if should_process:
                frame_started = perf_counter()
                last_segments = segmenter.segment(frame, classes=classes)
                if roi:
                    last_segments = _filter_segments_by_roi(last_segments, roi, frame.shape)
                frame_seconds = perf_counter() - frame_started
                processing_time_sum += frame_seconds
                frames_processed += 1

                last_counts = _count_segments(last_segments)
                last_frame_total = sum(last_counts.values())
                max_frame_total = max(max_frame_total, last_frame_total)
                max_counts_by_class = _merge_max_counts(max_counts_by_class, last_counts)
                last_frame_area_metrics = segmentation_area_metrics(last_segments, frame.shape)
                last_frame_mask_area = float(last_frame_area_metrics["total_mask_area"])
                max_frame_mask_area = max(max_frame_mask_area, last_frame_mask_area)
                max_frame_area_ratio = max(
                    max_frame_area_ratio,
                    float(last_frame_area_metrics["mask_area_ratio"]),
                )
                largest_mask_area = max(
                    largest_mask_area,
                    float(last_frame_area_metrics["largest_mask_area"]),
                )
                max_area_by_class = _merge_max_areas(
                    max_area_by_class,
                    last_frame_area_metrics["area_by_class"],
                )
                timestamp_seconds = frame_index / metadata.fps if metadata.fps > 0 else 0.0
                rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": round(timestamp_seconds, 4),
                        "processing_seconds": round(frame_seconds, 6),
                        "fps_estimate": round(1 / frame_seconds, 4)
                        if frame_seconds > 0
                        else 0.0,
                        "frame_total": last_frame_total,
                        "frame_counts_json": json.dumps(last_counts, ensure_ascii=False),
                        "frame_mask_area": round(last_frame_mask_area, 2),
                        "frame_area_ratio": round(
                            float(last_frame_area_metrics["mask_area_ratio"]),
                            6,
                        ),
                        "frame_average_mask_area": round(
                            float(last_frame_area_metrics["average_mask_area"]),
                            2,
                        ),
                        "frame_largest_mask_area": round(
                            float(last_frame_area_metrics["largest_mask_area"]),
                            2,
                        ),
                        "area_by_class_json": json.dumps(
                            last_frame_area_metrics["area_by_class"],
                            ensure_ascii=False,
                        ),
                    }
                )

            annotated = draw_segmentations(
                frame.copy(),
                last_segments,
                counts=last_counts,
                show_boxes=show_boxes,
                roi_config=roi.as_dict(frame.shape) if roi else None,
            )
            writer.write(annotated)
            frame_index += 1
            if progress_callback:
                progress_callback(frame_index, total_frames)
    finally:
        cap.release()
        writer.release()

    if csv_output:
        _write_segmentation_rows_to_csv(rows, csv_output)

    processing_seconds = perf_counter() - processing_started
    average_processing_fps = frames_processed / processing_time_sum if processing_time_sum > 0 else 0.0
    result = VideoSegmentationResult(
        input_path=str(input_path),
        output_path=str(output_path),
        summary_path=str(summary_output) if summary_output else None,
        csv_output=str(csv_output) if csv_output else None,
        frames_read=frame_index,
        frames_processed=frames_processed,
        max_counts_by_class=dict(sorted(max_counts_by_class.items())),
        max_area_by_class=dict(sorted(max_area_by_class.items())),
        last_frame_counts=dict(sorted(last_counts.items())),
        max_frame_total=max_frame_total,
        last_frame_total=sum(last_counts.values()),
        last_frame_mask_area=last_frame_mask_area,
        max_frame_mask_area=max_frame_mask_area,
        max_frame_area_ratio=max_frame_area_ratio,
        largest_mask_area=largest_mask_area,
        last_frame_area_metrics=last_frame_area_metrics,
        average_processing_fps=average_processing_fps,
        processing_seconds=processing_seconds,
        roi_config=roi.as_dict() if roi else None,
    )

    if summary_output:
        write_json(result.to_dict(), summary_output)

    return result


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0

    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def segmentation_area_metrics(
    segments: Iterable[SegmentationMask],
    frame_shape: tuple[int, ...],
) -> dict[str, Any]:
    segment_list = list(segments)
    height = int(frame_shape[0]) if frame_shape else 0
    width = int(frame_shape[1]) if len(frame_shape) > 1 else 0
    image_area = float(max(width * height, 0))
    total_mask_area = sum(segment.mask_area for segment in segment_list)
    area_by_class: dict[str, float] = {}
    for segment in segment_list:
        area_by_class[segment.label] = area_by_class.get(segment.label, 0.0) + segment.mask_area

    largest_segment = max(segment_list, key=lambda item: item.mask_area, default=None)
    return {
        "image_area": round(image_area, 2),
        "total_mask_area": round(total_mask_area, 2),
        "mask_area_ratio": round(total_mask_area / image_area, 6) if image_area > 0 else 0.0,
        "average_mask_area": round(total_mask_area / len(segment_list), 2)
        if segment_list
        else 0.0,
        "largest_mask_area": round(largest_segment.mask_area, 2) if largest_segment else 0.0,
        "largest_mask_class": largest_segment.label if largest_segment else None,
        "area_by_class": {
            label: round(area, 2) for label, area in sorted(area_by_class.items())
        },
    }


def _mask_polygons(masks: Any) -> list[list[tuple[float, float]]]:
    if masks is None:
        return []

    raw_polygons = getattr(masks, "xy", None)
    if raw_polygons is None:
        return []

    polygons: list[list[tuple[float, float]]] = []
    for raw_polygon in raw_polygons:
        points = raw_polygon.tolist() if hasattr(raw_polygon, "tolist") else raw_polygon
        polygons.append([(float(x), float(y)) for x, y in points])
    return polygons


def _count_segments(segments: Iterable[SegmentationMask]) -> dict[str, int]:
    counter = Counter(segment.label for segment in segments)
    return dict(sorted(counter.items()))


def _filter_segments_by_roi(
    segments: list[SegmentationMask],
    roi: RegionOfInterest,
    frame_shape: tuple[int, ...],
) -> list[SegmentationMask]:
    x_min, y_min, x_max, y_max = roi.pixel_bounds(frame_shape)
    filtered = []
    for segment in segments:
        center_x, center_y = segment.center
        if x_min <= center_x <= x_max and y_min <= center_y <= y_max:
            filtered.append(segment)
    return filtered


def _progress_total(frame_count: int, max_frames: int | None) -> int:
    if frame_count > 0 and max_frames is not None:
        return min(frame_count, max_frames)
    if frame_count > 0:
        return frame_count
    return max_frames or 1


def _merge_max_counts(current: dict[str, int], new_counts: dict[str, int]) -> dict[str, int]:
    labels = set(current) | set(new_counts)
    return {label: max(int(current.get(label, 0)), int(new_counts.get(label, 0))) for label in labels}


def _merge_max_areas(current: dict[str, float], new_areas: dict[str, Any]) -> dict[str, float]:
    labels = set(current) | set(new_areas)
    return {
        label: max(float(current.get(label, 0.0)), float(new_areas.get(label, 0.0)))
        for label in labels
    }


def _write_segmentation_rows_to_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "frame_index",
        "timestamp_seconds",
        "processing_seconds",
        "fps_estimate",
        "frame_total",
        "frame_counts_json",
        "frame_mask_area",
        "frame_area_ratio",
        "frame_average_mask_area",
        "frame_largest_mask_area",
        "area_by_class_json",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
