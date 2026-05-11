from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from object_counter.counting.counts import count_by_label, merge_max_counts, total_count
from object_counter.counting.line_counter import CountLine, LineCounter, LineDirection, LineOrientation
from object_counter.counting.roi import RegionOfInterest
from object_counter.detection.detector import YoloDetector
from object_counter.tracking.centroid_tracker import CentroidTracker
from object_counter.utils.io import ensure_parent_dir, write_json
from object_counter.utils.video import create_video_writer, open_video_capture
from object_counter.visualization.draw import draw_detections


@dataclass
class VideoCounterResult:
    input_path: str
    output_path: str
    summary_path: str | None
    csv_output: str | None
    frames_read: int
    frames_processed: int
    max_counts_by_class: dict[str, int]
    last_frame_counts: dict[str, int]
    line_counts_by_class: dict[str, int]
    total_line_crossings: int
    line_config: dict | None
    roi_config: dict | None
    average_processing_fps: float
    processing_seconds: float
    counting_mode: str
    tracking_backend: str

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "summary_path": self.summary_path,
            "csv_output": self.csv_output,
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "max_counts_by_class": self.max_counts_by_class,
            "last_frame_counts": self.last_frame_counts,
            "line_counts_by_class": self.line_counts_by_class,
            "total_line_crossings": self.total_line_crossings,
            "line_config": self.line_config,
            "roi_config": self.roi_config,
            "average_processing_fps": round(self.average_processing_fps, 4),
            "processing_seconds": round(self.processing_seconds, 4),
            "counting_mode": self.counting_mode,
            "tracking_backend": self.tracking_backend,
        }


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    detector: YoloDetector,
    classes: list[str] | None = None,
    csv_output: str | Path | None = None,
    summary_output: str | Path | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    counting_mode: str = "frame",
    line_orientation: LineOrientation = "horizontal",
    line_position: float = 0.5,
    line_direction: LineDirection = "both",
    tracker_max_distance: float = 80.0,
    tracker_max_missing: int = 10,
    roi: RegionOfInterest | None = None,
    tracking_backend: str = "centroid",
) -> VideoCounterResult:
    if frame_stride < 1:
        raise ValueError("frame_stride precisa ser maior ou igual a 1.")
    if counting_mode not in {"frame", "line"}:
        raise ValueError("counting_mode precisa ser 'frame' ou 'line'.")
    if tracking_backend not in {"centroid", "bytetrack"}:
        raise ValueError("tracking_backend precisa ser 'centroid' ou 'bytetrack'.")

    cap, metadata = open_video_capture(input_path)
    writer = create_video_writer(output_path, metadata)
    count_line = (
        CountLine(
            orientation=line_orientation,
            position_ratio=line_position,
            direction=line_direction,
        )
        if counting_mode == "line"
        else None
    )
    tracker = (
        CentroidTracker(max_distance=tracker_max_distance, max_missing=tracker_max_missing)
        if counting_mode == "line" and tracking_backend == "centroid"
        else None
    )
    line_counter = LineCounter(count_line) if count_line else None

    rows: list[dict] = []
    frame_index = 0
    frames_processed = 0
    max_counts_by_class: dict[str, int] = {}
    last_counts: dict[str, int] = {}
    last_detections = []
    processing_started = perf_counter()
    processing_time_sum = 0.0

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
                detections = (
                    detector.track(frame, classes=classes)
                    if counting_mode == "line" and tracking_backend == "bytetrack"
                    else detector.detect(frame, classes=classes)
                )
                if roi:
                    detections = roi.filter(detections, frame.shape)
                last_detections = tracker.update(detections) if tracker else detections
                frame_seconds = perf_counter() - frame_started
                processing_time_sum += frame_seconds
                frames_processed += 1

                last_counts = count_by_label(last_detections)
                max_counts_by_class = merge_max_counts(max_counts_by_class, last_counts)
                new_events = (
                    line_counter.update(last_detections, frame.shape, frame_index)
                    if line_counter
                    else []
                )
                line_counts = line_counter.counts() if line_counter else {}
                timestamp_seconds = frame_index / metadata.fps if metadata.fps > 0 else 0.0
                rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": round(timestamp_seconds, 4),
                        "processing_seconds": round(frame_seconds, 6),
                        "fps_estimate": round(1 / frame_seconds, 4) if frame_seconds > 0 else 0.0,
                        "frame_total": total_count(last_counts),
                        "frame_counts_json": json.dumps(last_counts, ensure_ascii=False),
                        "line_total": total_count(line_counts),
                        "line_counts_json": json.dumps(line_counts, ensure_ascii=False),
                        "new_events_json": json.dumps(
                            [event.to_dict() for event in new_events],
                            ensure_ascii=False,
                        ),
                    }
                )

            average_fps = frames_processed / processing_time_sum if processing_time_sum > 0 else None
            line_counts = line_counter.counts() if line_counter else {}
            display_counts = line_counts if counting_mode == "line" else last_counts
            counts_title = "Eventos contados" if counting_mode == "line" else "Contagem por frame"
            footer = (
                "Modo linha: conta cada ID uma vez ao cruzar a linha"
                if counting_mode == "line"
                else "Modo frame: contagem instantânea, sem eventos únicos"
            )
            annotated = draw_detections(
                frame.copy(),
                last_detections,
                counts=display_counts,
                counts_title=counts_title,
                fps=average_fps,
                line_config=count_line.as_dict(frame.shape) if count_line else None,
                roi_config=roi.as_dict(frame.shape) if roi else None,
                footer=footer,
            )
            writer.write(annotated)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if csv_output:
        _write_rows_to_csv(rows, csv_output)

    processing_seconds = perf_counter() - processing_started
    average_processing_fps = frames_processed / processing_time_sum if processing_time_sum > 0 else 0.0
    result = VideoCounterResult(
        input_path=str(input_path),
        output_path=str(output_path),
        summary_path=str(summary_output) if summary_output else None,
        csv_output=str(csv_output) if csv_output else None,
        frames_read=frame_index,
        frames_processed=frames_processed,
        max_counts_by_class=dict(sorted(max_counts_by_class.items())),
        last_frame_counts=dict(sorted(last_counts.items())),
        line_counts_by_class=line_counter.counts() if line_counter else {},
        total_line_crossings=line_counter.total() if line_counter else 0,
        line_config=count_line.as_dict() if count_line else None,
        roi_config=roi.as_dict() if roi else None,
        average_processing_fps=average_processing_fps,
        processing_seconds=processing_seconds,
        counting_mode=counting_mode,
        tracking_backend=tracking_backend if counting_mode == "line" else "none",
    )

    if summary_output:
        write_json(result.to_dict(), summary_output)

    return result


def _write_rows_to_csv(rows: list[dict], path: str | Path) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "frame_index",
        "timestamp_seconds",
        "processing_seconds",
        "fps_estimate",
        "frame_total",
        "frame_counts_json",
        "line_total",
        "line_counts_json",
        "new_events_json",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
