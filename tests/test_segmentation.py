from __future__ import annotations

from pathlib import Path

import numpy as np

from object_counter.counting.roi import RegionOfInterest
from object_counter.segmentation import (
    SegmentationMask,
    polygon_area,
    process_segmentation_video,
    segmentation_area_metrics,
)
from object_counter.visualization.segmentation import draw_segmentations


def test_polygon_area_square() -> None:
    polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    assert polygon_area(polygon) == 100.0


def test_segmentation_mask_to_dict_includes_areas() -> None:
    segment = SegmentationMask(
        x1=0,
        y1=0,
        x2=20,
        y2=10,
        confidence=0.8,
        class_id=0,
        label="person",
        polygon=[(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)],
        mask_area=200.0,
    )

    data = segment.to_dict()

    assert data["label"] == "person"
    assert data["mask_area"] == 200.0
    assert data["bbox_area"] == 200.0
    assert data["polygon"][0] == [0.0, 0.0]


def test_segmentation_area_metrics_groups_area_by_class() -> None:
    segments = [
        SegmentationMask(
            x1=0,
            y1=0,
            x2=20,
            y2=10,
            confidence=0.8,
            class_id=0,
            label="person",
            polygon=[(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)],
            mask_area=200.0,
        ),
        SegmentationMask(
            x1=30,
            y1=0,
            x2=40,
            y2=10,
            confidence=0.7,
            class_id=5,
            label="bus",
            polygon=[(30.0, 0.0), (40.0, 0.0), (40.0, 10.0), (30.0, 10.0)],
            mask_area=100.0,
        ),
    ]

    metrics = segmentation_area_metrics(segments, (30, 40, 3))

    assert metrics["total_mask_area"] == 300.0
    assert metrics["mask_area_ratio"] == 0.25
    assert metrics["largest_mask_class"] == "person"
    assert metrics["area_by_class"] == {"bus": 100.0, "person": 200.0}


def test_draw_segmentations_returns_annotated_frame() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    segment = SegmentationMask(
        x1=10,
        y1=10,
        x2=50,
        y2=50,
        confidence=0.9,
        class_id=0,
        label="person",
        polygon=[(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        mask_area=1600.0,
    )

    annotated = draw_segmentations(frame.copy(), [segment], counts={"person": 1})

    assert annotated.shape == frame.shape
    assert annotated.sum() > 0


def test_process_segmentation_video_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    csv_path = tmp_path / "frames.csv"
    summary_path = tmp_path / "summary.json"
    _write_tiny_video(input_path)

    result = process_segmentation_video(
        input_path=input_path,
        output_path=output_path,
        segmenter=_FakeSegmenter(),
        csv_output=csv_path,
        summary_output=summary_path,
        max_frames=3,
    )

    assert result.frames_processed == 3
    assert result.max_counts_by_class == {"person": 1}
    assert result.max_area_by_class == {"person": 400.0}
    assert result.max_frame_mask_area == 400.0
    assert result.max_frame_area_ratio > 0
    assert output_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()


def test_process_segmentation_video_filters_roi_and_reports_progress(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _write_tiny_video(input_path)
    progress_events: list[tuple[int, int]] = []

    result = process_segmentation_video(
        input_path=input_path,
        output_path=output_path,
        segmenter=_FakeSegmenterWithOutsideObject(),
        max_frames=3,
        roi=RegionOfInterest(0.0, 0.0, 0.5, 1.0),
        progress_callback=lambda current, total: progress_events.append((current, total)),
    )

    assert result.max_counts_by_class == {"person": 1}
    assert result.roi_config == {"x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 1.0}
    assert progress_events[-1] == (3, 3)


class _FakeSegmenter:
    def segment(self, frame, classes=None):  # noqa: ANN001, ANN201
        return [
            SegmentationMask(
                x1=5,
                y1=5,
                x2=25,
                y2=25,
                confidence=0.9,
                class_id=0,
                label="person",
                polygon=[(5.0, 5.0), (25.0, 5.0), (25.0, 25.0), (5.0, 25.0)],
                mask_area=400.0,
            )
        ]


class _FakeSegmenterWithOutsideObject:
    def segment(self, frame, classes=None):  # noqa: ANN001, ANN201
        return [
            SegmentationMask(
                x1=5,
                y1=5,
                x2=20,
                y2=20,
                confidence=0.9,
                class_id=0,
                label="person",
                polygon=[(5.0, 5.0), (20.0, 5.0), (20.0, 20.0), (5.0, 20.0)],
                mask_area=225.0,
            ),
            SegmentationMask(
                x1=30,
                y1=5,
                x2=45,
                y2=20,
                confidence=0.9,
                class_id=5,
                label="bus",
                polygon=[(30.0, 5.0), (45.0, 5.0), (45.0, 20.0), (30.0, 20.0)],
                mask_area=225.0,
            ),
        ]


def _write_tiny_video(path: Path) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (48, 48))
    assert writer.isOpened()
    for index in range(3):
        frame = np.full((48, 48, 3), index * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()
