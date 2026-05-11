from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from object_counter.segmentation.segmenter import SegmentationMask


def draw_segmentations(
    frame: Any,
    segments: Iterable[SegmentationMask],
    counts: Mapping[str, int] | None = None,
    show_boxes: bool = True,
    roi_config: Mapping[str, Any] | None = None,
) -> Any:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV e NumPy precisam estar instalados.") from exc

    segments = list(segments)
    counts = dict(counts or {})
    overlay = frame.copy()

    if roi_config:
        _draw_roi(cv2, frame, roi_config)

    for segment in segments:
        if len(segment.polygon) < 3:
            continue

        color = _color_for_label(segment.label)
        points = np.array(segment.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [points], color)

    cv2.addWeighted(overlay, 0.34, frame, 0.66, 0, frame)

    for segment in segments:
        color = _color_for_label(segment.label)
        if len(segment.polygon) >= 3:
            points = np.array(segment.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)

        x1, y1, x2, y2 = segment.bbox_xyxy
        if show_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{segment.label} {segment.confidence:.2f} area:{segment.mask_area:.0f}"
        _draw_label(cv2, frame, label, x1, max(0, y1 - 8), color)

    _draw_counter_panel(cv2, frame, counts, title="Mascaras")
    return frame


def _color_for_label(label: str) -> tuple[int, int, int]:
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    return (
        80 + int(digest[0:2], 16) % 160,
        80 + int(digest[2:4], 16) % 160,
        80 + int(digest[4:6], 16) % 160,
    )


def _draw_label(cv2: Any, frame: Any, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    top_left = (x, max(0, y - text_height - baseline - 4))
    bottom_right = (x + text_width + 8, y + baseline)
    cv2.rectangle(frame, top_left, bottom_right, color, -1)
    cv2.putText(
        frame,
        text,
        (x + 4, y - 4),
        font,
        font_scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )


def _draw_counter_panel(
    cv2: Any,
    frame: Any,
    counts: Mapping[str, int],
    title: str,
) -> None:
    panel_lines = [title]
    panel_lines.extend(f"{label}: {count}" for label, count in sorted(counts.items()))
    panel_lines.append(f"Total: {sum(counts.values())}")

    line_height = 22
    width = 260
    height = 16 + line_height * len(panel_lines)
    x1, y1 = 10, 10
    x2, y2 = x1 + width, y1 + height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    for index, line in enumerate(panel_lines):
        color = (255, 255, 255) if index == 0 else (220, 235, 255)
        cv2.putText(
            frame,
            line,
            (x1 + 12, y1 + 24 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_roi(cv2: Any, frame: Any, roi_config: Mapping[str, Any]) -> None:
    x_min = int(roi_config.get("x_min_px", 0))
    y_min = int(roi_config.get("y_min_px", 0))
    x_max = int(roi_config.get("x_max_px", frame.shape[1]))
    y_max = int(roi_config.get("y_max_px", frame.shape[0]))
    color = (30, 180, 90)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(
        frame,
        "ROI",
        (x_min + 8, max(24, y_min + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
