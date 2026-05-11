from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from object_counter.detection.detector import Detection


def draw_detections(
    frame: Any,
    detections: Iterable[Detection],
    counts: Mapping[str, int] | None = None,
    counts_title: str = "Contagem",
    fps: float | None = None,
    line_config: Mapping[str, Any] | None = None,
    roi_config: Mapping[str, Any] | None = None,
    footer: str | None = None,
) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV não está instalado. Rode: pip install -r requirements.txt") from exc

    detections = list(detections)
    counts = dict(counts or {})

    if line_config:
        _draw_count_line(cv2, frame, line_config)
    if roi_config:
        _draw_roi(cv2, frame, roi_config)

    for detection in detections:
        color = _color_for_label(detection.label)
        x1, y1, x2, y2 = detection.bbox_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{detection.label} {detection.confidence:.2f}"
        if detection.track_id is not None:
            label = f"{label} id:{detection.track_id}"
        _draw_label(cv2, frame, label, x1, max(0, y1 - 8), color)

    _draw_counter_panel(cv2, frame, counts, title=counts_title, fps=fps)

    if footer:
        cv2.putText(
            frame,
            footer,
            (12, frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

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
    title: str = "Contagem",
    fps: float | None = None,
) -> None:
    panel_lines = [title]
    panel_lines.extend(f"{label}: {count}" for label, count in sorted(counts.items()))
    panel_lines.append(f"Total: {sum(counts.values())}")
    if fps is not None:
        panel_lines.append(f"FPS proc.: {fps:.1f}")

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


def _draw_count_line(cv2: Any, frame: Any, line_config: Mapping[str, Any]) -> None:
    height, width = frame.shape[:2]
    orientation = str(line_config.get("orientation", "horizontal"))
    position = int(line_config.get("pixel_position", height // 2))
    direction = str(line_config.get("direction", "both"))
    color = (40, 220, 255)

    if orientation == "vertical":
        start = (position, 0)
        end = (position, height)
        label_position = (min(position + 8, width - 160), 28)
    else:
        start = (0, position)
        end = (width, position)
        label_position = (12, max(28, position - 10))

    cv2.line(frame, start, end, color, 3)
    cv2.putText(
        frame,
        f"linha: {direction}",
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
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
