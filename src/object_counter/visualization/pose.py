from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from object_counter.pose.pose_detector import PoseEstimate


COCO_SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def draw_poses(
    frame: Any,
    poses: Iterable["PoseEstimate"],
    counts: Mapping[str, int] | None = None,
    keypoint_confidence: float = 0.25,
    roi_config: Mapping[str, Any] | None = None,
) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV precisa estar instalado.") from exc

    poses = list(poses)
    counts = dict(counts or {})

    if roi_config:
        _draw_roi(cv2, frame, roi_config)

    for pose in poses:
        x1, y1, x2, y2 = pose.bbox_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (38, 38, 38), 2)
        label = (
            f"{pose.label} {pose.confidence:.2f} "
            f"kps:{pose.visible_keypoints_count(keypoint_confidence)}"
        )
        _draw_label(cv2, frame, label, x1, max(0, y1 - 8), (38, 38, 38))

        for start_index, end_index in COCO_SKELETON:
            if start_index >= len(pose.keypoints) or end_index >= len(pose.keypoints):
                continue
            start = pose.keypoints[start_index]
            end = pose.keypoints[end_index]
            if not start.is_visible(keypoint_confidence) or not end.is_visible(keypoint_confidence):
                continue
            cv2.line(
                frame,
                (int(start.x), int(start.y)),
                (int(end.x), int(end.y)),
                (40, 190, 255),
                2,
                cv2.LINE_AA,
            )

        for keypoint in pose.keypoints:
            if not keypoint.is_visible(keypoint_confidence):
                continue
            cv2.circle(frame, (int(keypoint.x), int(keypoint.y)), 4, (20, 20, 20), -1)
            cv2.circle(frame, (int(keypoint.x), int(keypoint.y)), 3, (255, 255, 255), -1)

    _draw_counter_panel(cv2, frame, counts, title="Poses")
    return frame


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
        (255, 255, 255),
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
