from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from object_counter.counting.roi import RegionOfInterest
from object_counter.detection.detector import _class_label
from object_counter.utils.io import ensure_parent_dir, read_image, save_image, write_json
from object_counter.utils.video import create_video_writer, open_video_capture
from object_counter.visualization.pose import draw_poses


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


@dataclass(frozen=True)
class PoseKeypoint:
    name: str
    x: float
    y: float
    confidence: float | None = None

    def is_visible(self, min_confidence: float = 0.25) -> bool:
        if self.confidence is None:
            return self.x > 0 or self.y > 0
        return self.confidence >= min_confidence

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["x"] = round(self.x, 2)
        data["y"] = round(self.y, 2)
        if self.confidence is not None:
            data["confidence"] = round(self.confidence, 4)
        return data


@dataclass(frozen=True)
class PoseEstimate:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str
    keypoints: list[PoseKeypoint]

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def visible_keypoints_count(self, min_confidence: float = 0.25) -> int:
        return sum(keypoint.is_visible(min_confidence) for keypoint in self.keypoints)

    def average_keypoint_confidence(self) -> float | None:
        confidences = [
            keypoint.confidence for keypoint in self.keypoints if keypoint.confidence is not None
        ]
        if not confidences:
            return None
        return sum(confidences) / len(confidences)

    def to_dict(self, min_confidence: float = 0.25) -> dict[str, Any]:
        average_confidence = self.average_keypoint_confidence()
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "label": self.label,
            "visible_keypoints": self.visible_keypoints_count(min_confidence),
            "average_keypoint_confidence": (
                round(average_confidence, 4) if average_confidence is not None else None
            ),
            "keypoints": [keypoint.to_dict() for keypoint in self.keypoints],
        }


@dataclass
class ImagePoseResult:
    input_path: str
    output_path: str
    summary_path: str | None
    counts: dict[str, int]
    total: int
    visible_keypoints: int
    average_keypoint_confidence: float | None
    inference_seconds: float
    poses: list[dict]
    roi_config: dict | None = None
    model_task: str = "keypoint_detection"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_task": self.model_task,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "summary_path": self.summary_path,
            "counts": self.counts,
            "total": self.total,
            "visible_keypoints": self.visible_keypoints,
            "average_keypoint_confidence": (
                round(self.average_keypoint_confidence, 4)
                if self.average_keypoint_confidence is not None
                else None
            ),
            "inference_seconds": round(self.inference_seconds, 6),
            "poses": self.poses,
            "roi_config": self.roi_config,
        }


@dataclass
class VideoPoseResult:
    input_path: str
    output_path: str
    summary_path: str | None
    csv_output: str | None
    frames_read: int
    frames_processed: int
    max_people: int
    last_frame_people: int
    max_visible_keypoints: int
    last_frame_visible_keypoints: int
    average_keypoint_confidence: float | None
    average_processing_fps: float
    processing_seconds: float
    roi_config: dict | None = None
    model_task: str = "keypoint_detection_video"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_task": self.model_task,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "summary_path": self.summary_path,
            "csv_output": self.csv_output,
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "max_people": self.max_people,
            "last_frame_people": self.last_frame_people,
            "max_visible_keypoints": self.max_visible_keypoints,
            "last_frame_visible_keypoints": self.last_frame_visible_keypoints,
            "average_keypoint_confidence": (
                round(self.average_keypoint_confidence, 4)
                if self.average_keypoint_confidence is not None
                else None
            ),
            "average_processing_fps": round(self.average_processing_fps, 4),
            "processing_seconds": round(self.processing_seconds, 4),
            "roi_config": self.roi_config,
        }


class YoloPoseDetector:
    """Thin wrapper around Ultralytics YOLO pose models."""

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

    def detect_poses(self, frame: Any) -> list[PoseEstimate]:
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
        keypoint_xy, keypoint_conf = _extract_keypoints(getattr(result, "keypoints", None))

        poses: list[PoseEstimate] = []
        for index, box in enumerate(boxes):
            class_id = int(box.cls[0].item())
            label = _class_label(names, class_id)
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf[0].item())
            pose_keypoints = _build_pose_keypoints(
                keypoint_xy[index] if index < len(keypoint_xy) else [],
                keypoint_conf[index] if index < len(keypoint_conf) else [],
            )
            poses.append(
                PoseEstimate(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    label=label,
                    keypoints=pose_keypoints,
                )
            )

        return poses


def process_pose_image(
    input_path: str | Path,
    output_path: str | Path,
    detector: YoloPoseDetector,
    summary_output: str | Path | None = None,
    keypoint_confidence: float = 0.25,
    roi: RegionOfInterest | None = None,
) -> ImagePoseResult:
    image = read_image(input_path)

    started_at = perf_counter()
    poses = detector.detect_poses(image)
    if roi:
        poses = _filter_poses_by_roi(poses, roi, image.shape)
    inference_seconds = perf_counter() - started_at

    counts = {"person": len(poses)}
    annotated = draw_poses(
        image.copy(),
        poses,
        counts=counts,
        keypoint_confidence=keypoint_confidence,
        roi_config=roi.as_dict(image.shape) if roi else None,
    )
    save_image(output_path, annotated)

    visible_keypoints = sum(
        pose.visible_keypoints_count(keypoint_confidence) for pose in poses
    )
    average_confidence = _average_pose_confidence(poses)
    result = ImagePoseResult(
        input_path=str(input_path),
        output_path=str(output_path),
        summary_path=str(summary_output) if summary_output else None,
        counts=counts,
        total=len(poses),
        visible_keypoints=visible_keypoints,
        average_keypoint_confidence=average_confidence,
        inference_seconds=inference_seconds,
        poses=[pose.to_dict(keypoint_confidence) for pose in poses],
        roi_config=roi.as_dict() if roi else None,
    )

    if summary_output:
        write_json(result.to_dict(), summary_output)

    return result


def process_pose_video(
    input_path: str | Path,
    output_path: str | Path,
    detector: YoloPoseDetector,
    csv_output: str | Path | None = None,
    summary_output: str | Path | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    keypoint_confidence: float = 0.25,
    roi: RegionOfInterest | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> VideoPoseResult:
    if frame_stride < 1:
        raise ValueError("frame_stride precisa ser maior ou igual a 1.")

    cap, metadata = open_video_capture(input_path)
    writer = create_video_writer(output_path, metadata)

    rows: list[dict[str, Any]] = []
    frame_index = 0
    frames_processed = 0
    max_people = 0
    last_frame_people = 0
    max_visible_keypoints = 0
    last_frame_visible_keypoints = 0
    last_poses: list[PoseEstimate] = []
    confidence_samples: list[float] = []
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
                last_poses = detector.detect_poses(frame)
                if roi:
                    last_poses = _filter_poses_by_roi(last_poses, roi, frame.shape)
                frame_seconds = perf_counter() - frame_started
                processing_time_sum += frame_seconds
                frames_processed += 1

                last_frame_people = len(last_poses)
                max_people = max(max_people, last_frame_people)
                last_frame_visible_keypoints = sum(
                    pose.visible_keypoints_count(keypoint_confidence) for pose in last_poses
                )
                max_visible_keypoints = max(max_visible_keypoints, last_frame_visible_keypoints)
                average_confidence = _average_pose_confidence(last_poses)
                if average_confidence is not None:
                    confidence_samples.append(average_confidence)

                timestamp_seconds = frame_index / metadata.fps if metadata.fps > 0 else 0.0
                rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": round(timestamp_seconds, 4),
                        "processing_seconds": round(frame_seconds, 6),
                        "fps_estimate": round(1 / frame_seconds, 4)
                        if frame_seconds > 0
                        else 0.0,
                        "people_total": last_frame_people,
                        "visible_keypoints": last_frame_visible_keypoints,
                        "average_keypoint_confidence": (
                            round(average_confidence, 4)
                            if average_confidence is not None
                            else None
                        ),
                        "poses_json": json.dumps(
                            [pose.to_dict(keypoint_confidence) for pose in last_poses],
                            ensure_ascii=False,
                        ),
                    }
                )

            annotated = draw_poses(
                frame.copy(),
                last_poses,
                counts={"person": last_frame_people},
                keypoint_confidence=keypoint_confidence,
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
        _write_pose_rows_to_csv(rows, csv_output)

    processing_seconds = perf_counter() - processing_started
    average_processing_fps = frames_processed / processing_time_sum if processing_time_sum > 0 else 0.0
    overall_keypoint_confidence = (
        sum(confidence_samples) / len(confidence_samples) if confidence_samples else None
    )
    result = VideoPoseResult(
        input_path=str(input_path),
        output_path=str(output_path),
        summary_path=str(summary_output) if summary_output else None,
        csv_output=str(csv_output) if csv_output else None,
        frames_read=frame_index,
        frames_processed=frames_processed,
        max_people=max_people,
        last_frame_people=last_frame_people,
        max_visible_keypoints=max_visible_keypoints,
        last_frame_visible_keypoints=last_frame_visible_keypoints,
        average_keypoint_confidence=overall_keypoint_confidence,
        average_processing_fps=average_processing_fps,
        processing_seconds=processing_seconds,
        roi_config=roi.as_dict() if roi else None,
    )

    if summary_output:
        write_json(result.to_dict(), summary_output)

    return result


def _extract_keypoints(keypoints: Any) -> tuple[list[list[list[float]]], list[list[float]]]:
    if keypoints is None:
        return [], []

    xy = getattr(keypoints, "xy", None)
    conf = getattr(keypoints, "conf", None)

    xy_values = xy.cpu().numpy().tolist() if xy is not None else []
    conf_values = conf.cpu().numpy().tolist() if conf is not None else []
    return xy_values, conf_values


def _build_pose_keypoints(points: list[list[float]], confidences: list[float]) -> list[PoseKeypoint]:
    pose_keypoints: list[PoseKeypoint] = []
    for index, point in enumerate(points):
        if len(point) < 2:
            continue

        confidence = confidences[index] if index < len(confidences) else None
        name = COCO_KEYPOINT_NAMES[index] if index < len(COCO_KEYPOINT_NAMES) else f"keypoint_{index}"
        pose_keypoints.append(
            PoseKeypoint(
                name=name,
                x=float(point[0]),
                y=float(point[1]),
                confidence=float(confidence) if confidence is not None else None,
            )
        )
    return pose_keypoints


def _average_pose_confidence(poses: list[PoseEstimate]) -> float | None:
    confidences = [
        confidence
        for pose in poses
        for confidence in [pose.average_keypoint_confidence()]
        if confidence is not None
    ]
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def _filter_poses_by_roi(
    poses: list[PoseEstimate],
    roi: RegionOfInterest,
    frame_shape: tuple[int, ...],
) -> list[PoseEstimate]:
    x_min, y_min, x_max, y_max = roi.pixel_bounds(frame_shape)
    filtered = []
    for pose in poses:
        center_x, center_y = pose.center
        if x_min <= center_x <= x_max and y_min <= center_y <= y_max:
            filtered.append(pose)
    return filtered


def _progress_total(frame_count: int, max_frames: int | None) -> int:
    if frame_count > 0 and max_frames is not None:
        return min(frame_count, max_frames)
    if frame_count > 0:
        return frame_count
    return max_frames or 1


def _write_pose_rows_to_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "frame_index",
        "timestamp_seconds",
        "processing_seconds",
        "fps_estimate",
        "people_total",
        "visible_keypoints",
        "average_keypoint_confidence",
        "poses_json",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
