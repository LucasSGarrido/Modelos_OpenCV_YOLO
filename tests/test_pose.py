from __future__ import annotations

from pathlib import Path

import numpy as np

from object_counter.counting.roi import RegionOfInterest
from object_counter.pose import PoseEstimate, PoseKeypoint, process_pose_video
from object_counter.visualization.pose import draw_poses


def test_pose_estimate_counts_visible_keypoints() -> None:
    pose = PoseEstimate(
        x1=0,
        y1=0,
        x2=100,
        y2=100,
        confidence=0.9,
        class_id=0,
        label="person",
        keypoints=[
            PoseKeypoint(name="nose", x=10, y=10, confidence=0.8),
            PoseKeypoint(name="left_eye", x=12, y=9, confidence=0.1),
            PoseKeypoint(name="right_eye", x=8, y=9, confidence=0.7),
        ],
    )

    assert pose.visible_keypoints_count(min_confidence=0.25) == 2
    assert round(pose.average_keypoint_confidence() or 0, 4) == 0.5333


def test_pose_to_dict_includes_summary_fields() -> None:
    pose = PoseEstimate(
        x1=0,
        y1=0,
        x2=100,
        y2=100,
        confidence=0.9,
        class_id=0,
        label="person",
        keypoints=[PoseKeypoint(name="nose", x=10, y=10, confidence=0.8)],
    )

    data = pose.to_dict(min_confidence=0.25)

    assert data["label"] == "person"
    assert data["visible_keypoints"] == 1
    assert data["average_keypoint_confidence"] == 0.8


def test_draw_poses_returns_annotated_frame() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pose = PoseEstimate(
        x1=5,
        y1=5,
        x2=90,
        y2=95,
        confidence=0.9,
        class_id=0,
        label="person",
        keypoints=[
            PoseKeypoint(name="nose", x=50, y=20, confidence=0.9),
            PoseKeypoint(name="left_eye", x=45, y=18, confidence=0.8),
            PoseKeypoint(name="right_eye", x=55, y=18, confidence=0.8),
            PoseKeypoint(name="left_ear", x=40, y=22, confidence=0.8),
            PoseKeypoint(name="right_ear", x=60, y=22, confidence=0.8),
            PoseKeypoint(name="left_shoulder", x=35, y=45, confidence=0.9),
            PoseKeypoint(name="right_shoulder", x=65, y=45, confidence=0.9),
        ],
    )

    annotated = draw_poses(frame.copy(), [pose], counts={"person": 1})

    assert annotated.shape == frame.shape
    assert annotated.sum() > 0


def test_process_pose_video_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    csv_path = tmp_path / "frames.csv"
    summary_path = tmp_path / "summary.json"
    _write_tiny_video(input_path)

    result = process_pose_video(
        input_path=input_path,
        output_path=output_path,
        detector=_FakePoseDetector(),
        csv_output=csv_path,
        summary_output=summary_path,
        max_frames=3,
    )

    assert result.frames_processed == 3
    assert result.max_people == 1
    assert result.max_visible_keypoints == 2
    assert output_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()


def test_process_pose_video_filters_roi_and_reports_progress(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _write_tiny_video(input_path)
    progress_events: list[tuple[int, int]] = []

    result = process_pose_video(
        input_path=input_path,
        output_path=output_path,
        detector=_FakePoseDetectorWithOutsidePerson(),
        max_frames=3,
        roi=RegionOfInterest(0.0, 0.0, 0.5, 1.0),
        progress_callback=lambda current, total: progress_events.append((current, total)),
    )

    assert result.max_people == 1
    assert result.max_visible_keypoints == 2
    assert result.roi_config == {"x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 1.0}
    assert progress_events[-1] == (3, 3)


class _FakePoseDetector:
    def detect_poses(self, frame):  # noqa: ANN001, ANN201
        return [
            PoseEstimate(
                x1=5,
                y1=5,
                x2=35,
                y2=40,
                confidence=0.9,
                class_id=0,
                label="person",
                keypoints=[
                    PoseKeypoint(name="nose", x=20, y=10, confidence=0.9),
                    PoseKeypoint(name="left_eye", x=18, y=9, confidence=0.8),
                ],
            )
        ]


class _FakePoseDetectorWithOutsidePerson:
    def detect_poses(self, frame):  # noqa: ANN001, ANN201
        return [
            PoseEstimate(
                x1=5,
                y1=5,
                x2=20,
                y2=35,
                confidence=0.9,
                class_id=0,
                label="person",
                keypoints=[
                    PoseKeypoint(name="nose", x=12, y=10, confidence=0.9),
                    PoseKeypoint(name="left_eye", x=10, y=9, confidence=0.8),
                ],
            ),
            PoseEstimate(
                x1=30,
                y1=5,
                x2=45,
                y2=35,
                confidence=0.9,
                class_id=0,
                label="person",
                keypoints=[
                    PoseKeypoint(name="nose", x=38, y=10, confidence=0.9),
                    PoseKeypoint(name="left_eye", x=36, y=9, confidence=0.8),
                ],
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
