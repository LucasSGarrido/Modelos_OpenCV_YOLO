from __future__ import annotations

from pathlib import Path

import numpy as np

from app_pages.shared import (
    active_input_source,
    model_matches_suffix,
    video_first_frame_jpeg,
    yolo_model_option_rows,
    yolo_model_help_text,
    yolo_model_size_key,
    yolo_model_size_summary,
)


def test_model_matches_suffix_uses_file_name() -> None:
    assert model_matches_suffix("models/yolov8n-seg.pt", "-seg.pt")
    assert model_matches_suffix("C:/models/yolov8n-pose.pt", "-pose.pt")
    assert not model_matches_suffix("yolov8n.pt", "-seg.pt")


def test_active_input_source_uses_upload_url_sample_priority() -> None:
    upload = object()

    active, ignored = active_input_source(upload, "bus.jpg", "https://example.com/video.mp4")

    assert active == "upload"
    assert ignored == ["url", "amostra"]


def test_yolo_model_size_helpers_describe_default_options() -> None:
    assert yolo_model_size_key("models/yolov8m-pose.pt") == "m"
    assert yolo_model_size_summary("yolov8s-seg.pt")["label"] == "Small"
    assert yolo_model_size_summary("custom-seg.pt")["label"] == "Customizado"

    rows = yolo_model_option_rows(["yolov8n-seg.pt", "yolov8s-seg.pt"], "-seg.pt")

    assert rows[0]["size"] == "Nano"
    assert rows[1]["size"] == "Small"
    assert rows[-1]["model"] == "Personalizado"
    assert "-seg.pt" in rows[-1]["use"]

    help_text = yolo_model_help_text("pose", ["yolov8n-pose.pt"], "-pose.pt")
    assert "Pose" in help_text
    assert "yolov8n-pose.pt" in help_text
    assert "Personalizado" in help_text


def test_video_first_frame_jpeg_returns_bytes(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mp4"
    _write_tiny_video(path)

    preview = video_first_frame_jpeg(path)

    assert preview is not None
    assert preview.startswith(b"\xff\xd8")


def _write_tiny_video(path: Path) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (24, 24))
    assert writer.isOpened()
    writer.write(np.full((24, 24, 3), 120, dtype=np.uint8))
    writer.release()
