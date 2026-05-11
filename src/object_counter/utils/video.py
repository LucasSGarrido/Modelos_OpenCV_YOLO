from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from object_counter.utils.io import ensure_parent_dir


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV não está instalado. Rode: pip install -r requirements.txt") from exc
    return cv2


def open_video_capture(path: str | Path) -> tuple[Any, VideoMetadata]:
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata = VideoMetadata(width=width, height=height, fps=fps, frame_count=frame_count)
    return cap, metadata


def create_video_writer(path: str | Path, metadata: VideoMetadata) -> Any:
    cv2 = _import_cv2()
    ensure_parent_dir(path)
    suffix = Path(path).suffix.lower()
    fourcc_name = "mp4v" if suffix in {".mp4", ".m4v", ".mov"} else "XVID"
    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
    writer = cv2.VideoWriter(str(path), fourcc, metadata.fps, (metadata.width, metadata.height))
    if not writer.isOpened():
        raise ValueError(f"Não foi possível criar o vídeo de saída: {path}")
    return writer


def ensure_browser_compatible_mp4(path: str | Path) -> Path:
    """Create an H.264/yuv420p copy for HTML video playback when possible."""

    source = Path(path)
    if source.suffix.lower() != ".mp4" or not source.exists():
        return source

    output = source.with_name(f"{source.stem}_web.mp4")
    if output.exists() and output.stat().st_mtime >= source.stat().st_mtime:
        return output

    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        return source

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output.exists():
        return source
    return output


def ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable

    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()
