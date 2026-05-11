from __future__ import annotations

from pathlib import Path

from object_counter.utils import video


def test_ensure_browser_compatible_mp4_uses_cached_web_copy(tmp_path: Path) -> None:
    source = tmp_path / "result.mp4"
    source.write_bytes(b"source")
    web = tmp_path / "result_web.mp4"
    web.write_bytes(b"web")

    assert video.ensure_browser_compatible_mp4(source) == web


def test_ensure_browser_compatible_mp4_returns_source_without_ffmpeg(monkeypatch, tmp_path) -> None:
    source = tmp_path / "result.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(video, "ffmpeg_executable", lambda: None)

    assert video.ensure_browser_compatible_mp4(source) == source


def test_ensure_browser_compatible_mp4_creates_web_copy(monkeypatch, tmp_path) -> None:
    source = tmp_path / "result.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(video, "ffmpeg_executable", lambda: "ffmpeg")

    def fake_run(command, capture_output, text, check):  # noqa: ANN001, ANN202, FBT001
        Path(command[-1]).write_bytes(b"web")

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(video.subprocess, "run", fake_run)

    output = video.ensure_browser_compatible_mp4(source)

    assert output.name == "result_web.mp4"
    assert output.read_bytes() == b"web"
