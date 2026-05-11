from __future__ import annotations

from io import BytesIO

import pytest

from object_counter.utils import downloads


def test_media_filename_from_url_accepts_direct_media_url() -> None:
    filename = downloads.media_filename_from_url("https://example.com/media/video%20demo.mp4?x=1")

    assert filename == "video_demo.mp4"


def test_media_filename_from_url_rejects_pages_and_invalid_schemes() -> None:
    with pytest.raises(ValueError):
        downloads.media_filename_from_url("https://example.com/watch?v=123")

    with pytest.raises(ValueError):
        downloads.media_filename_from_url("file:///tmp/video.mp4")


def test_media_filename_from_url_accepts_youtube_url() -> None:
    filename = downloads.media_filename_from_url("https://www.youtube.com/watch?v=abc123")

    assert filename == "youtube_abc123.mp4"
    assert downloads.is_youtube_url("https://youtu.be/abc123")


def test_media_filename_from_url_accepts_google_drive_and_dropbox() -> None:
    drive_url = "https://drive.google.com/file/d/abc123/view?usp=sharing"
    dropbox_url = "https://www.dropbox.com/s/abc123/video%20demo.mp4?dl=0"

    assert downloads.media_filename_from_url(drive_url) == "google_drive_abc123.mp4"
    assert downloads.google_drive_file_id(drive_url) == "abc123"
    assert downloads.media_filename_from_url(dropbox_url) == "video_demo.mp4"
    assert "dl=1" in downloads.dropbox_direct_url(dropbox_url)


def test_download_media_url_writes_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(downloads, "urlopen", lambda request, timeout: _FakeResponse(b"demo"))

    path = downloads.download_media_url("https://example.com/video.mp4", tmp_path)

    assert path.name == "video.mp4"
    assert path.read_bytes() == b"demo"


def test_download_google_drive_url_uses_content_disposition(monkeypatch, tmp_path) -> None:
    response = _FakeResponse(
        b"drive",
        headers={
            "Content-Length": "5",
            "Content-Type": "video/mp4",
            "Content-Disposition": 'attachment; filename="drive video.mp4"',
        },
    )
    monkeypatch.setattr(downloads, "urlopen", lambda request, timeout: response)

    path = downloads.download_media_url(
        "https://drive.google.com/file/d/abc123/view?usp=sharing",
        tmp_path,
    )

    assert path.name == "drive_video.mp4"
    assert path.read_bytes() == b"drive"


def test_download_media_url_rejects_html_response(monkeypatch, tmp_path) -> None:
    response = _FakeResponse(
        b"<html></html>",
        headers={"Content-Length": "13", "Content-Type": "text/html"},
    )
    monkeypatch.setattr(downloads, "urlopen", lambda request, timeout: response)

    with pytest.raises(ValueError, match="página HTML"):
        downloads.download_media_url("https://example.com/video.mp4", tmp_path)


def test_download_media_url_uses_youtube_downloader(monkeypatch, tmp_path) -> None:
    _FakeYoutubeModule.last_options = None
    monkeypatch.setattr(downloads, "_load_youtube_dl", lambda: _FakeYoutubeModule)

    path = downloads.download_media_url("https://youtu.be/abc123", tmp_path)

    assert path.name == "youtube_abc123.mp4"
    assert path.read_bytes() == b"youtube"
    assert _FakeYoutubeModule.last_options["merge_output_format"] == "mp4"
    assert "http_headers" in _FakeYoutubeModule.last_options


def test_download_media_url_explains_youtube_403(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(downloads, "_load_youtube_dl", lambda: _FailingYoutubeModule)

    with pytest.raises(ValueError, match="YouTube bloqueou"):
        downloads.download_media_url("https://youtu.be/abc123", tmp_path)


class _FakeResponse:
    def __init__(self, payload: bytes, headers: dict[str, str] | None = None) -> None:
        self._buffer = BytesIO(payload)
        self.headers = headers or {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        self._buffer.close()

    def read(self, size: int) -> bytes:
        return self._buffer.read(size)


class _FakeYoutubeModule:
    last_options = None

    class YoutubeDL:
        def __init__(self, options):
            self.options = options
            _FakeYoutubeModule.last_options = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
            return None

        def extract_info(self, url: str, download: bool):  # noqa: FBT001
            output = self.options["outtmpl"].replace("%(ext)s", "mp4")
            with open(output, "wb") as file:
                file.write(b"youtube")
            return {"id": "abc123"}


class _FailingYoutubeModule:
    class YoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
            return None

        def extract_info(self, url: str, download: bool):  # noqa: FBT001
            raise RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
