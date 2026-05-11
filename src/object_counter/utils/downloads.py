from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlencode, unquote, urlparse, urlunparse
from urllib.request import Request, urlopen
from typing import Any

from object_counter.config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from object_counter.utils.io import ensure_parent_dir

SUPPORTED_URL_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
DEFAULT_MAX_DOWNLOAD_BYTES = 250 * 1024 * 1024
YOUTUBE_BLOCKED_MESSAGE = (
    "O YouTube bloqueou o download server-side deste vídeo (HTTP 403). "
    "No Streamlit Cloud isso pode acontecer mesmo quando a prévia aparece no navegador. "
    "Use upload, um link direto .mp4, Google Drive/Dropbox público, ou tente outro vídeo."
)


def media_filename_from_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Use uma URL direta com protocolo http ou https.")

    if is_youtube_url(url):
        video_id = youtube_video_id(url) or "video"
        return sanitize_filename(f"youtube_{video_id}.mp4")

    if is_google_drive_url(url):
        file_id = google_drive_file_id(url) or "media"
        return sanitize_filename(f"google_drive_{file_id}.mp4")

    if is_dropbox_url(url):
        parsed = urlparse(dropbox_direct_url(url))

    filename = Path(unquote(parsed.path)).name
    if not filename:
        raise ValueError("A URL precisa apontar para um arquivo de mídia.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_URL_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_URL_EXTENSIONS))
        raise ValueError(f"Extensão da URL não suportada: {suffix}. Use uma destas: {supported}")

    return sanitize_filename(filename)


def download_media_url(
    url: str,
    output_dir: str | Path,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> Path:
    if is_youtube_url(url):
        return download_youtube_url(url, output_dir, max_bytes=max_bytes)
    if is_google_drive_url(url):
        return download_google_drive_url(url, output_dir, max_bytes=max_bytes)

    filename = media_filename_from_url(url)
    output_path = Path(output_dir) / filename
    download_url = dropbox_direct_url(url) if is_dropbox_url(url) else url.strip()
    return _download_http_url(download_url, output_path, max_bytes)


def download_google_drive_url(
    url: str,
    output_dir: str | Path,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> Path:
    file_id = google_drive_file_id(url)
    if not file_id:
        raise ValueError("Não foi possível identificar o ID do arquivo do Google Drive.")

    filename = media_filename_from_url(url)
    output_path = Path(output_dir) / filename
    query = urlencode({"export": "download", "id": file_id})
    download_url = urlunparse(("https", "drive.google.com", "/uc", "", query, ""))
    return _download_http_url(download_url, output_path, max_bytes)


def streamlit_preview_url(url: str) -> str | None:
    if is_google_drive_url(url):
        return None
    if is_dropbox_url(url):
        return dropbox_direct_url(url)
    return url.strip()


def _download_http_url(url: str, output_path: Path, max_bytes: int) -> Path:
    request = Request(
        url.strip(),
        headers={"User-Agent": "contador-objetos/0.1"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - local app, user-provided URL.
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type.lower():
            raise ValueError(
                "O link retornou uma página HTML em vez de mídia. "
                "Use um link público de download direto."
            )

        remote_name = filename_from_content_disposition(
            response.headers.get("Content-Disposition", "")
        )
        if remote_name:
            output_path = output_path.with_name(remote_name)
        ensure_parent_dir(output_path)

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError("Arquivo remoto maior que o limite permitido.")

        total_read = 0
        with output_path.open("wb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total_read += len(chunk)
                if total_read > max_bytes:
                    raise ValueError("Arquivo remoto maior que o limite permitido.")
                file.write(chunk)

    return output_path


def is_youtube_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()
    return hostname in {"youtu.be", "youtube.com", "www.youtube.com", "m.youtube.com"}


def youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()
    if hostname == "youtu.be":
        return Path(parsed.path).name or None
    if hostname.endswith("youtube.com"):
        query = parse_qs(parsed.query)
        values = query.get("v", [])
        return values[0] if values else None
    return None


def is_google_drive_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()
    return hostname in {"drive.google.com", "docs.google.com"}


def google_drive_file_id(url: str) -> str | None:
    parsed = urlparse(url.strip())
    query = parse_qs(parsed.query)
    values = query.get("id", [])
    if values:
        return values[0]

    parts = [part for part in parsed.path.split("/") if part]
    if "d" in parts:
        index = parts.index("d")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def is_dropbox_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()
    return hostname in {"dropbox.com", "www.dropbox.com", "dl.dropboxusercontent.com"}


def dropbox_direct_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if (parsed.hostname or "").lower() == "dl.dropboxusercontent.com":
        return url.strip()

    query = parse_qs(parsed.query)
    query["dl"] = ["1"]
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            urlencode(query, doseq=True),
            parsed.fragment,
        )
    )


def filename_from_content_disposition(value: str) -> str | None:
    if not value:
        return None
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', value, flags=re.IGNORECASE)
    if not match:
        return None
    filename = sanitize_filename(unquote(match.group(1)))
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_URL_EXTENSIONS:
        return None
    return filename


def download_youtube_url(
    url: str,
    output_dir: str | Path,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> Path:
    youtube_dl = _load_youtube_dl()
    filename = media_filename_from_url(url)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{Path(filename).stem}.%(ext)s")

    options = {
        "format": (
            "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/"
            "best[ext=mp4][height<=720]/best[height<=720]/best"
        ),
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "max_filesize": max_bytes,
        "retries": 2,
        "fragment_retries": 2,
        "socket_timeout": 30,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        },
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }
    try:
        with youtube_dl.YoutubeDL(options) as ydl:
            ydl.extract_info(url.strip(), download=True)
    except Exception as exc:
        if _is_youtube_blocked_error(exc):
            raise ValueError(YOUTUBE_BLOCKED_MESSAGE) from exc
        raise

    candidates = sorted(output_dir.glob(f"{Path(filename).stem}.*"))
    if not candidates:
        raise ValueError("Não foi possível baixar o vídeo do YouTube.")

    output_path = _prefer_mp4(candidates)
    if output_path.stat().st_size > max_bytes:
        raise ValueError("Arquivo do YouTube maior que o limite permitido.")
    return output_path


def _load_youtube_dl() -> Any:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "A dependência 'yt-dlp' não está instalada. Rode: pip install -r requirements.txt"
        ) from exc
    return yt_dlp


def _prefer_mp4(paths: list[Path]) -> Path:
    for path in paths:
        if path.suffix.lower() == ".mp4":
            return path
    return paths[0]


def _is_youtube_blocked_error(exc: Exception) -> bool:
    message = str(exc).lower()
    blocked_markers = [
        "403",
        "forbidden",
        "unable to download video data",
        "sign in to confirm",
        "not a bot",
    ]
    return any(marker in message for marker in blocked_markers)


def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    if not sanitized:
        raise ValueError("Nome de arquivo remoto inválido.")
    return sanitized
