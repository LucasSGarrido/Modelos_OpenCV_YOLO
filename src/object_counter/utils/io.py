from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from object_counter.config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

MediaType = Literal["image", "video"]


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def infer_media_type(path: str | Path) -> MediaType:
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
    raise ValueError(f"Extensão não suportada: {suffix}. Use uma destas: {supported}")


def read_image(path: str | Path) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV não está instalado. Rode: pip install -r requirements.txt") from exc

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Não foi possível abrir a imagem: {path}")
    return image


def save_image(path: str | Path, image: Any) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV não está instalado. Rode: pip install -r requirements.txt") from exc

    ensure_parent_dir(path)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise ValueError(f"Não foi possível salvar a imagem em: {path}")


def write_json(data: dict, path: str | Path) -> None:
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def default_output_path(input_path: str | Path, media_type: MediaType) -> Path:
    path = Path(input_path)
    stem = f"{path.stem}_processado"
    if media_type == "image":
        return Path("reports/figures") / f"{stem}.jpg"
    return Path("reports/videos") / f"{stem}.mp4"


def default_summary_path(output_path: str | Path) -> Path:
    return Path(output_path).with_suffix(".json")
