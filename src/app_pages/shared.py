from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import pandas as pd
import streamlit as st

from object_counter.config import IMAGE_EXTENSIONS
from object_counter.counting.roi import RegionOfInterest
from object_counter.utils.downloads import (
    YOUTUBE_BLOCKED_MESSAGE,
    download_media_url,
    is_youtube_url,
    media_filename_from_url,
    streamlit_preview_url,
)
from object_counter.utils.io import infer_media_type
from object_counter.utils.video import ensure_browser_compatible_mp4


IMAGE_UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_UPLOAD_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]
MEDIA_UPLOAD_TYPES = [*IMAGE_UPLOAD_TYPES, *VIDEO_UPLOAD_TYPES]

YOLO_SIZE_INFO = {
    "n": {
        "label": "Nano",
        "use": "mais rápido e leve; bom para CPU, testes e demos curtas",
        "tradeoff": "menor qualidade em objetos pequenos, oclusão e cenas difíceis",
    },
    "s": {
        "label": "Small",
        "use": "equilíbrio entre velocidade e qualidade",
        "tradeoff": "mais lento que o nano, mas costuma ser mais estável",
    },
    "m": {
        "label": "Medium",
        "use": "mais pesado e geralmente mais preciso",
        "tradeoff": "melhor para GPU ou processamento offline com mais tempo",
    },
}

YOLO_TASK_INFO = {
    "segmentation": {
        "title": "Segmentação",
        "description": "modelos `-seg.pt` detectam objetos e desenham máscaras/polígonos.",
    },
    "pose": {
        "title": "Pose",
        "description": "modelos `-pose.pt` detectam pessoas e pontos do corpo/esqueleto.",
    },
}


def render_page_header(title: str, subtitle: str, pills: list[str]) -> None:
    pills_html = "".join(f'<span class="status-pill">{pill}</span>' for pill in pills)
    st.markdown(
        f"""
        <div class="app-title">
          <div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
          <div class="status-line">{pills_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_sample_options(samples: list[Path]) -> list[str]:
    return ["Nenhuma"] + [path.name for path in samples if path.suffix.lower() in IMAGE_EXTENSIONS]


def media_sample_options(samples: list[Path]) -> list[str]:
    return ["Nenhuma"] + [path.name for path in samples]


def model_matches_suffix(model_path: str, suffix: str) -> bool:
    return Path(model_path).name.lower().endswith(suffix)


def yolo_model_size_key(model_path: str) -> str | None:
    match = re.search(r"yolo(?:v)?\d+([nsm])", Path(model_path).name.lower())
    return match.group(1) if match else None


def yolo_model_size_summary(model_path: str) -> dict[str, str]:
    size_key = yolo_model_size_key(model_path)
    if size_key and size_key in YOLO_SIZE_INFO:
        return YOLO_SIZE_INFO[size_key]
    return {
        "label": "Customizado",
        "use": "peso treinado ou baixado fora da lista padrão",
        "tradeoff": "confirme se o sufixo bate com a tarefa da página",
    }


def yolo_model_option_rows(model_options: list[str], custom_suffix: str) -> list[dict[str, str]]:
    rows = []
    for model_option in model_options:
        size = yolo_model_size_summary(model_option)
        rows.append(
            {
                "model": model_option,
                "size": size["label"],
                "use": size["use"],
                "tradeoff": size["tradeoff"],
            }
        )
    rows.append(
        {
            "model": "Personalizado",
            "size": "Customizado",
            "use": f"um peso próprio compatível com `{custom_suffix}`",
            "tradeoff": "depende do dataset, classes e treinamento do modelo",
        }
    )
    return rows


def yolo_model_help_text(
    task: str,
    model_options: list[str],
    custom_suffix: str,
) -> str:
    task_info = YOLO_TASK_INFO[task]
    lines = [f"{task_info['title']}: {task_info['description']}"]
    for row in yolo_model_option_rows(model_options, custom_suffix):
        lines.append(f"{row['model']}: {row['size']} - {row['use']}.")
    return "\n".join(lines)


def render_roi_controls(prefix: str) -> dict[str, Any]:
    st.subheader("ROI")
    enabled = st.checkbox("Ativar ROI", value=False, key=f"{prefix}_roi_enabled")
    x_min = st.slider(
        "ROI x min.",
        min_value=0.0,
        max_value=0.9,
        value=0.0,
        disabled=not enabled,
        key=f"{prefix}_roi_x_min",
    )
    x_max = st.slider(
        "ROI x max.",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        disabled=not enabled,
        key=f"{prefix}_roi_x_max",
    )
    y_min = st.slider(
        "ROI y min.",
        min_value=0.0,
        max_value=0.9,
        value=0.0,
        disabled=not enabled,
        key=f"{prefix}_roi_y_min",
    )
    y_max = st.slider(
        "ROI y max.",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        disabled=not enabled,
        key=f"{prefix}_roi_y_max",
    )
    valid = not enabled or (x_min < x_max and y_min < y_max)
    if not valid:
        st.warning("A ROI precisa ter mínimos menores que máximos.")
    return {
        "roi_enabled": enabled,
        "roi": [x_min, y_min, x_max, y_max],
        "roi_valid": valid,
    }


def build_roi(config: dict[str, Any]) -> RegionOfInterest | None:
    if not config.get("roi_enabled"):
        return None
    return RegionOfInterest.from_values(config["roi"])


def create_progress_callback(label: str):
    progress = st.progress(0)
    status = st.empty()

    def update(current: int, total: int) -> None:
        total = max(total, 1)
        progress.progress(min(current / total, 1.0))
        status.caption(f"{label}: {min(current, total)} / {total} frames")

    return update


def resolve_selected_image(
    samples: list[Path],
    uploaded_file: Any,
    selected_sample: str,
) -> Path | None:
    if uploaded_file is not None:
        return Path(uploaded_file.name)
    if selected_sample == "Nenhuma":
        return None
    for sample in samples:
        if sample.name == selected_sample:
            return sample
    return None


def resolve_selected_media(
    samples: list[Path],
    uploaded_file: Any,
    selected_sample: str,
    media_url: str = "",
) -> Path | None:
    if uploaded_file is not None:
        return Path(uploaded_file.name)
    if media_url.strip():
        try:
            return Path(media_filename_from_url(media_url))
        except ValueError as exc:
            st.warning(str(exc))
            return None
    if selected_sample == "Nenhuma":
        return None
    for sample in samples:
        if sample.name == selected_sample:
            return sample
    return None


def active_input_source(uploaded_file: Any, selected_sample: str, media_url: str = "") -> tuple[str, list[str]]:
    sources = []
    if uploaded_file is not None:
        sources.append("upload")
    if media_url.strip():
        sources.append("url")
    if selected_sample != "Nenhuma":
        sources.append("amostra")

    if not sources:
        return "nenhuma", []

    priority = ["upload", "url", "amostra"]
    active = next(source for source in priority if source in sources)
    ignored = [source for source in sources if source != active]
    return active, ignored


def render_input_source_status(uploaded_file: Any, selected_sample: str, media_url: str = "") -> None:
    active, ignored = active_input_source(uploaded_file, selected_sample, media_url)
    labels = {
        "upload": "upload",
        "url": "URL",
        "amostra": "amostra local",
        "nenhuma": "nenhuma",
    }
    st.caption(f"Origem ativa: {labels[active]}")
    if active == "url" and is_youtube_url(media_url):
        st.warning(YOUTUBE_BLOCKED_MESSAGE)
    if ignored:
        ignored_labels = ", ".join(labels[source] for source in ignored)
        st.warning(f"Entradas ignoradas nesta execução: {ignored_labels}.")


def render_input_preview(input_path: Path | None, uploaded_file: Any, media_url: str = "") -> None:
    with st.container(border=True):
        st.subheader("Entrada")
        if input_path is None:
            st.info("Nenhuma mídia selecionada.")
            return

        media_type = infer_media_type(input_path)
        if media_url.strip() and uploaded_file is None:
            preview_url = streamlit_preview_url(media_url)
            if preview_url is None:
                st.info("Prévia remota indisponível antes do download.")
                return
            if media_type == "image":
                st.image(preview_url, width="stretch")
            else:
                st.video(preview_url)
            return

        if uploaded_file is not None and media_type == "image":
            st.image(uploaded_file.getvalue(), width="stretch")
            return
        if uploaded_file is not None and media_type == "video":
            st.video(uploaded_file.getvalue())
            return

        if media_type == "image":
            st.image(str(input_path), width="stretch")
        else:
            st.video(str(input_path))


def render_json_download(label: str, data: dict[str, Any], file_name: str) -> None:
    st.download_button(
        label,
        data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=file_name,
        mime="application/json",
    )


def render_image_download(label: str, output_path: str) -> None:
    path = Path(output_path)
    if not path.exists():
        return
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime="image/jpeg",
    )


def render_media_download(label: str, output_path: str) -> None:
    path = Path(output_path)
    if not path.exists():
        return
    mime = "video/mp4" if path.suffix.lower() == ".mp4" else "image/jpeg"
    st.download_button(label, data=path.read_bytes(), file_name=path.name, mime=mime)


def render_local_video_result(output_path: str | Path) -> None:
    path = Path(output_path)
    browser_path = ensure_browser_compatible_mp4(path)
    preview = video_first_frame_jpeg(browser_path)
    if preview is not None:
        st.image(preview, caption="Prévia do resultado processado", width="stretch")
    st.video(browser_path.read_bytes(), format="video/mp4")


def video_first_frame_jpeg(path: str | Path) -> bytes | None:
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        if not ok:
            return None
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return encoded.tobytes()
    finally:
        cap.release()


def render_csv_download(label: str, csv_path: str | None) -> None:
    if not csv_path:
        return
    path = Path(csv_path)
    if not path.exists():
        return
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/csv",
    )


def render_counts_table(counts: dict[str, int]) -> None:
    rows = [{"classe": label, "quantidade": count} for label, count in sorted(counts.items())]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def process_uploaded_or_sample(
    input_path: Path,
    uploaded_file: Any,
    media_url: str,
    output_dir: Path,
    suffix: str,
    processor: Callable[[Path, Path, Path], Any],
) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    with TemporaryDirectory() as tmpdir:
        actual_input = input_path
        if uploaded_file is not None:
            actual_input = Path(tmpdir) / uploaded_file.name
            actual_input.write_bytes(uploaded_file.getbuffer())
        elif media_url.strip():
            actual_input = download_media_url(media_url, tmpdir)

        output_path = output_dir / f"{actual_input.stem}_{run_id}_processado{suffix}"
        summary_path = output_dir / f"{actual_input.stem}_{run_id}_resumo.json"
        return processor(actual_input, output_path, summary_path)
