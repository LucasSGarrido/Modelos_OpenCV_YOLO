from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app_pages.shared import (
    MEDIA_UPLOAD_TYPES,
    build_roi,
    create_progress_callback,
    media_sample_options,
    model_matches_suffix,
    process_uploaded_or_sample,
    render_csv_download,
    render_counts_table,
    render_input_preview,
    render_input_source_status,
    render_json_download,
    render_local_video_result,
    render_media_download,
    render_page_header,
    render_roi_controls,
    resolve_selected_media,
    yolo_model_help_text,
)
from object_counter.evaluation import pose_diagnostic_rows
from object_counter.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_POSE_MODEL_PATH,
    POSE_MODEL_OPTIONS,
)
from object_counter.pose import YoloPoseDetector, process_pose_image, process_pose_video
from object_counter.utils.io import infer_media_type


def render_keypoint_detection_page(samples: list[Path], output_dir: Path) -> None:
    render_page_header(
        "Keypoints e Detecção de Pose",
        "Detecta pessoas e desenha pontos corporais/esqueleto com modelos YOLO pose.",
        ["YOLO Pose", "Imagem", "Vídeo", "Esqueleto"],
    )

    with st.sidebar:
        uploaded_file, selected_sample, media_url, model_config, render_config = _render_controls(samples)

    input_path = resolve_selected_media(samples, uploaded_file, selected_sample, media_url)
    preview_col, result_col = st.columns([0.88, 1.12], gap="large")
    with preview_col:
        render_input_preview(input_path, uploaded_file, media_url)

    with result_col:
        if st.session_state.pop("pose_run_requested", False) and input_path:
            try:
                result = _run_pose(
                    input_path=input_path,
                    uploaded_file=uploaded_file,
                    media_url=media_url,
                    output_dir=output_dir,
                    model_config=model_config,
                    render_config=render_config,
                )
                st.session_state["last_pose_result"] = result.to_dict()
            except Exception as exc:  # pragma: no cover - shown in Streamlit runtime
                st.error(f"Não foi possível processar os keypoints: {exc}")

        _render_result_panel(st.session_state.get("last_pose_result"))


def _render_controls(samples: list[Path]) -> tuple:
    st.subheader("Entrada")
    uploaded_file = st.file_uploader("Arquivo", type=MEDIA_UPLOAD_TYPES, key="pose_upload")
    selected_sample = st.selectbox("Amostra local", media_sample_options(samples), key="pose_sample")
    media_url = st.text_input(
        "URL",
        value="",
        placeholder="YouTube, Google Drive, Dropbox ou arquivo .mp4/.jpg",
        key="pose_media_url",
    )
    render_input_source_status(uploaded_file, selected_sample, media_url)

    st.subheader("Modelo")
    model_choice = st.selectbox(
        "Peso YOLO",
        [*POSE_MODEL_OPTIONS, "Personalizado"],
        index=POSE_MODEL_OPTIONS.index(DEFAULT_POSE_MODEL_PATH),
        help=yolo_model_help_text("pose", POSE_MODEL_OPTIONS, "-pose.pt"),
    )
    model_path = (
        st.text_input(
            "Peso customizado",
            value=DEFAULT_POSE_MODEL_PATH,
            help="Use um peso YOLO de pose terminado em `-pose.pt`.",
        )
        if model_choice == "Personalizado"
        else model_choice
    )
    model_valid = model_matches_suffix(model_path, "-pose.pt")
    if not model_valid:
        st.warning("Use um peso de pose YOLO terminado em `-pose.pt`.")
    confidence = st.slider(
        "Confiança mínima",
        min_value=0.05,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE,
        key="pose_confidence",
    )
    iou = st.slider("IOU", min_value=0.1, max_value=0.9, value=DEFAULT_IOU, key="pose_iou")

    st.subheader("Keypoints")
    keypoint_confidence = st.slider(
        "Confiança dos pontos",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        key="pose_keypoint_confidence",
    )
    roi_config = render_roi_controls("pose")

    st.subheader("Vídeo")
    frame_stride = st.number_input(
        "Processar a cada N frames",
        min_value=1,
        max_value=30,
        value=1,
        step=1,
        key="pose_frame_stride",
    )
    max_frames = st.number_input(
        "Limite de frames",
        min_value=1,
        max_value=2000,
        value=300,
        step=25,
        key="pose_max_frames",
    )

    input_available = uploaded_file is not None or selected_sample != "Nenhuma" or bool(media_url.strip())
    process_disabled = not input_available or not model_valid or not roi_config["roi_valid"]
    if st.button("Processar keypoints", disabled=process_disabled):
        st.session_state["pose_run_requested"] = True

    return (
        uploaded_file,
        selected_sample,
        media_url,
        {
            "model_path": model_path,
            "confidence": confidence,
            "iou": iou,
            "model_valid": model_valid,
        },
        {
            "keypoint_confidence": keypoint_confidence,
            "frame_stride": int(frame_stride),
            "max_frames": int(max_frames),
            **roi_config,
        },
    )


def _run_pose(
    input_path: Path,
    uploaded_file,
    media_url: str,
    output_dir: Path,
    model_config: dict,
    render_config: dict,
):
    media_type = infer_media_type(input_path)
    roi = build_roi(render_config)

    def processor(actual_input: Path, output_path: Path, summary_path: Path):
        detector = _load_pose_detector(
            model_path=model_config["model_path"],
            confidence=model_config["confidence"],
            iou=model_config["iou"],
        )
        with st.spinner("Processando keypoints..."):
            if media_type == "video":
                csv_path = output_path.with_name(output_path.stem.replace("_processado", "_frames"))
                csv_path = csv_path.with_suffix(".csv")
                progress_callback = create_progress_callback("Keypoints")
                return process_pose_video(
                    input_path=actual_input,
                    output_path=output_path,
                    detector=detector,
                    csv_output=csv_path,
                    summary_output=summary_path,
                    frame_stride=render_config["frame_stride"],
                    max_frames=render_config["max_frames"],
                    keypoint_confidence=render_config["keypoint_confidence"],
                    roi=roi,
                    progress_callback=progress_callback,
                )

            return process_pose_image(
                input_path=actual_input,
                output_path=output_path,
                detector=detector,
                summary_output=summary_path,
                keypoint_confidence=render_config["keypoint_confidence"],
                roi=roi,
            )

    output_suffix = ".mp4" if media_type == "video" else ".jpg"
    return process_uploaded_or_sample(
        input_path,
        uploaded_file,
        media_url,
        output_dir,
        output_suffix,
        processor,
    )


@st.cache_resource(show_spinner=False)
def _load_pose_detector(model_path: str, confidence: float, iou: float) -> YoloPoseDetector:
    return YoloPoseDetector(model_path=model_path, confidence=confidence, iou=iou)


def _render_result_panel(result: dict | None) -> None:
    with st.container(border=True):
        st.subheader("Resultado")
        if not result:
            st.info("Processe uma imagem ou vídeo para ver keypoints e esqueletos.")
            return

        output_path = Path(result["output_path"])
        is_video = str(result.get("model_task", "")).endswith("_video")
        if output_path.exists():
            if is_video:
                render_local_video_result(output_path)
            else:
                st.image(str(output_path), width="stretch")

        col_a, col_b, col_c = st.columns(3)
        average = result.get("average_keypoint_confidence")
        if is_video:
            col_a.metric("Máx. pessoas", result.get("max_people", 0))
            col_b.metric("Frames processados", result.get("frames_processed", 0))
            col_c.metric("FPS proc.", f"{result.get('average_processing_fps', 0):.1f}")
        else:
            col_a.metric("Pessoas", result.get("total", 0))
            col_b.metric("Keypoints visíveis", result.get("visible_keypoints", 0))
            col_c.metric("Confiança média", f"{average:.2f}" if average is not None else "-")

        tab_counts, tab_poses, tab_diagnostic, tab_downloads, tab_json = st.tabs(
            ["Classes", "Detalhes", "Diagnóstico", "Downloads", "JSON"]
        )
        with tab_counts:
            counts = {"person": result.get("max_people", 0)} if is_video else result.get("counts", {})
            render_counts_table(counts)
        with tab_poses:
            if is_video and result.get("csv_output") and Path(result["csv_output"]).exists():
                rows = pd.read_csv(result["csv_output"])
            else:
                rows = pd.DataFrame(
                    [
                        {
                            "pessoa": index + 1,
                            "confiança": item.get("confidence"),
                            "keypoints_visíveis": item.get("visible_keypoints"),
                            "confiança_média": item.get("average_keypoint_confidence"),
                        }
                        for index, item in enumerate(result.get("poses", []))
                    ]
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with tab_diagnostic:
            st.dataframe(
                pd.DataFrame(pose_diagnostic_rows(result)),
                hide_index=True,
                width="stretch",
            )
        with tab_downloads:
            render_media_download("Baixar mídia processada", result["output_path"])
            render_csv_download("Baixar CSV por frame", result.get("csv_output"))
            render_json_download("Baixar resumo JSON", result, Path(result["summary_path"]).name)
        with tab_json:
            st.json(result)
