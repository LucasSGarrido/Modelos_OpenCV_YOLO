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
from object_counter.evaluation import segmentation_diagnostic_rows
from object_counter.config import (
    COCO_CLASSES,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_SEGMENTATION_MODEL_PATH,
    PEOPLE_VEHICLE_CLASSES,
    SEGMENTATION_MODEL_OPTIONS,
)
from object_counter.segmentation import (
    YoloSegmenter,
    process_segmentation_image,
    process_segmentation_video,
)
from object_counter.utils.io import infer_media_type


def render_instance_segmentation_page(samples: list[Path], output_dir: Path) -> None:
    render_page_header(
        "Segmentação de Instâncias",
        "Detecta objetos e desenha máscaras/polígonos para estimar a forma real de cada instância.",
        ["YOLO Seg", "Imagem", "Vídeo", "Áreas"],
    )

    with st.sidebar:
        uploaded_file, selected_sample, media_url, model_config, render_config = _render_controls(samples)

    input_path = resolve_selected_media(samples, uploaded_file, selected_sample, media_url)
    preview_col, result_col = st.columns([0.88, 1.12], gap="large")
    with preview_col:
        render_input_preview(input_path, uploaded_file, media_url)

    with result_col:
        if st.session_state.pop("segmentation_run_requested", False) and input_path:
            try:
                result = _run_segmentation(
                    input_path=input_path,
                    uploaded_file=uploaded_file,
                    media_url=media_url,
                    output_dir=output_dir,
                    model_config=model_config,
                    render_config=render_config,
                )
                st.session_state["last_segmentation_result"] = result.to_dict()
            except Exception as exc:  # pragma: no cover - shown in Streamlit runtime
                st.error(f"Não foi possível processar a segmentação: {exc}")

        _render_result_panel(st.session_state.get("last_segmentation_result"))


def _render_controls(samples: list[Path]) -> tuple:
    st.subheader("Entrada")
    uploaded_file = st.file_uploader(
        "Arquivo",
        type=MEDIA_UPLOAD_TYPES,
        key="segmentation_upload",
    )
    selected_sample = st.selectbox(
        "Amostra local",
        media_sample_options(samples),
        key="segmentation_sample",
    )
    media_url = st.text_input(
        "URL",
        value="",
        placeholder="Link direto, Google Drive, Dropbox ou arquivo .mp4/.jpg",
        help="No deploy, prefira link direto .mp4/.jpg, Google Drive público ou Dropbox. YouTube pode ser bloqueado por HTTP 403 no Streamlit Cloud.",
        key="segmentation_media_url",
    )
    render_input_source_status(uploaded_file, selected_sample, media_url)

    st.subheader("Modelo")
    model_choice = st.selectbox(
        "Peso YOLO",
        [*SEGMENTATION_MODEL_OPTIONS, "Personalizado"],
        index=SEGMENTATION_MODEL_OPTIONS.index(DEFAULT_SEGMENTATION_MODEL_PATH),
        help=yolo_model_help_text("segmentation", SEGMENTATION_MODEL_OPTIONS, "-seg.pt"),
    )
    model_path = (
        st.text_input(
            "Peso customizado",
            value=DEFAULT_SEGMENTATION_MODEL_PATH,
            help="Use um peso YOLO de segmentação terminado em `-seg.pt`.",
        )
        if model_choice == "Personalizado"
        else model_choice
    )
    model_valid = model_matches_suffix(model_path, "-seg.pt")
    if not model_valid:
        st.warning("Use um peso de segmentação YOLO terminado em `-seg.pt`.")

    class_preset_options = ["Pessoas e veículos", "Todas as classes COCO (80)", "Seleção manual"]
    _migrate_session_label(
        "segmentation_class_preset",
        {
            "Pessoas e veiculos": "Pessoas e veículos",
            "Selecao manual": "Seleção manual",
        },
        class_preset_options[0],
        class_preset_options,
    )
    class_preset = st.selectbox(
        "Preset de classes",
        class_preset_options,
        key="segmentation_class_preset",
    )
    if class_preset == "Todas as classes COCO (80)":
        default_classes = COCO_CLASSES
    elif class_preset == "Seleção manual":
        default_classes = []
    else:
        default_classes = PEOPLE_VEHICLE_CLASSES

    selected_classes = st.multiselect(
        "Classes YOLO",
        COCO_CLASSES,
        default=default_classes,
        key="segmentation_classes",
    )
    confidence = st.slider(
        "Confiança mínima",
        min_value=0.05,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE,
        key="segmentation_confidence",
    )
    iou = st.slider(
        "IOU",
        min_value=0.1,
        max_value=0.9,
        value=DEFAULT_IOU,
        key="segmentation_iou",
    )

    st.subheader("Visual")
    show_boxes = st.checkbox("Mostrar bounding boxes", value=True)
    roi_config = render_roi_controls("segmentation")

    st.subheader("Vídeo")
    frame_stride = st.number_input(
        "Processar a cada N frames",
        min_value=1,
        max_value=30,
        value=1,
        step=1,
        key="segmentation_frame_stride",
    )
    max_frames = st.number_input(
        "Limite de frames",
        min_value=1,
        max_value=2000,
        value=300,
        step=25,
        key="segmentation_max_frames",
    )

    input_available = uploaded_file is not None or selected_sample != "Nenhuma" or bool(media_url.strip())
    process_disabled = not input_available or not model_valid or not roi_config["roi_valid"]
    if st.button("Processar segmentação", disabled=process_disabled):
        st.session_state["segmentation_run_requested"] = True

    return (
        uploaded_file,
        selected_sample,
        media_url,
        {
            "model_path": model_path,
            "classes": selected_classes,
            "confidence": confidence,
            "iou": iou,
            "model_valid": model_valid,
        },
        {
            "show_boxes": show_boxes,
            "frame_stride": int(frame_stride),
            "max_frames": int(max_frames),
            **roi_config,
        },
    )


def _migrate_session_label(
    key: str,
    aliases: dict[str, str],
    default: str,
    valid_options: list[str],
) -> None:
    current = st.session_state.get(key)
    if current in aliases:
        st.session_state[key] = aliases[current]
    elif current is not None and current not in valid_options:
        st.session_state[key] = default


def _run_segmentation(
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
        segmenter = _load_segmenter(
            model_path=model_config["model_path"],
            confidence=model_config["confidence"],
            iou=model_config["iou"],
        )
        with st.spinner("Processando segmentação..."):
            if media_type == "video":
                csv_path = output_path.with_name(output_path.stem.replace("_processado", "_frames"))
                csv_path = csv_path.with_suffix(".csv")
                progress_callback = create_progress_callback("Segmentação")
                return process_segmentation_video(
                    input_path=actual_input,
                    output_path=output_path,
                    segmenter=segmenter,
                    classes=model_config["classes"],
                    csv_output=csv_path,
                    summary_output=summary_path,
                    frame_stride=render_config["frame_stride"],
                    max_frames=render_config["max_frames"],
                    show_boxes=render_config["show_boxes"],
                    roi=roi,
                    progress_callback=progress_callback,
                )

            return process_segmentation_image(
                input_path=actual_input,
                output_path=output_path,
                segmenter=segmenter,
                classes=model_config["classes"],
                summary_output=summary_path,
                show_boxes=render_config["show_boxes"],
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
def _load_segmenter(model_path: str, confidence: float, iou: float) -> YoloSegmenter:
    return YoloSegmenter(model_path=model_path, confidence=confidence, iou=iou)


def _render_result_panel(result: dict | None) -> None:
    with st.container(border=True):
        st.subheader("Resultado")
        if not result:
            st.info("Processe uma imagem ou vídeo para ver as máscaras.")
            return

        output_path = Path(result["output_path"])
        is_video = str(result.get("model_task", "")).endswith("_video")
        if output_path.exists():
            if is_video:
                render_local_video_result(output_path)
            else:
                st.image(str(output_path), width="stretch")

        col_a, col_b, col_c = st.columns(3)
        if is_video:
            col_a.metric("Máx. instâncias", result.get("max_frame_total", 0))
            col_b.metric("Área máx.", f"{result.get('max_frame_mask_area', 0):.0f}px")
            col_c.metric("Área máx. %", f"{result.get('max_frame_area_ratio', 0) * 100:.1f}%")
        else:
            area_metrics = result.get("area_metrics", {})
            col_a.metric("Instâncias", result.get("total", 0))
            col_b.metric("Área total", f"{result.get('total_mask_area', 0):.0f}px")
            col_c.metric("Área %", f"{area_metrics.get('mask_area_ratio', 0) * 100:.1f}%")

        if is_video:
            perf_a, perf_b = st.columns(2)
            perf_a.metric("Frames processados", result.get("frames_processed", 0))
            perf_b.metric("FPS proc.", f"{result.get('average_processing_fps', 0):.1f}")
        else:
            st.caption(f"Tempo de inferência: {result.get('inference_seconds', 0):.2f}s")

        tab_counts, tab_segments, tab_metrics, tab_diagnostic, tab_downloads, tab_json = st.tabs(
            ["Classes", "Detalhes", "Áreas", "Diagnóstico", "Downloads", "JSON"]
        )
        with tab_counts:
            counts = result.get("max_counts_by_class", {}) if is_video else result.get("counts", {})
            render_counts_table(counts)
        with tab_segments:
            if is_video and result.get("csv_output") and Path(result["csv_output"]).exists():
                rows = pd.read_csv(result["csv_output"])
            else:
                rows = pd.DataFrame(
                    [
                        {
                            "classe": item.get("label"),
                            "confiança": item.get("confidence"),
                            "área_máscara": item.get("mask_area"),
                            "área_bbox": item.get("bbox_area"),
                            "pontos": len(item.get("polygon", [])),
                        }
                        for item in result.get("segments", [])
                    ]
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with tab_metrics:
            _render_area_metrics(result, is_video)
        with tab_diagnostic:
            st.dataframe(
                pd.DataFrame(segmentation_diagnostic_rows(result)),
                hide_index=True,
                width="stretch",
            )
        with tab_downloads:
            render_media_download("Baixar mídia processada", result["output_path"])
            render_csv_download("Baixar CSV por frame", result.get("csv_output"))
            render_json_download("Baixar resumo JSON", result, Path(result["summary_path"]).name)
        with tab_json:
            st.json(result)


def _render_area_metrics(result: dict, is_video: bool) -> None:
    if is_video:
        rows = [
            {"métrica": "maior total de área por frame", "valor": result.get("max_frame_mask_area", 0)},
            {"métrica": "maior percentual da imagem", "valor": result.get("max_frame_area_ratio", 0)},
            {"métrica": "maior máscara", "valor": result.get("largest_mask_area", 0)},
        ]
        area_by_class = result.get("max_area_by_class", {})
    else:
        area_metrics = result.get("area_metrics", {})
        rows = [
            {"métrica": "área da imagem", "valor": area_metrics.get("image_area", 0)},
            {"métrica": "área total das máscaras", "valor": area_metrics.get("total_mask_area", 0)},
            {"métrica": "percentual mascarado", "valor": area_metrics.get("mask_area_ratio", 0)},
            {"métrica": "área média por máscara", "valor": area_metrics.get("average_mask_area", 0)},
            {"métrica": "maior máscara", "valor": area_metrics.get("largest_mask_area", 0)},
            {"métrica": "classe da maior máscara", "valor": area_metrics.get("largest_mask_class", "-")},
        ]
        area_by_class = area_metrics.get("area_by_class", {})

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if area_by_class:
        class_rows = [
            {"classe": label, "área_px": area} for label, area in sorted(area_by_class.items())
        ]
        st.dataframe(pd.DataFrame(class_rows), hide_index=True, width="stretch")
