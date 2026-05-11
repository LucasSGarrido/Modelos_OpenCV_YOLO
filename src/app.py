from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st
from PIL import Image

from app_pages.custom_training import render_custom_training_page
from app_pages.instance_segmentation import render_instance_segmentation_page
from app_pages.keypoint_detection import render_keypoint_detection_page
from app_pages.shared import render_input_source_status, render_local_video_result
from object_counter.config import (
    COCO_CLASSES,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_MODEL_PATH,
    PEOPLE_VEHICLE_CLASSES,
)
from object_counter.counting.image_counter import ImageCounterResult, process_image
from object_counter.counting.roi import RegionOfInterest
from object_counter.counting.video_counter import VideoCounterResult, process_video
from object_counter.detection.detector import YoloDetector
from object_counter.evaluation.count_report import error_analysis_rows
from object_counter.utils.downloads import (
    download_media_url,
    media_filename_from_url,
    streamlit_preview_url,
)
from object_counter.utils.history import (
    append_history_csv,
    build_history_record,
    filter_history,
    load_history_csv,
)
from object_counter.utils.io import infer_media_type
from object_counter.utils.reports import (
    build_comparison_html,
    build_comparison_markdown,
    build_run_html,
    build_run_markdown,
    compare_history_records,
    history_record_label,
)


st.set_page_config(page_title="Contador de Objetos", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
APP_OUTPUT_DIR = PROJECT_ROOT / "reports" / "app"
EVALUATION_DIR = PROJECT_ROOT / "reports" / "evaluation"
HISTORY_PATH = APP_OUTPUT_DIR / "history.csv"
SUPPORTED_MEDIA = {".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"}
HELP_TEXTS = {
    "url": "URL de mídia. Aceita link direto, YouTube, Google Drive público ou Dropbox. Use apenas vídeos que você tem direito de processar.",
    "arquivo": "Envie uma imagem ou vídeo para processar. Formatos aceitos: JPG, PNG, MP4, AVI e MOV.",
    "amostra": "Usa um arquivo local em data/samples/ para testar sem precisar fazer upload.",
    "modelo": "Arquivo .pt com os pesos do YOLO. yolov8n.pt é rápido; modelos maiores tendem a ser mais precisos e mais lentos.",
    "preset_classes": "Define o conjunto inicial de classes. Pessoas e veículos é o domínio principal deste projeto.",
    "classes_yolo": "Classes que o YOLO pré-treinado em COCO deve manter na contagem.",
    "classes_extras": "Use apenas com modelos customizados que tenham classes fora das 80 classes COCO.",
    "confidence": "Confiança mínima da detecção. Valores maiores reduzem falsos positivos, mas podem perder objetos difíceis.",
    "iou": "IOU controla a supressão de caixas sobrepostas. Valores menores removem sobreposições com mais força; valores maiores preservam mais caixas.",
    "counting_mode": "frame conta objetos visíveis no frame atual. line conta eventos quando um objeto rastreado cruza a linha.",
    "line_orientation": "Define se a linha de contagem será horizontal ou vertical.",
    "line_position": "Posição relativa da linha entre 0 e 1. Ex.: 0.50 coloca a linha no centro.",
    "line_direction": "Direção aceita no cruzamento: ambas, positiva ou negativa, conforme o movimento do objeto pela linha.",
    "tracking": "Método para manter IDs em vídeos. centroid é simples; ByteTrack tende a ser mais robusto em cenas difíceis.",
    "max_frames": "Limita quantos frames do vídeo serão processados. Útil para testes rápidos e vídeos longos.",
    "roi": "ROI é a Região de Interesse. Objetos fora dessa área são ignorados na contagem. Em vídeos, o desenho usa o primeiro frame como referência.",
    "roi_x_min": "Borda esquerda da ROI em coordenada relativa. 0.00 é o início da imagem.",
    "roi_x_max": "Borda direita da ROI em coordenada relativa. 1.00 é o fim da imagem.",
    "roi_y_min": "Borda superior da ROI em coordenada relativa. 0.00 é o topo da imagem.",
    "roi_y_max": "Borda inferior da ROI em coordenada relativa. 1.00 é a base da imagem.",
}


def main() -> None:
    inject_css()

    samples = get_sample_files()
    with st.sidebar:
        st.subheader("Navegação")
        page_options = [
            "Detecção e Contagem",
            "Segmentação de Instâncias",
            "Keypoints / Pose",
            "Treinamento Customizado",
        ]
        _migrate_session_label(
            "app_page",
            {
                "Deteccao e Contagem": "Detecção e Contagem",
                "Segmentacao de Instancias": "Segmentação de Instâncias",
            },
            page_options[0],
            page_options,
        )
        page = st.radio(
            "Página",
            page_options,
            label_visibility="collapsed",
            key="app_page",
        )
        st.divider()

    if page == "Segmentação de Instâncias":
        render_instance_segmentation_page(
            samples=samples,
            output_dir=PROJECT_ROOT / "reports" / "segmentation",
        )
        return

    if page == "Keypoints / Pose":
        render_keypoint_detection_page(
            samples=samples,
            output_dir=PROJECT_ROOT / "reports" / "pose",
        )
        return

    if page == "Treinamento Customizado":
        render_custom_training_page()
        return

    render_header()

    with st.sidebar:
        uploaded_file, selected_sample, media_url, model_config, video_config = render_controls(samples)

    input_path = resolve_input_path(uploaded_file, selected_sample, media_url)
    media_type = infer_media_type(input_path) if input_path else None

    preview_col, result_col = st.columns([0.88, 1.12], gap="large")
    with preview_col:
        render_input_panel(input_path, media_type, uploaded_file, media_url)

    with result_col:
        run_requested = st.session_state.pop("run_requested", False)
        if run_requested and input_path:
            result = run_pipeline(
                input_path=input_path,
                uploaded_file=uploaded_file,
                media_url=media_url,
                media_type=media_type,
                model_config=model_config,
                video_config=video_config,
            )
            st.session_state["last_result"] = result.to_dict()
            st.session_state["last_media_type"] = media_type
            st.session_state["last_output_path"] = result.output_path
            append_history(result.to_dict(), media_type, model_config, video_config)

        render_result_panel()

    render_history_panel()
    render_comparison_panel()
    render_evaluation_panel()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --app-bg: #ffffff;
          --surface: #ffffff;
          --surface-muted: #f4f4f5;
          --ink: #050505;
          --muted: #4b5563;
          --line: #d4d4d8;
          --control-bg: #050505;
          --control-bg-hover: #262626;
          --control-text: #ffffff;
          --control-muted: #d4d4d8;
          --sidebar-bg: #111827;
          --sidebar-border: #374151;
          --sidebar-text: #f9fafb;
          --sidebar-muted: #d1d5db;
          --sidebar-control-bg: #050505;
          --sidebar-control-hover: #1f2937;
          --warning: #525252;
        }

        .stApp {
          background: var(--app-bg);
          color: var(--ink);
        }

        .stApp * {
          letter-spacing: 0;
        }

        [data-testid="stSidebar"] {
          background: var(--sidebar-bg);
          border-right: 1px solid var(--sidebar-border);
        }

        [data-testid="stSidebar"] * {
          color: var(--sidebar-text);
        }

        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div {
          color: var(--sidebar-text);
        }

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
          background: var(--sidebar-control-bg);
          border-color: var(--sidebar-border);
          color: var(--control-text);
        }

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] div {
          color: var(--control-text);
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCaptionContainer,
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] p {
          color: var(--sidebar-muted);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
          color: var(--sidebar-text);
        }

        [data-testid="stSidebar"] div.stButton > button {
          border-color: var(--sidebar-border);
          background: var(--sidebar-control-bg);
          color: var(--sidebar-text);
        }

        [data-testid="stSidebar"] div.stButton > button:hover {
          border-color: var(--sidebar-muted);
          background: var(--sidebar-control-hover);
          color: var(--sidebar-text);
        }

        [data-testid="stSidebar"] div.stButton > button:disabled {
          border-color: var(--sidebar-border);
          background: var(--sidebar-control-bg);
          color: #6b7280;
        }

        [data-testid="stSidebar"] [data-testid="stNumberInput"] button,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] [role="button"] {
          background: var(--sidebar-control-bg);
          border-color: var(--sidebar-border);
          color: var(--sidebar-text);
        }

        [data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] [role="button"]:hover {
          background: var(--sidebar-control-hover);
          border-color: var(--sidebar-muted);
          color: var(--sidebar-text);
        }

        [data-testid="stSidebar"] [data-testid="stNumberInput"] svg {
          color: var(--sidebar-text);
          fill: var(--sidebar-text);
        }

        [data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] {
          background: var(--sidebar-control-bg);
          border-color: var(--sidebar-border);
        }

        [data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] button {
          background: var(--sidebar-text);
          border-color: var(--sidebar-text);
          color: var(--ink);
        }

        .main .block-container {
          max-width: 1440px;
          padding-top: 2rem;
          padding-bottom: 3rem;
        }

        h1, h2, h3 {
          color: var(--ink);
          letter-spacing: 0;
        }

        .app-title {
          display: flex;
          justify-content: space-between;
          align-items: flex-end;
          gap: 1rem;
          margin-bottom: 1.2rem;
          border-bottom: 1px solid var(--line);
          padding-bottom: 1.1rem;
        }

        .app-title h1 {
          margin: 0;
          font-size: 2rem;
          line-height: 1.1;
          font-weight: 750;
        }

        .app-title p {
          margin: 0.35rem 0 0;
          color: var(--muted);
          font-size: 0.98rem;
        }

        .status-line {
          display: flex;
          flex-wrap: wrap;
          justify-content: flex-end;
          gap: 0.5rem;
          min-width: 260px;
        }

        .status-pill {
          border: 1px solid var(--control-bg);
          background: var(--surface);
          border-radius: 999px;
          padding: 0.38rem 0.7rem;
          color: var(--ink);
          font-size: 0.8rem;
          font-weight: 650;
          white-space: nowrap;
        }

        .panel {
          background: var(--surface);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 1rem;
          box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 1rem;
          margin-bottom: 0.8rem;
        }

        .panel-header h2 {
          margin: 0;
          font-size: 1rem;
          font-weight: 750;
        }

        .panel-header span {
          color: var(--muted);
          font-size: 0.82rem;
        }

        .metric-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 0.7rem;
          margin: 0 0 1rem;
        }

        .metric-card {
          background: var(--surface);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 0.85rem;
          min-height: 82px;
        }

        .metric-label {
          color: var(--muted);
          font-size: 0.78rem;
          font-weight: 700;
          text-transform: uppercase;
        }

        .metric-value {
          color: var(--ink);
          font-size: 1.55rem;
          font-weight: 800;
          margin-top: 0.25rem;
        }

        .empty-state {
          border: 1px dashed var(--ink);
          background: var(--surface);
          border-radius: 8px;
          padding: 2.4rem 1rem;
          text-align: center;
          color: var(--ink);
        }

        .artifact-row {
          display: grid;
          grid-template-columns: 130px 1fr;
          gap: 0.65rem;
          padding: 0.55rem 0;
          border-bottom: 1px solid var(--line);
          font-size: 0.9rem;
        }

        .artifact-row:last-child {
          border-bottom: 0;
        }

        .artifact-label {
          color: var(--muted);
          font-weight: 700;
        }

        .artifact-value {
          color: var(--ink);
          overflow-wrap: anywhere;
        }

        .history-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 0.7rem;
          margin-top: 1.2rem;
        }

        .history-card {
          background: var(--surface);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 0.85rem;
        }

        .history-title {
          color: var(--ink);
          font-weight: 750;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .history-meta {
          color: var(--muted);
          font-size: 0.8rem;
          margin-top: 0.25rem;
        }

        div.stButton > button {
          width: 100%;
          border-radius: 6px;
          border: 1px solid var(--control-bg);
          background: var(--control-bg);
          color: var(--control-text);
          font-weight: 750;
          min-height: 2.7rem;
        }

        div.stButton > button:hover {
          border-color: var(--control-bg-hover);
          background: var(--control-bg-hover);
          color: var(--control-text);
        }

        div.stButton > button:disabled {
          background: var(--surface-muted);
          border-color: var(--line);
          color: var(--muted);
        }

        div.stDownloadButton > button {
          border: 1px solid var(--control-bg);
          background: var(--control-bg);
          color: var(--control-text);
          border-radius: 6px;
          font-weight: 750;
          min-height: 2.5rem;
        }

        div.stDownloadButton > button:hover {
          border-color: var(--control-bg-hover);
          background: var(--control-bg-hover);
          color: var(--control-text);
        }

        div[data-testid="stAlert"] {
          background: var(--surface);
          border: 1px solid var(--ink);
          color: var(--ink);
          border-radius: 8px;
        }

        div[data-testid="stAlert"] * {
          color: var(--ink);
        }

        div[data-testid="stMetric"] {
          background: var(--surface);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 0.75rem;
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div {
          color: var(--ink);
        }

        div[data-testid="stDataFrame"] {
          border: 1px solid var(--ink);
          border-radius: 8px;
          overflow: hidden;
        }

        div[data-testid="stDataFrame"] div,
        div[data-testid="stDataFrame"] span {
          color: var(--ink);
        }

        section[data-testid="stFileUploaderDropzone"] {
          background: var(--surface);
          border: 1px dashed var(--ink);
          border-radius: 8px;
        }

        section[data-testid="stFileUploaderDropzone"] * {
          color: var(--ink);
        }

        section[data-testid="stFileUploaderDropzone"] button {
          background: var(--control-bg);
          border: 1px solid var(--control-bg);
          color: var(--control-text);
          border-radius: 6px;
        }

        [data-baseweb="tag"] {
          background: var(--control-bg);
          color: var(--control-text);
          border-radius: 6px;
        }

        [data-baseweb="tag"] * {
          color: var(--control-text);
        }

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div {
          background: var(--control-bg);
          border-color: var(--control-bg);
          color: var(--control-text);
        }

        [data-baseweb="select"] span,
        [data-baseweb="select"] div,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {
          color: var(--control-text);
        }

        [data-baseweb="popover"] {
          background: var(--control-bg);
          color: var(--control-text);
        }

        [data-baseweb="popover"] * {
          color: var(--control-text);
        }

        [role="listbox"] {
          background: var(--control-bg);
        }

        [role="option"] {
          background: var(--control-bg);
          color: var(--control-text);
        }

        [role="option"]:hover {
          background: var(--control-bg-hover);
        }

        .stTabs [data-baseweb="tab-list"] {
          border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
          color: var(--ink);
        }

        .stTabs [aria-selected="true"] {
          color: var(--ink);
          border-bottom-color: var(--ink);
        }

        .stSlider [data-baseweb="slider"] div {
          color: var(--ink);
        }

        @media (max-width: 980px) {
          .app-title {
            align-items: flex-start;
            flex-direction: column;
          }
          .status-line {
            justify-content: flex-start;
          }
          .metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
          .history-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="app-title">
          <div>
            <h1>Contador Inteligente de Objetos</h1>
            <p>Contagem de pessoas e veículos com OpenCV + YOLO, ROI, tracking e linha de passagem.</p>
          </div>
          <div class="status-line">
            <span class="status-pill">Pessoas e veículos</span>
            <span class="status-pill">Modo frame</span>
            <span class="status-pill">Modo line</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_controls(samples: list[Path]) -> tuple:
    st.subheader("Entrada")
    uploaded_file = st.file_uploader(
        "Arquivo",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        help=HELP_TEXTS["arquivo"],
    )
    sample_options = ["Nenhuma"] + [path.name for path in samples]
    selected_sample = st.selectbox("Amostra local", sample_options, help=HELP_TEXTS["amostra"])
    media_url = st.text_input(
        "URL",
        value="",
        placeholder="YouTube, Google Drive, Dropbox ou arquivo .mp4/.jpg",
        help=HELP_TEXTS["url"],
    )
    render_input_source_status(uploaded_file, selected_sample, media_url)

    st.subheader("Modelo")
    model_path = st.text_input("Peso YOLO", value=DEFAULT_MODEL_PATH, help=HELP_TEXTS["modelo"])
    class_preset = st.selectbox(
        "Preset de classes",
        ["Pessoas e veículos", "Todas as classes COCO (80)", "Seleção manual"],
        help=HELP_TEXTS["preset_classes"],
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
        help=HELP_TEXTS["classes_yolo"],
    )
    custom_classes_text = st.text_input(
        "Classes extras",
        value="",
        help=HELP_TEXTS["classes_extras"],
    )
    confidence = st.slider(
        "Confiança mínima",
        min_value=0.05,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE,
        help=HELP_TEXTS["confidence"],
    )
    iou = st.slider("IOU", min_value=0.1, max_value=0.9, value=DEFAULT_IOU, help=HELP_TEXTS["iou"])

    st.subheader("Vídeo")
    counting_mode = st.radio(
        "Contagem",
        ["frame", "line"],
        horizontal=True,
        help=HELP_TEXTS["counting_mode"],
    )
    line_orientation = st.selectbox(
        "Linha",
        ["horizontal", "vertical"],
        help=HELP_TEXTS["line_orientation"],
    )
    line_position = st.slider(
        "Posição",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        help=HELP_TEXTS["line_position"],
    )
    line_direction = st.selectbox(
        "Direção",
        ["both", "positive", "negative"],
        help=HELP_TEXTS["line_direction"],
    )
    tracking_backend = st.selectbox(
        "Tracking",
        ["centroid", "bytetrack"],
        help=HELP_TEXTS["tracking"],
    )
    max_frames = st.number_input(
        "Limite de frames",
        min_value=1,
        max_value=2000,
        value=300,
        step=25,
        help=HELP_TEXTS["max_frames"],
    )

    st.subheader("Região de Interesse")
    roi_mode = st.selectbox(
        "ROI",
        ["Desativada", "Sliders", "Desenhar na imagem"],
        help=HELP_TEXTS["roi"],
    )
    st.session_state["roi_mode"] = roi_mode
    roi_enabled = roi_mode != "Desativada"
    if roi_mode == "Desenhar na imagem":
        default_roi = st.session_state.get("drawn_roi", [0.0, 0.0, 1.0, 1.0])
    else:
        default_roi = [0.0, 0.0, 1.0, 1.0]
    roi_x_min = st.slider(
        "ROI x mín.",
        min_value=0.0,
        max_value=0.9,
        value=float(default_roi[0]),
        disabled=roi_mode != "Sliders",
        help=HELP_TEXTS["roi_x_min"],
    )
    roi_x_max = st.slider(
        "ROI x máx.",
        min_value=0.1,
        max_value=1.0,
        value=float(default_roi[2]),
        disabled=roi_mode != "Sliders",
        help=HELP_TEXTS["roi_x_max"],
    )
    roi_y_min = st.slider(
        "ROI y mín.",
        min_value=0.0,
        max_value=0.9,
        value=float(default_roi[1]),
        disabled=roi_mode != "Sliders",
        help=HELP_TEXTS["roi_y_min"],
    )
    roi_y_max = st.slider(
        "ROI y máx.",
        min_value=0.1,
        max_value=1.0,
        value=float(default_roi[3]),
        disabled=roi_mode != "Sliders",
        help=HELP_TEXTS["roi_y_max"],
    )
    if roi_mode == "Desenhar na imagem":
        roi_x_min, roi_y_min, roi_x_max, roi_y_max = default_roi
    roi_valid = not roi_enabled or (roi_x_min < roi_x_max and roi_y_min < roi_y_max)
    if not roi_valid:
        st.warning("A ROI precisa ter mínimos menores que máximos.")

    if media_url.strip() and roi_mode == "Desenhar na imagem":
        st.warning("Para entrada por URL, use ROI por sliders.")
        roi_valid = False

    input_available = uploaded_file is not None or selected_sample != "Nenhuma" or bool(media_url.strip())
    process_clicked = st.button("Processar", disabled=not input_available or not roi_valid)
    if process_clicked:
        st.session_state["run_requested"] = True

    custom_classes = [item.strip() for item in custom_classes_text.split(",") if item.strip()]
    classes = deduplicate_classes([*selected_classes, *custom_classes])
    model_config = {
        "model_path": model_path,
        "classes": classes,
        "confidence": confidence,
        "iou": iou,
    }
    video_config = {
        "counting_mode": counting_mode,
        "line_orientation": line_orientation,
        "line_position": line_position,
        "line_direction": line_direction,
        "tracking_backend": tracking_backend,
        "max_frames": int(max_frames),
        "roi_enabled": roi_enabled,
        "roi": [roi_x_min, roi_y_min, roi_x_max, roi_y_max],
    }
    return uploaded_file, selected_sample, media_url, model_config, video_config


def render_input_panel(
    input_path: Path | None,
    media_type: str | None,
    uploaded_file,
    media_url: str = "",
) -> None:
    with st.container(border=True):
        render_section_header("Entrada", "amostra ou upload")

        if input_path is None or media_type is None:
            st.info("Nenhum arquivo selecionado.")
        elif media_url.strip() and uploaded_file is None and media_type == "image":
            preview_url = streamlit_preview_url(media_url)
            if preview_url is None:
                st.info("Prévia remota indisponível antes do download.")
            else:
                st.image(preview_url, width="stretch")
        elif media_url.strip() and uploaded_file is None:
            preview_url = streamlit_preview_url(media_url)
            if preview_url is None:
                st.info("Prévia remota indisponível antes do download.")
            else:
                st.video(preview_url)
        elif uploaded_file is not None and media_type == "image":
            if st.session_state.get("roi_mode") == "Desenhar na imagem":
                render_roi_drawer(input_path, uploaded_file, media_type)
            else:
                st.image(uploaded_file.getvalue(), width="stretch")
        elif uploaded_file is not None:
            if st.session_state.get("roi_mode") == "Desenhar na imagem":
                render_roi_drawer(input_path, uploaded_file, media_type)
            else:
                st.video(uploaded_file.getvalue())
        elif media_type == "image":
            if st.session_state.get("roi_mode") == "Desenhar na imagem":
                render_roi_drawer(input_path, uploaded_file, media_type)
            else:
                st.image(str(input_path), width="stretch")
        else:
            if st.session_state.get("roi_mode") == "Desenhar na imagem":
                render_roi_drawer(input_path, uploaded_file, media_type)
            else:
                st.video(str(input_path))


def render_result_panel() -> None:
    result = st.session_state.get("last_result")
    media_type = st.session_state.get("last_media_type")
    output_path = st.session_state.get("last_output_path")

    with st.container(border=True):
        render_section_header("Resultado", "overlay e métricas")

        if not result or not output_path:
            st.info("Aguardando processamento.")
            return

        render_metrics(result, media_type)

        if media_type == "image":
            st.image(output_path, width="stretch")
        else:
            render_local_video_result(output_path)

        tab_classes, tab_detections, tab_events, tab_diagnostics, tab_report, tab_downloads, tab_artifacts, tab_json = st.tabs(
            ["Classes", "Detecções", "Eventos", "Diagnóstico", "Relatório", "Downloads", "Artefatos", "JSON"]
        )
        with tab_classes:
            render_class_table(result, media_type)
        with tab_detections:
            render_detection_table(result, media_type)
        with tab_events:
            render_event_table(result)
        with tab_diagnostics:
            render_diagnostics(result, media_type)
        with tab_report:
            render_run_report(result, media_type)
        with tab_downloads:
            render_downloads(result, media_type)
        with tab_artifacts:
            render_artifacts(result)
        with tab_json:
            st.json(result)


def render_metrics(result: dict, media_type: str | None) -> None:
    if media_type == "image":
        metrics = [
            ("Total", result.get("total", 0)),
            ("Classes", len(result.get("counts", {}))),
            ("Detecções", len(result.get("detections", []))),
            ("Inferência", f"{float(result.get('inference_seconds', 0)):.3f}s"),
        ]
    else:
        line_total = result.get("total_line_crossings", 0)
        frame_total = sum(result.get("last_frame_counts", {}).values())
        total = line_total if result.get("counting_mode") == "line" else frame_total
        metrics = [
            ("Total", total),
            ("Frames", result.get("frames_processed", 0)),
            ("FPS", f"{float(result.get('average_processing_fps', 0)):.1f}"),
            ("Modo", result.get("counting_mode", "-")),
        ]

    render_metric_row(metrics)


def render_class_table(result: dict, media_type: str | None) -> None:
    if media_type == "image":
        counts = result.get("counts", {})
        title = "Contagem da imagem"
    elif result.get("counting_mode") == "line":
        counts = result.get("line_counts_by_class", {})
        title = "Eventos por classe"
    else:
        counts = result.get("last_frame_counts", {})
        title = "Contagem do último frame"

    if not counts:
        st.info("Nenhuma classe encontrada com os filtros atuais.")
        return

    table = pd.DataFrame(
        [{"classe": label, "contagem": count} for label, count in sorted(counts.items())]
    )
    st.caption(title)
    render_dataframe(table)


def render_event_table(result: dict) -> None:
    csv_path = result.get("csv_output")
    if not csv_path:
        st.info("Eventos aparecem aqui quando um vídeo é processado no modo line.")
        return

    events = load_line_events(Path(csv_path))
    if events.empty:
        st.info("Nenhum cruzamento de linha foi registrado nessa execução.")
        return

    render_dataframe(events)


def render_detection_table(result: dict, media_type: str | None) -> None:
    if media_type != "image":
        st.info("A tabela de detecções detalhadas está disponível para imagens.")
        return

    detections = result.get("detections", [])
    if not detections:
        st.info("Nenhuma detecção detalhada disponível.")
        return

    table = pd.DataFrame(detections)
    if not table.empty:
        table = table.rename(
            columns={
                "label": "classe",
                "confidence": "confiança",
                "class_id": "classe_id",
            }
        )
        if "confiança" in table.columns:
            table["confiança"] = table["confiança"].round(4)
    render_dataframe(table)


def render_diagnostics(result: dict, media_type: str | None) -> None:
    rows = [
        {"item": "tipo", "valor": media_type or "-"},
        {"item": "modo", "valor": result.get("counting_mode", "image")},
        {"item": "saída", "valor": result.get("output_path", "-")},
        {"item": "resumo", "valor": result.get("summary_path", "-")},
    ]

    if media_type == "image":
        rows.append({"item": "tempo de inferência", "valor": result.get("inference_seconds", 0)})
        rows.append({"item": "roi", "valor": json.dumps(result.get("roi_config"), ensure_ascii=False)})
    else:
        rows.extend(
            [
                {"item": "frames processados", "valor": result.get("frames_processed", 0)},
                {"item": "fps médio", "valor": result.get("average_processing_fps", 0)},
                {"item": "linha", "valor": json.dumps(result.get("line_config"), ensure_ascii=False)},
                {"item": "roi", "valor": json.dumps(result.get("roi_config"), ensure_ascii=False)},
            ]
        )

    render_dataframe(pd.DataFrame(rows))


def render_downloads(result: dict, media_type: str | None) -> None:
    files = [
        ("Resultado", result.get("output_path")),
        ("Resumo JSON", result.get("summary_path")),
        ("CSV por frame", result.get("csv_output")),
    ]
    rendered = False

    for label, raw_path in files:
        if not raw_path:
            continue

        path = Path(raw_path)
        if not path.exists():
            continue

        st.download_button(
            label=f"Baixar {label}",
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime_for_path(path, media_type),
            key=f"download_{label}_{path.name}",
        )
        rendered = True

    if not rendered:
        st.info("Nenhum arquivo persistido encontrado para download.")


def render_run_report(result: dict, media_type: str | None) -> None:
    report = build_run_markdown(result, media_type)
    html_report = build_run_html(result, media_type)
    st.markdown(report)
    st.download_button(
        "Baixar relatório Markdown",
        data=report.encode("utf-8"),
        file_name="relatorio_execucao.md",
        mime="text/markdown",
    )
    st.download_button(
        "Baixar relatório HTML",
        data=html_report.encode("utf-8"),
        file_name="relatorio_execucao.html",
        mime="text/html",
    )


def render_artifacts(result: dict) -> None:
    rows = {
        "Entrada": result.get("input_path"),
        "Saída": result.get("output_path"),
        "Resumo": result.get("summary_path"),
        "CSV": result.get("csv_output"),
    }
    table = [{"artefato": label, "caminho": value} for label, value in rows.items() if value]
    if not table:
        st.info("Nenhum artefato registrado.")
        return

    render_dataframe(pd.DataFrame(table))


def render_history_panel() -> None:
    history = load_history_csv(HISTORY_PATH)
    if history.empty:
        return

    with st.container(border=True):
        render_section_header("Histórico", "execuções persistidas")
        filter_col1, filter_col2, filter_col3 = st.columns([0.24, 0.24, 0.52])
        media_filter = filter_col1.selectbox(
            "Tipo",
            ["Todos"] + sorted(history["media_type"].dropna().astype(str).unique().tolist()),
            key="history_media_filter",
        )
        mode_filter = filter_col2.selectbox(
            "Modo",
            ["Todos"] + sorted(history["mode"].dropna().astype(str).unique().tolist()),
            key="history_mode_filter",
        )
        search = filter_col3.text_input("Buscar", key="history_search")
        filtered = filter_history(history, media_type=media_filter, mode=mode_filter, search=search)
        render_dataframe(filtered, height=360)

        st.download_button(
            "Baixar histórico",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="history.csv",
            mime="text/csv",
        )


def render_comparison_panel() -> None:
    history = load_history_csv(HISTORY_PATH)
    if len(history) < 2:
        return

    records = history.to_dict(orient="records")
    options = {history_record_label(record, index): record for index, record in enumerate(records, start=1)}
    labels = list(options)

    with st.container(border=True):
        render_section_header("Comparação", "duas execuções")
        left_col, right_col = st.columns(2)
        left_label = left_col.selectbox("Execução A", labels, key="compare_left")
        right_label = right_col.selectbox(
            "Execução B",
            labels,
            index=1 if len(labels) > 1 else 0,
            key="compare_right",
        )

        left_record = options[left_label]
        right_record = options[right_label]
        comparison = compare_history_records(left_record, right_record)
        overview = pd.DataFrame(comparison["overview"])
        classes = pd.DataFrame(comparison["classes"])

        render_dataframe(overview)
        if classes.empty:
            st.info("Sem contagens por classe disponíveis para comparar.")
        else:
            render_dataframe(classes)

        report = build_comparison_markdown(left_record, right_record, comparison)
        html_report = build_comparison_html(left_record, right_record, comparison)
        st.download_button(
            "Baixar comparação Markdown",
            data=report.encode("utf-8"),
            file_name="comparacao_execucoes.md",
            mime="text/markdown",
        )
        st.download_button(
            "Baixar comparação HTML",
            data=html_report.encode("utf-8"),
            file_name="comparacao_execucoes.html",
            mime="text/html",
        )


def render_evaluation_panel() -> None:
    summary_path = EVALUATION_DIR / "counts_summary.json"
    report_path = EVALUATION_DIR / "counts_report.csv"
    if not summary_path.exists() or not report_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = pd.read_csv(report_path)
    with st.container(border=True):
        render_section_header("Avaliação Manual", "real vs previsto")
        tab_summary, tab_errors, tab_specialized, tab_thresholds, tab_models = st.tabs(
            ["Resumo", "Análise de erros", "Segmentação/Pose", "Thresholds", "Modelos"]
        )
        with tab_summary:
            render_metric_row(
                [
                    ("Amostras", summary.get("sample_count", 0)),
                    ("Erro total", summary.get("total_absolute_error", 0)),
                    ("MAE", summary.get("mean_absolute_count_error", 0)),
                    ("Acerto exato", f"{float(summary.get('exact_count_match_rate', 0)):.0%}"),
                ]
            )
            render_dataframe(report, height=260)
        with tab_errors:
            render_error_analysis(report)
        with tab_specialized:
            render_specialized_evaluation_panel()
        with tab_thresholds:
            render_threshold_panel()
        with tab_models:
            render_model_comparison_panel()


def render_specialized_evaluation_panel() -> None:
    summary_path = EVALUATION_DIR / "segmentation_pose_summary.json"
    report_path = EVALUATION_DIR / "segmentation_pose_report.csv"
    error_path = EVALUATION_DIR / "segmentation_pose_error_analysis.csv"
    if not summary_path.exists() or not report_path.exists():
        st.info("Rode `python scripts/evaluate_specialized_tasks.py` para gerar este painel.")
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = pd.read_csv(report_path)
    render_metric_row(
        [
            ("Amostras", summary.get("sample_count", 0)),
            ("Métricas", summary.get("metric_rows", 0)),
            ("MAE", summary.get("mean_absolute_error", 0)),
            ("Dentro da tolerância", f"{float(summary.get('within_tolerance_rate', 0)):.0%}"),
        ]
    )
    render_dataframe(report, height=260)
    if error_path.exists():
        st.caption("Análise comentada por métrica")
        render_dataframe(pd.read_csv(error_path), height=260)


def render_metric_row(metrics: list[tuple[str, object]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics, strict=False):
        column.metric(label, value)


def render_section_header(title: str, subtitle: str) -> None:
    left, right = st.columns([0.65, 0.35])
    left.markdown(f"**{title}**")
    right.caption(subtitle)


def render_dataframe(
    table: pd.DataFrame,
    *,
    height: int | None = None,
    column_config: dict | None = None,
) -> None:
    table = normalize_dataframe_for_streamlit(table)
    options = {
        "width": "stretch",
        "hide_index": True,
        "row_height": 32,
        "column_config": column_config,
    }
    if height is not None:
        options["height"] = height

    st.dataframe(table, **options)


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


def normalize_dataframe_for_streamlit(table: pd.DataFrame) -> pd.DataFrame:
    normalized = table.copy()
    for column in normalized.select_dtypes(include=["object"]).columns:
        normalized[column] = normalized[column].map(normalize_streamlit_cell)
    return normalized


def normalize_streamlit_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    try:
        if pd.isna(value):
            return ""
    except ValueError:
        pass
    return str(value)


def run_pipeline(
    input_path: Path,
    uploaded_file,
    media_url: str,
    media_type: str,
    model_config: dict,
    video_config: dict,
) -> ImageCounterResult | VideoCounterResult:
    APP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    with TemporaryDirectory() as tmpdir:
        actual_input = input_path
        if uploaded_file is not None:
            actual_input = Path(tmpdir) / uploaded_file.name
            actual_input.write_bytes(uploaded_file.getbuffer())
        elif media_url.strip():
            actual_input = download_media_url(media_url, tmpdir)

        output_suffix = ".jpg" if media_type == "image" else ".mp4"
        output_path = APP_OUTPUT_DIR / f"{actual_input.stem}_{run_id}_processado{output_suffix}"
        summary_path = APP_OUTPUT_DIR / f"{actual_input.stem}_{run_id}_resumo.json"
        csv_path = APP_OUTPUT_DIR / f"{actual_input.stem}_{run_id}_frames.csv"

        detector = YoloDetector(
            model_path=model_config["model_path"],
            confidence=model_config["confidence"],
            iou=model_config["iou"],
        )
        roi = (
            RegionOfInterest.from_values(video_config["roi"])
            if video_config.get("roi_enabled")
            else None
        )

        with st.spinner("Processando..."):
            if media_type == "image":
                return process_image(
                    input_path=actual_input,
                    output_path=output_path,
                    detector=detector,
                    classes=model_config["classes"],
                    summary_output=summary_path,
                    roi=roi,
                )

            return process_video(
                input_path=actual_input,
                output_path=output_path,
                detector=detector,
                classes=model_config["classes"],
                csv_output=csv_path,
                summary_output=summary_path,
                max_frames=video_config["max_frames"],
                counting_mode=video_config["counting_mode"],
                line_orientation=video_config["line_orientation"],
                line_position=video_config["line_position"],
                line_direction=video_config["line_direction"],
                roi=roi,
                tracking_backend=video_config["tracking_backend"],
            )


def render_roi_drawer(input_path: Path | None, uploaded_file, media_type: str | None) -> None:
    if media_type not in {"image", "video"}:
        return
    if st.session_state.get("roi_mode") != "Desenhar na imagem":
        return
    if input_path is None:
        return

    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.caption("Instale `streamlit-drawable-canvas` para desenhar a ROI visualmente.")
        return

    image = preview_image_for_roi(input_path, uploaded_file, media_type)
    if image is None:
        st.caption("Não foi possível extrair uma prévia para desenhar a ROI.")
        return

    max_width = 520
    scale = min(1.0, max_width / image.width)
    display_size = (int(image.width * scale), int(image.height * scale))
    display_image = image.resize(display_size)
    caption = (
        "Desenhe um retângulo abaixo para atualizar a ROI visual."
        if media_type == "image"
        else "Desenhe a ROI sobre o primeiro frame do vídeo."
    )
    st.caption(caption)
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0.18)",
        stroke_width=2,
        stroke_color="#050505",
        background_image=display_image,
        update_streamlit=True,
        height=display_size[1],
        width=display_size[0],
        drawing_mode="rect",
        key=f"roi_canvas_{media_type}_{input_path.name}",
    )

    objects = canvas.json_data.get("objects", []) if canvas.json_data else []
    rectangles = [item for item in objects if item.get("type") == "rect"]
    if not rectangles:
        return

    rect = rectangles[-1]
    left = max(0.0, float(rect.get("left", 0)))
    top = max(0.0, float(rect.get("top", 0)))
    width = max(1.0, float(rect.get("width", 1)) * float(rect.get("scaleX", 1)))
    height = max(1.0, float(rect.get("height", 1)) * float(rect.get("scaleY", 1)))
    roi = [
        round(left / display_size[0], 4),
        round(top / display_size[1], 4),
        round(min(display_size[0], left + width) / display_size[0], 4),
        round(min(display_size[1], top + height) / display_size[1], 4),
    ]
    st.session_state["drawn_roi"] = roi
    st.success(f"ROI desenhada: {roi}")


def preview_image_for_roi(
    input_path: Path,
    uploaded_file,
    media_type: str,
) -> Image.Image | None:
    if media_type == "image":
        if uploaded_file is not None:
            return Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
        return Image.open(input_path).convert("RGB")

    if uploaded_file is not None:
        with TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / uploaded_file.name
            temp_path.write_bytes(uploaded_file.getbuffer())
            return first_video_frame(temp_path)
    return first_video_frame(input_path)


def first_video_frame(path: Path) -> Image.Image | None:
    try:
        import cv2
    except ImportError:
        return None

    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok or frame is None:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)


def render_error_analysis(report: pd.DataFrame) -> None:
    if report.empty:
        return
    error_rows = pd.DataFrame(error_analysis_rows(report.to_dict(orient="records")))
    st.subheader("Análise de Erros")
    false_positive_total = int(error_rows["false_positive_count"].sum())
    false_negative_total = int(error_rows["false_negative_count"].sum())
    affected_rows = int((error_rows["absolute_error"] > 0).sum())
    render_metric_row(
        [
            ("Linhas com erro", affected_rows),
            ("Falsos positivos", false_positive_total),
            ("Falsos negativos", false_negative_total),
            ("Status dominante", error_rows["status"].mode().iloc[0]),
        ]
    )
    status_options = ["Todos"] + sorted(error_rows["status"].dropna().astype(str).unique())
    selected_status = st.selectbox("Filtrar análise", status_options, key="error_status_filter")
    if selected_status != "Todos":
        error_rows = error_rows[error_rows["status"] == selected_status]

    columns = [
        "sample_id",
        "class_name",
        "expected_count",
        "predicted_count",
        "absolute_error",
        "false_positive_count",
        "false_negative_count",
        "status",
        "suggested_issue",
        "condition_tags",
    ]
    visible_columns = [column for column in columns if column in error_rows.columns]
    render_dataframe(error_rows[visible_columns], height=300)

    if "suggested_issue" in error_rows.columns:
        grouped = (
            error_rows.groupby(["status", "suggested_issue"], dropna=False)
            .size()
            .reset_index(name="linhas")
            .sort_values(["status", "linhas"], ascending=[True, False])
        )
        st.caption("Resumo por status e causa provável")
        render_dataframe(grouped)


def render_threshold_panel() -> None:
    threshold_path = EVALUATION_DIR / "threshold_search.csv"
    best_path = EVALUATION_DIR / "best_thresholds.json"
    if not threshold_path.exists():
        st.info("Rode `python scripts/optimize_thresholds.py` para gerar o ranking.")
        return

    thresholds = pd.read_csv(threshold_path)
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8")).get("best_config", {})
        render_metric_row(
            [
                ("Confiança", best.get("confidence", "-")),
                ("IOU", best.get("iou", "-")),
                ("MAE", best.get("mean_absolute_count_error", "-")),
                ("Erro total", best.get("total_absolute_error", "-")),
            ]
        )
    render_dataframe(thresholds, height=280)


def render_model_comparison_panel() -> None:
    comparison_path = EVALUATION_DIR / "model_comparison.csv"
    best_path = EVALUATION_DIR / "best_model.json"
    if not comparison_path.exists():
        st.info("Rode `python scripts/compare_models.py --models yolov8n.pt yolov8s.pt`.")
        return

    comparison = pd.read_csv(comparison_path)
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8")).get("best_model", {})
        render_metric_row(
            [
                ("Modelo", best.get("model", "-")),
                ("MAE", best.get("mean_absolute_count_error", "-")),
                ("Erro total", best.get("total_absolute_error", "-")),
                ("s/amostra", best.get("seconds_per_sample", "-")),
            ]
        )
    render_dataframe(comparison, height=280)


def append_history(
    result: dict,
    media_type: str | None,
    model_config: dict,
    video_config: dict,
) -> None:
    record = build_history_record(result, media_type, model_config, video_config)
    append_history_csv(HISTORY_PATH, record)


def load_line_events(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()

    rows = []
    frame_table = pd.read_csv(csv_path)
    if "new_events_json" not in frame_table.columns:
        return pd.DataFrame()

    for row in frame_table.to_dict(orient="records"):
        raw_events = row.get("new_events_json")
        if not raw_events or pd.isna(raw_events):
            continue

        try:
            events = json.loads(raw_events)
        except json.JSONDecodeError:
            continue

        for event in events:
            rows.append(
                {
                    "frame": row.get("frame_index"),
                    "tempo_s": row.get("timestamp_seconds"),
                    "track_id": event.get("track_id"),
                    "classe": event.get("label"),
                    "direção": event.get("direction"),
                }
            )

    return pd.DataFrame(rows)


def mime_for_path(path: Path, media_type: str | None) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix == ".csv":
        return "text/csv"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".mp4" or media_type == "video":
        return "video/mp4"
    return "application/octet-stream"


def resolve_input_path(uploaded_file, selected_sample: str, media_url: str = "") -> Path | None:
    if uploaded_file is not None:
        return Path(uploaded_file.name)
    if media_url.strip():
        try:
            return Path(media_filename_from_url(media_url))
        except ValueError as exc:
            st.warning(str(exc))
            return None
    if selected_sample != "Nenhuma":
        return SAMPLES_DIR / selected_sample
    return None


def get_sample_files() -> list[Path]:
    return sorted(path for path in SAMPLES_DIR.glob("*") if path.suffix.lower() in SUPPORTED_MEDIA)


def deduplicate_classes(classes: list[str]) -> list[str]:
    deduplicated = []
    seen = set()
    for class_name in classes:
        normalized = class_name.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(normalized)
    return deduplicated


if __name__ == "__main__":
    main()
