from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app_pages.shared import render_page_header
from object_counter.training import (
    TRAINING_TASKS,
    build_training_command,
    build_yolo_data_yaml,
    parse_class_names,
    training_recommendation,
)


def render_custom_training_page() -> None:
    render_page_header(
        "Treinamento Customizado",
        "Planeje um YOLO ajustado para as suas imagens marcadas.",
        ["Dataset", "Marcações", "YOLO Train"],
    )

    controls = _render_training_controls()
    class_names = parse_class_names(controls["classes_text"])
    recommendation = training_recommendation(
        task=controls["task"],
        quality_goal=controls["quality_goal"],
        class_names=class_names,
        annotated_images=controls["annotated_images"],
    )
    data_yaml = build_yolo_data_yaml(controls["dataset_path"], class_names)
    data_yaml_path = str(Path(controls["dataset_path"]) / "data.yaml").replace("\\", "/")
    command = build_training_command(
        task=controls["task"],
        model=controls["base_model"],
        data_yaml_path=data_yaml_path,
        epochs=controls["epochs"],
        imgsz=controls["imgsz"],
        device=controls["device"],
    )

    intro_col, status_col = st.columns([0.58, 0.42], gap="large")
    with intro_col:
        _render_training_explanation(controls["task"])
    with status_col:
        _render_recommendation(recommendation)

    tab_flow, tab_dataset, tab_command = st.tabs(
        ["Fluxo", "Estrutura do dataset", "Comando"]
    )
    with tab_flow:
        _render_annotation_flow(controls["task"])
    with tab_dataset:
        _render_dataset_structure(data_yaml, class_names)
    with tab_command:
        st.code(command, language="bash")
        st.download_button(
            "Baixar data.yaml",
            data=data_yaml.encode("utf-8"),
            file_name="data.yaml",
            mime="text/yaml",
        )


def _render_training_controls() -> dict:
    with st.sidebar:
        st.subheader("Treinamento")
        task_options = list(TRAINING_TASKS)
        _migrate_session_label(
            "training_task",
            {"Deteccao": "Detecção", "Segmentacao": "Segmentação"},
            task_options[0],
            task_options,
        )
        task = st.selectbox("Tarefa", task_options, key="training_task")
        classes_text = st.text_area(
            "Classes alvo",
            value="bus\ncar\ntruck\nperson",
            help="Uma classe por linha ou separada por vírgula. Use nomes consistentes com a marcação.",
            key="training_classes",
        )
        dataset_path = st.text_input(
            "Pasta do dataset",
            value="datasets/meu_yolo",
            help="Pasta que terá images/train, images/val, labels/train, labels/val e data.yaml.",
            key="training_dataset_path",
        )
        quality_options = ["Prova de conceito", "Portfólio consistente", "Produção inicial"]
        _migrate_session_label(
            "training_quality_goal",
            {
                "Portfolio consistente": "Portfólio consistente",
                "Producao inicial": "Produção inicial",
            },
            quality_options[1],
            quality_options,
        )
        quality_goal = st.selectbox(
            "Meta de qualidade",
            quality_options,
            index=1,
            key="training_quality_goal",
        )
        annotated_images = st.number_input(
            "Imagens marcadas",
            min_value=0,
            max_value=100000,
            value=200,
            step=25,
            help="Conte frames extraídos de vídeos como imagens marcadas.",
            key="training_annotated_images",
        )
        base_model = st.text_input(
            "Modelo base",
            value=TRAINING_TASKS[task]["default_model"],
            help="Comece pelo modelo nano para validar e suba para s/m se houver máquina e dados.",
            key=f"training_base_model_{task}",
        )
        epochs = st.number_input(
            "Épocas",
            min_value=1,
            max_value=500,
            value=80,
            step=10,
            key="training_epochs",
        )
        imgsz = st.number_input(
            "imgsz",
            min_value=320,
            max_value=1280,
            value=640,
            step=32,
            key="training_imgsz",
        )
        device = st.text_input(
            "Device",
            value="",
            placeholder="cpu, 0, 0,1",
            help="Vazio deixa o Ultralytics escolher. Use 0 para GPU CUDA principal.",
            key="training_device",
        )

    return {
        "task": task,
        "classes_text": classes_text,
        "dataset_path": dataset_path,
        "quality_goal": quality_goal,
        "annotated_images": int(annotated_images),
        "base_model": base_model,
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "device": device,
    }


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


def _render_training_explanation(task: str) -> None:
    config = TRAINING_TASKS[task]
    st.subheader("Como funciona")
    st.write(
        "Para treinar um modelo customizado, primeiro você separa imagens ou frames de vídeo, "
        "marca manualmente o objeto-alvo e exporta essas marcações no formato YOLO. "
        "O treino usa um peso pré-treinado como ponto de partida e ajusta o modelo para o seu domínio."
    )
    st.write(
        f"Nesta tarefa, o foco é em {config['annotation']}. "
        f"O formato esperado é {config['format']}."
    )
    st.info(
        "Vídeos não são marcados diretamente pelo YOLO. Extraia frames variados, marque esses frames "
        "e mantenha frames de validação vindos de vídeos/cenas diferentes."
    )


def _render_recommendation(recommendation) -> None:  # noqa: ANN001
    st.subheader("Volume recomendado")
    col_a, col_b = st.columns(2)
    col_a.metric("Mínimo sugerido", recommendation.min_images)
    col_b.metric("Alvo melhor", recommendation.target_images)
    st.metric("Imagens marcadas", recommendation.current_images)
    if recommendation.status == "insuficiente":
        st.warning(recommendation.message)
    elif recommendation.status == "bom com ressalvas":
        st.info(recommendation.message)
    else:
        st.success(recommendation.message)


def _render_annotation_flow(task: str) -> None:
    config = TRAINING_TASKS[task]
    rows = [
        {
            "etapa": "1. Coletar",
            "ação": "separe imagens/frames variados do objeto real",
            "cuidado": "evite imagens quase iguais dominando o dataset",
        },
        {
            "etapa": "2. Marcar",
            "ação": config["annotation"],
            "cuidado": "use a mesma regra de marcação em todas as imagens",
        },
        {
            "etapa": "3. Dividir",
            "ação": "use treino e validação separados",
            "cuidado": "não coloque frames quase idênticos nos dois conjuntos",
        },
        {
            "etapa": "4. Treinar",
            "ação": "rode o comando YOLO gerado nesta página",
            "cuidado": "acompanhe métricas de validação, não apenas loss de treino",
        },
        {
            "etapa": "5. Testar",
            "ação": "teste em imagens e vídeos que não entraram no treino",
            "cuidado": "documente falhas por oclusão, luz e objetos pequenos",
        },
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_dataset_structure(data_yaml: str, class_names: list[str]) -> None:
    st.subheader("Estrutura esperada")
    st.code(
        """datasets/meu_yolo/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml""",
        language="text",
    )
    st.subheader("Classes")
    st.dataframe(
        pd.DataFrame(
            [{"id": index, "classe": name} for index, name in enumerate(class_names or ["classe_0"])]
        ),
        hide_index=True,
        width="stretch",
    )
    st.subheader("data.yaml")
    st.code(data_yaml, language="yaml")
