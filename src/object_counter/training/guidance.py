from __future__ import annotations

from dataclasses import dataclass


TRAINING_TASKS = {
    "Detecção": {
        "yolo_task": "detect",
        "default_model": "yolov8n.pt",
        "annotation": "caixas delimitadoras em cada objeto de interesse",
        "format": "um arquivo .txt por imagem com classe e caixa normalizada",
    },
    "Segmentação": {
        "yolo_task": "segment",
        "default_model": "yolov8n-seg.pt",
        "annotation": "polígonos contornando cada instância",
        "format": "um arquivo .txt por imagem com classe e pontos do polígono",
    },
    "Pose": {
        "yolo_task": "pose",
        "default_model": "yolov8n-pose.pt",
        "annotation": "pontos-chave marcados em cada pessoa/objeto articulado",
        "format": "um arquivo .txt por imagem com caixa e keypoints normalizados",
    },
}

QUALITY_GOALS = {
    "Prova de conceito": (50, 100),
    "Portfólio consistente": (200, 500),
    "Produção inicial": (1000, 3000),
}


@dataclass(frozen=True)
class TrainingRecommendation:
    class_count: int
    min_images: int
    target_images: int
    current_images: int
    status: str
    message: str


def parse_class_names(raw_text: str) -> list[str]:
    names = []
    seen = set()
    normalized_text = raw_text.replace("\n", ",")
    for item in normalized_text.split(","):
        class_name = item.strip()
        if not class_name or class_name in seen:
            continue
        names.append(class_name)
        seen.add(class_name)
    return names


def training_recommendation(
    task: str,
    quality_goal: str,
    class_names: list[str],
    annotated_images: int,
) -> TrainingRecommendation:
    class_count = max(len(class_names), 1)
    min_per_class, target_per_class = QUALITY_GOALS[quality_goal]
    multiplier = _task_multiplier(task)
    min_images = int(min_per_class * class_count * multiplier)
    target_images = int(target_per_class * class_count * multiplier)

    if annotated_images < min_images:
        status = "insuficiente"
        message = "Ainda faltam imagens marcadas para treinar com confiança."
    elif annotated_images < target_images:
        status = "bom com ressalvas"
        message = "Já dá para treinar um baseline, mas mais variedade deve melhorar o resultado."
    else:
        status = "bom"
        message = "Volume adequado para uma primeira versão robusta, desde que as marcações estejam consistentes."

    return TrainingRecommendation(
        class_count=class_count,
        min_images=min_images,
        target_images=target_images,
        current_images=annotated_images,
        status=status,
        message=message,
    )


def build_yolo_data_yaml(dataset_path: str, class_names: list[str]) -> str:
    names = class_names or ["classe_0"]
    lines = [
        f"path: {dataset_path}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for index, class_name in enumerate(names):
        lines.append(f"  {index}: {class_name}")
    return "\n".join(lines) + "\n"


def build_training_command(
    task: str,
    model: str,
    data_yaml_path: str,
    epochs: int,
    imgsz: int,
    device: str = "",
) -> str:
    yolo_task = TRAINING_TASKS[task]["yolo_task"]
    command = (
        f"yolo {yolo_task} train model={model} data={data_yaml_path} "
        f"epochs={epochs} imgsz={imgsz}"
    )
    if device.strip():
        command += f" device={device.strip()}"
    return command


def _task_multiplier(task: str) -> float:
    if task == "Segmentação":
        return 1.25
    if task == "Pose":
        return 1.5
    return 1.0
