from __future__ import annotations

from object_counter.training import (
    build_training_command,
    build_yolo_data_yaml,
    parse_class_names,
    training_recommendation,
)


def test_parse_class_names_deduplicates_lines_and_commas() -> None:
    names = parse_class_names("bus\ncar, truck, bus,,person")

    assert names == ["bus", "car", "truck", "person"]


def test_training_recommendation_scales_by_task_and_goal() -> None:
    recommendation = training_recommendation(
        task="Segmentação",
        quality_goal="Prova de conceito",
        class_names=["bus", "car"],
        annotated_images=120,
    )

    assert recommendation.class_count == 2
    assert recommendation.min_images == 125
    assert recommendation.status == "insuficiente"


def test_build_yolo_data_yaml_and_training_command() -> None:
    yaml_text = build_yolo_data_yaml("datasets/transito", ["bus", "car"])
    command = build_training_command(
        task="Detecção",
        model="yolov8n.pt",
        data_yaml_path="datasets/transito/data.yaml",
        epochs=50,
        imgsz=640,
        device="0",
    )

    assert "path: datasets/transito" in yaml_text
    assert "0: bus" in yaml_text
    assert "1: car" in yaml_text
    assert command == (
        "yolo detect train model=yolov8n.pt data=datasets/transito/data.yaml "
        "epochs=50 imgsz=640 device=0"
    )
