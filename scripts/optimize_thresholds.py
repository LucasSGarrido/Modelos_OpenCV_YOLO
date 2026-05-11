from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from object_counter.config import DEFAULT_MODEL_PATH  # noqa: E402
from object_counter.counting.image_counter import process_image  # noqa: E402
from object_counter.counting.roi import RegionOfInterest  # noqa: E402
from object_counter.counting.video_counter import process_video  # noqa: E402
from object_counter.detection.detector import YoloDetector  # noqa: E402
from object_counter.evaluation.count_report import load_annotations  # noqa: E402
from object_counter.evaluation.threshold_search import (  # noqa: E402
    ThresholdConfig,
    build_threshold_grid,
    classes_from_annotations,
    evaluation_rows_from_summaries,
    score_threshold_config,
    sort_threshold_scores,
    unique_samples,
)
from object_counter.utils.io import ensure_parent_dir, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Busca a melhor combinação de confidence/IOU para reduzir erro de contagem."
    )
    parser.add_argument("--annotations", default="data/annotations/counts.csv")
    parser.add_argument("--output", default="reports/evaluation/threshold_search.csv")
    parser.add_argument("--best-output", default="reports/evaluation/best_thresholds.json")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Peso YOLO. Ex.: yolov8n.pt")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Classes YOLO. Se omitido, usa as classes presentes nas anotações.",
    )
    parser.add_argument(
        "--confidence-values",
        nargs="*",
        type=float,
        default=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    )
    parser.add_argument(
        "--iou-values",
        nargs="*",
        type=float,
        default=[0.4, 0.5, 0.6, 0.7],
    )
    parser.add_argument("--device", default=None, help="Dispositivo do YOLO. Ex.: cpu, 0")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho de inferência do YOLO.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--tracking-backend",
        choices=["centroid", "bytetrack"],
        default="centroid",
    )
    parser.add_argument(
        "--line-orientation",
        choices=["horizontal", "vertical"],
        default="horizontal",
    )
    parser.add_argument("--line-position", type=float, default=0.5)
    parser.add_argument(
        "--line-direction",
        choices=["both", "positive", "negative"],
        default="both",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="Região de interesse em coordenadas relativas.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Mantém imagens/vídeos temporários gerados durante a busca.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="reports/evaluation/threshold_artifacts",
        help="Pasta usada quando --keep-artifacts estiver ativo.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    annotations = load_annotations(resolve_project_path(args.annotations))
    if not annotations:
        print("Nenhuma anotação encontrada para otimizar.", file=sys.stderr)
        return 1

    samples = unique_samples(annotations)
    classes = args.classes if args.classes is not None else classes_from_annotations(annotations)
    configs = build_threshold_grid(args.confidence_values, args.iou_values)
    roi = RegionOfInterest.from_values(args.roi) if args.roi else None

    if args.keep_artifacts:
        artifact_root = resolve_project_path(args.artifact_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)
        scores, details = run_search(args, configs, samples, annotations, classes, roi, artifact_root)
    else:
        with TemporaryDirectory() as tmpdir:
            scores, details = run_search(
                args,
                configs,
                samples,
                annotations,
                classes,
                roi,
                Path(tmpdir),
            )

    ranked_scores = sort_threshold_scores(scores)
    write_search_csv(ranked_scores, resolve_project_path(args.output))
    best = {
        "best_config": ranked_scores[0],
        "classes": classes,
        "configs_tested": len(configs),
        "samples": [sample.__dict__ for sample in samples],
        "details": details.get(config_slug(ranked_scores[0]), []),
    }
    write_json(best, resolve_project_path(args.best_output))
    print(json.dumps(best["best_config"], ensure_ascii=False, indent=2))
    return 0


def run_search(
    args: argparse.Namespace,
    configs: list[ThresholdConfig],
    samples,
    annotations,
    classes: list[str],
    roi: RegionOfInterest | None,
    artifact_root: Path,
) -> tuple[list[dict], dict[str, list[dict]]]:
    scores = []
    details_by_config: dict[str, list[dict]] = {}
    for config in configs:
        started_at = perf_counter()
        detector = YoloDetector(
            model_path=args.model,
            confidence=config.confidence,
            iou=config.iou,
            device=args.device,
            imgsz=args.imgsz,
        )
        summaries = {}
        config_dir = artifact_root / config.slug
        config_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            summaries[sample.sample_id] = process_sample(
                sample=sample,
                detector=detector,
                classes=classes,
                roi=roi,
                config_dir=config_dir,
                args=args,
            )

        evaluation_rows = evaluation_rows_from_summaries(annotations, summaries)
        processing_seconds = perf_counter() - started_at
        score = score_threshold_config(config, evaluation_rows, processing_seconds)
        scores.append(score)
        details_by_config[config.slug] = evaluation_rows

    return scores, details_by_config


def process_sample(
    sample,
    detector: YoloDetector,
    classes: list[str],
    roi: RegionOfInterest | None,
    config_dir: Path,
    args: argparse.Namespace,
) -> dict:
    input_path = resolve_project_path(sample.input_path)
    if sample.media_type == "image":
        result = process_image(
            input_path=input_path,
            output_path=config_dir / f"{sample.sample_id}.jpg",
            detector=detector,
            classes=classes,
            summary_output=config_dir / f"{sample.sample_id}.json",
            roi=roi,
        )
        return result.to_dict()

    result = process_video(
        input_path=input_path,
        output_path=config_dir / f"{sample.sample_id}.mp4",
        detector=detector,
        classes=classes,
        summary_output=config_dir / f"{sample.sample_id}.json",
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        counting_mode="frame" if sample.counting_mode == "max_frame" else sample.counting_mode,
        line_orientation=args.line_orientation,
        line_position=args.line_position,
        line_direction=args.line_direction,
        roi=roi,
        tracking_backend=args.tracking_backend,
    )
    return result.to_dict()


def write_search_csv(rows: list[dict], path: Path) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "confidence",
        "iou",
        "sample_count",
        "class_rows",
        "total_absolute_error",
        "mean_absolute_count_error",
        "mean_absolute_percentage_error",
        "exact_count_match_rate",
        "false_positive_count",
        "false_negative_count",
        "processing_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def config_slug(score: dict) -> str:
    return ThresholdConfig(
        confidence=float(score["confidence"]),
        iou=float(score["iou"]),
    ).slug


if __name__ == "__main__":
    raise SystemExit(main())
