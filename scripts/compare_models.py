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

from object_counter.config import DEFAULT_CONFIDENCE, DEFAULT_IOU, DEFAULT_MODEL_PATH  # noqa: E402
from object_counter.counting.image_counter import process_image  # noqa: E402
from object_counter.counting.roi import RegionOfInterest  # noqa: E402
from object_counter.counting.video_counter import process_video  # noqa: E402
from object_counter.detection.detector import YoloDetector  # noqa: E402
from object_counter.evaluation.count_report import load_annotations  # noqa: E402
from object_counter.evaluation.threshold_search import (  # noqa: E402
    classes_from_annotations,
    evaluation_rows_from_summaries,
    score_threshold_config,
    sort_threshold_scores,
    unique_samples,
)
from object_counter.evaluation.threshold_search import ThresholdConfig  # noqa: E402
from object_counter.utils.io import ensure_parent_dir, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara modelos YOLO usando as anotações manuais de contagem."
    )
    parser.add_argument("--annotations", default="data/annotations/counts.csv")
    parser.add_argument("--output", default="reports/evaluation/model_comparison.csv")
    parser.add_argument("--best-output", default="reports/evaluation/best_model.json")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL_PATH])
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--tracking-backend", choices=["centroid", "bytetrack"], default="centroid")
    parser.add_argument("--line-orientation", choices=["horizontal", "vertical"], default="horizontal")
    parser.add_argument("--line-position", type=float, default=0.5)
    parser.add_argument("--line-direction", choices=["both", "positive", "negative"], default="both")
    parser.add_argument("--roi", nargs=4, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    annotations = load_annotations(resolve_project_path(args.annotations))
    samples = unique_samples(annotations)
    classes = args.classes if args.classes is not None else classes_from_annotations(annotations)
    roi = RegionOfInterest.from_values(args.roi) if args.roi else None
    rows = []
    details_by_model = {}

    with TemporaryDirectory() as tmpdir:
        artifact_root = Path(tmpdir)
        for model_path in args.models:
            row, details = evaluate_model(
                args=args,
                model_path=model_path,
                samples=samples,
                annotations=annotations,
                classes=classes,
                roi=roi,
                artifact_root=artifact_root,
            )
            rows.append(row)
            details_by_model[model_path] = details

    ranked = sort_threshold_scores(rows)
    write_model_comparison_csv(ranked, resolve_project_path(args.output))
    best = {
        "best_model": ranked[0],
        "models_tested": args.models,
        "classes": classes,
        "details": details_by_model.get(ranked[0]["model"], []),
    }
    write_json(best, resolve_project_path(args.best_output))
    print(json.dumps(best["best_model"], ensure_ascii=False, indent=2))
    return 0


def evaluate_model(
    args: argparse.Namespace,
    model_path: str,
    samples,
    annotations,
    classes: list[str],
    roi: RegionOfInterest | None,
    artifact_root: Path,
) -> tuple[dict, list[dict]]:
    started_at = perf_counter()
    detector = YoloDetector(
        model_path=model_path,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
        imgsz=args.imgsz,
    )
    summaries = {}
    model_dir = artifact_root / safe_model_slug(model_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        summaries[sample.sample_id] = process_sample(
            sample=sample,
            detector=detector,
            classes=classes,
            roi=roi,
            model_dir=model_dir,
            args=args,
        )

    evaluation_rows = evaluation_rows_from_summaries(annotations, summaries)
    processing_seconds = perf_counter() - started_at
    score = score_threshold_config(
        ThresholdConfig(confidence=args.confidence, iou=args.iou),
        evaluation_rows,
        processing_seconds,
    )
    score["model"] = model_path
    score["seconds_per_sample"] = round(processing_seconds / max(1, len(samples)), 4)
    return score, evaluation_rows


def process_sample(
    sample,
    detector: YoloDetector,
    classes: list[str],
    roi: RegionOfInterest | None,
    model_dir: Path,
    args: argparse.Namespace,
) -> dict:
    input_path = resolve_project_path(sample.input_path)
    if sample.media_type == "image":
        return process_image(
            input_path=input_path,
            output_path=model_dir / f"{sample.sample_id}.jpg",
            detector=detector,
            classes=classes,
            summary_output=model_dir / f"{sample.sample_id}.json",
            roi=roi,
        ).to_dict()

    return process_video(
        input_path=input_path,
        output_path=model_dir / f"{sample.sample_id}.mp4",
        detector=detector,
        classes=classes,
        summary_output=model_dir / f"{sample.sample_id}.json",
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        counting_mode="frame" if sample.counting_mode == "max_frame" else sample.counting_mode,
        line_orientation=args.line_orientation,
        line_position=args.line_position,
        line_direction=args.line_direction,
        roi=roi,
        tracking_backend=args.tracking_backend,
    ).to_dict()


def write_model_comparison_csv(rows: list[dict], path: Path) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "model",
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
        "seconds_per_sample",
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


def safe_model_slug(model_path: str) -> str:
    return Path(model_path).stem.replace(".", "_").replace(" ", "_")


if __name__ == "__main__":
    raise SystemExit(main())
