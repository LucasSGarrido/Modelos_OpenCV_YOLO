from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from object_counter.config import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_MODEL_PATH,
    IMAGE_EXTENSIONS,
    PEOPLE_VEHICLE_CLASSES,
)
from object_counter.counting.image_counter import process_image  # noqa: E402
from object_counter.counting.roi import RegionOfInterest  # noqa: E402
from object_counter.detection.detector import YoloDetector  # noqa: E402
from object_counter.utils.io import ensure_parent_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Processa uma pasta de imagens e consolida contagens em CSV."
    )
    parser.add_argument("--input-dir", default="data/samples", help="Pasta com imagens.")
    parser.add_argument("--output-dir", default="reports/batch", help="Pasta de saída.")
    parser.add_argument("--summary-output", default="reports/batch/batch_summary.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Peso YOLO. Ex.: yolov8n.pt")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=PEOPLE_VEHICLE_CLASSES,
        help="Classes YOLO a manter. Padrão: pessoas e veículos.",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Desativa filtro de classes e conta todas as classes detectadas pelo modelo.",
    )
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--device", default=None, help="Dispositivo do YOLO. Ex.: cpu, 0")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho de inferência do YOLO.")
    parser.add_argument("--recursive", action="store_true", help="Busca imagens em subpastas.")
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="Região de interesse em coordenadas relativas. Ex.: --roi 0.1 0.2 0.9 0.8",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    summary_output = Path(args.summary_output)

    image_paths = list(iter_image_paths(input_dir, recursive=args.recursive))
    if not image_paths:
        print(f"Nenhuma imagem encontrada em {input_dir}.", file=sys.stderr)
        return 1

    detector = YoloDetector(
        model_path=args.model,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
        imgsz=args.imgsz,
    )
    classes = None if args.all_classes else args.classes
    roi = RegionOfInterest.from_values(args.roi) if args.roi else None
    rows = []

    for image_path in image_paths:
        safe_stem = safe_output_stem(input_dir, image_path)
        output_path = output_dir / f"{safe_stem}_processado.jpg"
        summary_path = output_dir / f"{safe_stem}_resumo.json"
        result = process_image(
            input_path=image_path,
            output_path=output_path,
            detector=detector,
            classes=classes,
            summary_output=summary_path,
            roi=roi,
        )
        rows.append(
            {
                "input_path": str(image_path),
                "output_path": result.output_path,
                "summary_path": result.summary_path,
                "total": result.total,
                "class_count": len(result.counts),
                "detections": len(result.detections),
                "inference_seconds": round(result.inference_seconds, 6),
                "roi_enabled": roi is not None,
                "counts_json": json.dumps(result.counts, ensure_ascii=False),
            }
        )

    write_batch_summary(rows, summary_output)
    print(json.dumps({"images_processed": len(rows), "summary_output": str(summary_output)}))
    return 0


def iter_image_paths(input_dir: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def safe_output_stem(input_dir: Path, image_path: Path) -> str:
    try:
        relative = image_path.relative_to(input_dir).with_suffix("")
    except ValueError:
        relative = image_path.with_suffix("")
    return "_".join(part for part in relative.parts if part)


def write_batch_summary(rows: list[dict], path: Path) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "input_path",
        "output_path",
        "summary_path",
        "total",
        "class_count",
        "detections",
        "inference_seconds",
        "roi_enabled",
        "counts_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
