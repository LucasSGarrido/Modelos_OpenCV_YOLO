from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from object_counter.config import DEFAULT_CONFIDENCE, DEFAULT_IOU, DEFAULT_MODEL_PATH
from object_counter.counting.roi import RegionOfInterest
from object_counter.counting.image_counter import process_image
from object_counter.counting.video_counter import process_video
from object_counter.detection.detector import YoloDetector
from object_counter.utils.io import default_output_path, default_summary_path, infer_media_type


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detecta e conta objetos em imagens ou vídeos usando YOLO."
    )
    parser.add_argument("--input", required=True, help="Caminho da imagem ou vídeo de entrada.")
    parser.add_argument("--output", help="Caminho do arquivo processado.")
    parser.add_argument("--summary-output", help="Caminho do JSON de resumo.")
    parser.add_argument("--csv-output", help="Caminho do CSV por frame, usado em vídeos.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Peso YOLO. Ex.: yolov8n.pt")
    parser.add_argument("--classes", nargs="*", help="Classes YOLO a manter. Ex.: person car bottle")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--device", default=None, help="Dispositivo do YOLO. Ex.: cpu, 0")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho de inferência do YOLO.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Processa 1 a cada N frames.")
    parser.add_argument("--max-frames", type=int, help="Limita frames para testes rápidos.")
    parser.add_argument(
        "--counting-mode",
        choices=["frame", "line"],
        default="frame",
        help="Modo de contagem para vídeos: frame ou line.",
    )
    parser.add_argument(
        "--line-orientation",
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="Orientação da linha de contagem no modo line.",
    )
    parser.add_argument(
        "--line-position",
        type=float,
        default=0.5,
        help="Posição relativa da linha entre 0 e 1. Ex.: 0.5 para centro.",
    )
    parser.add_argument(
        "--line-direction",
        choices=["both", "positive", "negative"],
        default="both",
        help="Direção do cruzamento no modo line.",
    )
    parser.add_argument(
        "--tracker-max-distance",
        type=float,
        default=80.0,
        help="Distância máxima entre centros para manter o mesmo ID.",
    )
    parser.add_argument(
        "--tracker-max-missing",
        type=int,
        default=10,
        help="Frames que um track pode ficar ausente antes de expirar.",
    )
    parser.add_argument(
        "--tracking-backend",
        choices=["centroid", "bytetrack"],
        default="centroid",
        help="Backend de tracking para modo line.",
    )
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

    try:
        input_path = Path(args.input)
        media_type = infer_media_type(input_path)
        output_path = Path(args.output) if args.output else default_output_path(input_path, media_type)
        summary_output = (
            Path(args.summary_output) if args.summary_output else default_summary_path(output_path)
        )

        detector = YoloDetector(
            model_path=args.model,
            confidence=args.confidence,
            iou=args.iou,
            device=args.device,
            imgsz=args.imgsz,
        )
        roi = RegionOfInterest.from_values(args.roi) if args.roi else None

        if media_type == "image":
            result = process_image(
                input_path=input_path,
                output_path=output_path,
                detector=detector,
                classes=args.classes,
                summary_output=summary_output,
                roi=roi,
            )
        else:
            result = process_video(
                input_path=input_path,
                output_path=output_path,
                detector=detector,
                classes=args.classes,
                csv_output=args.csv_output,
                summary_output=summary_output,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
                counting_mode=args.counting_mode,
                line_orientation=args.line_orientation,
                line_position=args.line_position,
                line_direction=args.line_direction,
                tracker_max_distance=args.tracker_max_distance,
                tracker_max_missing=args.tracker_max_missing,
                roi=roi,
                tracking_backend=args.tracking_backend,
            )

        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
