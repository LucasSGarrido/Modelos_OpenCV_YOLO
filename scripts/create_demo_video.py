from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cria um vídeo curto de demo a partir de uma imagem.")
    parser.add_argument("--input", default="data/samples/bus.jpg", help="Imagem base.")
    parser.add_argument("--output", default="data/samples/bus_demo.mp4", help="Vídeo gerado.")
    parser.add_argument("--frames", type=int, default=72, help="Quantidade de frames.")
    parser.add_argument("--fps", type=float, default=24.0, help="FPS do vídeo.")
    parser.add_argument("--start-offset", type=int, default=-120, help="Deslocamento vertical inicial.")
    parser.add_argument("--end-offset", type=int, default=140, help="Deslocamento vertical final.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Instale as dependências com: pip install -r requirements.txt") from exc

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Não foi possível abrir a imagem: {input_path}")

    height, width = image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Não foi possível criar o vídeo: {output_path}")

    offsets = np.linspace(args.start_offset, args.end_offset, args.frames)
    background = np.full_like(image, fill_value=28)

    try:
        for offset in offsets:
            matrix = np.float32([[1, 0, 0], [0, 1, float(offset)]])
            frame = cv2.warpAffine(
                image,
                matrix,
                (width, height),
                dst=background.copy(),
                borderMode=cv2.BORDER_TRANSPARENT,
            )
            writer.write(frame)
    finally:
        writer.release()

    print(f"Vídeo criado em: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
