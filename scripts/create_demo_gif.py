from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gera um GIF leve a partir de um vídeo processado.")
    parser.add_argument("--input", default="reports/videos/bus_demo_line.mp4", help="Vídeo de entrada.")
    parser.add_argument("--output", default="reports/videos/demo.gif", help="GIF gerado.")
    parser.add_argument("--width", type=int, default=720, help="Largura máxima do GIF.")
    parser.add_argument("--fps", type=int, default=10, help="FPS aproximado do GIF.")
    parser.add_argument("--max-frames", type=int, default=48, help="Número máximo de frames no GIF.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        import cv2
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Instale as dependências com: pip install -r requirements.txt") from exc

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {input_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 24
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or args.max_frames
    stride = max(1, round(source_fps / args.fps))
    frames: list[Image.Image] = []
    frame_index = 0

    try:
        while len(frames) < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % stride == 0:
                frame = resize_frame(cv2, frame, max_width=args.width)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))

            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"Nenhum frame foi lido de: {input_path}")

    duration_ms = max(1, int(1000 / args.fps))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    print(
        f"GIF criado em: {output_path} "
        f"({len(frames)} frames de {source_frames}, {duration_ms} ms/frame)"
    )
    return 0


def resize_frame(cv2, frame, max_width: int):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / width
    new_size = (max_width, int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    raise SystemExit(main())
