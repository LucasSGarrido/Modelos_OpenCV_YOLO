from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Detection:
    """Single object detection in xyxy format."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str
    track_id: int | None = None

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class YoloDetector:
    """Thin wrapper around Ultralytics YOLO.

    The import is lazy so the project can be tested without forcing the model
    dependency to load during unit tests.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        iou: float = 0.5,
        device: str | None = None,
        imgsz: int = 640,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            message = (
                "A dependência 'ultralytics' não está instalada. "
                "Instale com: pip install -r requirements.txt"
            )
            raise RuntimeError(message) from exc

        return YOLO(model_path)

    def detect(self, frame: Any, classes: Iterable[str] | None = None) -> list[Detection]:
        return self._run_inference(frame, classes=classes, use_tracking=False)

    def track(
        self,
        frame: Any,
        classes: Iterable[str] | None = None,
        tracker: str = "bytetrack.yaml",
        persist: bool = True,
    ) -> list[Detection]:
        return self._run_inference(
            frame,
            classes=classes,
            use_tracking=True,
            tracker=tracker,
            persist=persist,
        )

    def _run_inference(
        self,
        frame: Any,
        classes: Iterable[str] | None = None,
        use_tracking: bool = False,
        tracker: str = "bytetrack.yaml",
        persist: bool = True,
    ) -> list[Detection]:
        class_filter = {item.lower() for item in classes or []}
        predict_args = {
            "source": frame,
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
        }
        if use_tracking:
            results = self.model.track(**predict_args, tracker=tracker, persist=persist)
        else:
            results = self.model.predict(**predict_args)

        if not results:
            return []

        result = results[0]
        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        detections: list[Detection] = []

        for box in getattr(result, "boxes", []) or []:
            class_id = int(box.cls[0].item())
            label = _class_label(names, class_id)

            if class_filter and label.lower() not in class_filter:
                continue

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf[0].item())
            track_id = None
            if getattr(box, "id", None) is not None:
                track_id = int(box.id[0].item())
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    label=label,
                    track_id=track_id,
                )
            )

        return detections


def _class_label(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)
