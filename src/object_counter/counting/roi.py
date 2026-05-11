from __future__ import annotations

from dataclasses import dataclass

from object_counter.detection.detector import Detection


@dataclass(frozen=True)
class RegionOfInterest:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = [self.x_min, self.y_min, self.x_max, self.y_max]
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("Coordenadas da ROI precisam estar entre 0 e 1.")
        if self.x_min >= self.x_max:
            raise ValueError("x_min precisa ser menor que x_max.")
        if self.y_min >= self.y_max:
            raise ValueError("y_min precisa ser menor que y_max.")

    @classmethod
    def from_values(cls, values: list[float] | tuple[float, float, float, float]):
        if len(values) != 4:
            raise ValueError("ROI precisa receber quatro valores: x_min y_min x_max y_max.")
        return cls(*[float(value) for value in values])

    def contains(self, detection: Detection, frame_shape: tuple[int, ...]) -> bool:
        center_x, center_y = detection.center
        x_min, y_min, x_max, y_max = self.pixel_bounds(frame_shape)
        return x_min <= center_x <= x_max and y_min <= center_y <= y_max

    def filter(self, detections: list[Detection], frame_shape: tuple[int, ...]) -> list[Detection]:
        return [detection for detection in detections if self.contains(detection, frame_shape)]

    def pixel_bounds(self, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        return (
            int(width * self.x_min),
            int(height * self.y_min),
            int(width * self.x_max),
            int(height * self.y_max),
        )

    def as_dict(self, frame_shape: tuple[int, ...] | None = None) -> dict:
        data = {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }
        if frame_shape is not None:
            x_min, y_min, x_max, y_max = self.pixel_bounds(frame_shape)
            data.update(
                {
                    "x_min_px": x_min,
                    "y_min_px": y_min,
                    "x_max_px": x_max,
                    "y_max_px": y_max,
                }
            )
        return data
