from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from object_counter.detection.detector import Detection

LineOrientation = Literal["horizontal", "vertical"]
LineDirection = Literal["both", "positive", "negative"]


@dataclass(frozen=True)
class CountLine:
    orientation: LineOrientation = "horizontal"
    position_ratio: float = 0.5
    direction: LineDirection = "both"

    def __post_init__(self) -> None:
        if self.orientation not in {"horizontal", "vertical"}:
            raise ValueError("orientation precisa ser 'horizontal' ou 'vertical'.")
        if not 0.0 < self.position_ratio < 1.0:
            raise ValueError("position_ratio precisa estar entre 0 e 1.")
        if self.direction not in {"both", "positive", "negative"}:
            raise ValueError("direction precisa ser 'both', 'positive' ou 'negative'.")

    def pixel_position(self, frame_shape: tuple[int, ...]) -> int:
        height, width = frame_shape[:2]
        if self.orientation == "horizontal":
            return int(height * self.position_ratio)
        return int(width * self.position_ratio)

    def signed_distance(self, detection: Detection, frame_shape: tuple[int, ...]) -> float:
        center_x, center_y = detection.center
        line_position = self.pixel_position(frame_shape)
        if self.orientation == "horizontal":
            return center_y - line_position
        return center_x - line_position

    def as_dict(self, frame_shape: tuple[int, ...] | None = None) -> dict:
        data = {
            "orientation": self.orientation,
            "position_ratio": self.position_ratio,
            "direction": self.direction,
        }
        if frame_shape is not None:
            data["pixel_position"] = self.pixel_position(frame_shape)
        return data


@dataclass
class LineCountEvent:
    frame_index: int
    track_id: int
    label: str
    direction: str

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "track_id": self.track_id,
            "label": self.label,
            "direction": self.direction,
        }


@dataclass
class LineCounter:
    line: CountLine
    previous_positions: dict[int, float] = field(default_factory=dict)
    counted_track_ids: set[int] = field(default_factory=set)
    counts_by_label: Counter[str] = field(default_factory=Counter)
    events: list[LineCountEvent] = field(default_factory=list)

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, ...],
        frame_index: int,
    ) -> list[LineCountEvent]:
        new_events: list[LineCountEvent] = []

        for detection in detections:
            if detection.track_id is None:
                continue

            current = self.line.signed_distance(detection, frame_shape)
            previous = self.previous_positions.get(detection.track_id)
            self.previous_positions[detection.track_id] = current

            if previous is None or detection.track_id in self.counted_track_ids:
                continue

            direction = crossing_direction(previous, current)
            if direction is None or not self._direction_allowed(direction):
                continue

            event = LineCountEvent(
                frame_index=frame_index,
                track_id=detection.track_id,
                label=detection.label,
                direction=direction,
            )
            self.counted_track_ids.add(detection.track_id)
            self.counts_by_label[detection.label] += 1
            self.events.append(event)
            new_events.append(event)

        return new_events

    def counts(self) -> dict[str, int]:
        return dict(sorted(self.counts_by_label.items()))

    def total(self) -> int:
        return sum(self.counts_by_label.values())

    def _direction_allowed(self, direction: str) -> bool:
        return self.line.direction == "both" or self.line.direction == direction


def crossing_direction(previous: float, current: float) -> str | None:
    if previous < 0 <= current:
        return "positive"
    if previous > 0 >= current:
        return "negative"
    return None
