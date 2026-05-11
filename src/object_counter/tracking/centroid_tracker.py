from __future__ import annotations

from dataclasses import dataclass, replace
from math import dist

from object_counter.detection.detector import Detection


@dataclass
class Track:
    track_id: int
    label: str
    center: tuple[float, float]
    missing_frames: int = 0


class CentroidTracker:
    """Class-aware centroid tracker for short videos.

    It is intentionally small: enough for portfolio demos and line-counting
    logic, while keeping the production-grade ByteTrack/DeepSORT upgrade clear.
    """

    def __init__(self, max_distance: float = 80.0, max_missing: int = 10) -> None:
        if max_distance <= 0:
            raise ValueError("max_distance precisa ser maior que zero.")
        if max_missing < 0:
            raise ValueError("max_missing não pode ser negativo.")

        self.max_distance = max_distance
        self.max_missing = max_missing
        self._tracks: dict[int, Track] = {}
        self._next_track_id = 1

    @property
    def tracks(self) -> dict[int, Track]:
        return dict(self._tracks)

    def update(self, detections: list[Detection]) -> list[Detection]:
        unmatched_track_ids = set(self._tracks)
        tracked_detections: list[Detection] = []

        for detection in detections:
            track_id = self._best_track_for_detection(detection, unmatched_track_ids)
            if track_id is None:
                track_id = self._create_track(detection)
            else:
                self._tracks[track_id] = Track(
                    track_id=track_id,
                    label=detection.label,
                    center=detection.center,
                    missing_frames=0,
                )
                unmatched_track_ids.remove(track_id)

            tracked_detections.append(replace(detection, track_id=track_id))

        self._mark_missing_tracks(unmatched_track_ids)
        return tracked_detections

    def _best_track_for_detection(
        self, detection: Detection, candidate_track_ids: set[int]
    ) -> int | None:
        best_track_id: int | None = None
        best_distance = self.max_distance

        for track_id in candidate_track_ids:
            track = self._tracks[track_id]
            if track.label != detection.label:
                continue

            distance = dist(track.center, detection.center)
            if distance <= best_distance:
                best_distance = distance
                best_track_id = track_id

        return best_track_id

    def _create_track(self, detection: Detection) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        self._tracks[track_id] = Track(
            track_id=track_id,
            label=detection.label,
            center=detection.center,
            missing_frames=0,
        )
        return track_id

    def _mark_missing_tracks(self, track_ids: set[int]) -> None:
        expired_track_ids: list[int] = []

        for track_id in track_ids:
            track = self._tracks[track_id]
            track.missing_frames += 1
            if track.missing_frames > self.max_missing:
                expired_track_ids.append(track_id)

        for track_id in expired_track_ids:
            del self._tracks[track_id]
