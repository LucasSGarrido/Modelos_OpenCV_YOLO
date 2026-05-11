from object_counter.detection.detector import Detection
from object_counter.tracking.centroid_tracker import CentroidTracker


def test_centroid_tracker_keeps_same_id_for_nearby_detection() -> None:
    tracker = CentroidTracker(max_distance=50)

    first = tracker.update([Detection(0, 0, 20, 20, 0.9, 0, "person")])
    second = tracker.update([Detection(5, 0, 25, 20, 0.9, 0, "person")])

    assert first[0].track_id == second[0].track_id


def test_centroid_tracker_creates_new_id_for_different_label() -> None:
    tracker = CentroidTracker(max_distance=50)

    first = tracker.update([Detection(0, 0, 20, 20, 0.9, 0, "person")])
    second = tracker.update([Detection(0, 0, 20, 20, 0.9, 5, "bus")])

    assert first[0].track_id != second[0].track_id


def test_centroid_tracker_expires_missing_tracks() -> None:
    tracker = CentroidTracker(max_distance=50, max_missing=0)
    tracker.update([Detection(0, 0, 20, 20, 0.9, 0, "person")])

    tracker.update([])

    assert tracker.tracks == {}
