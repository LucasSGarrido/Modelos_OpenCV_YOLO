from object_counter.counting.line_counter import CountLine, LineCounter, crossing_direction
from object_counter.detection.detector import Detection


def detection_with_center(center_y: float, track_id: int = 1) -> Detection:
    return Detection(
        x1=40,
        y1=center_y - 5,
        x2=60,
        y2=center_y + 5,
        confidence=0.9,
        class_id=0,
        label="person",
        track_id=track_id,
    )


def test_crossing_direction() -> None:
    assert crossing_direction(-1, 1) == "positive"
    assert crossing_direction(1, -1) == "negative"
    assert crossing_direction(-2, -1) is None


def test_line_counter_counts_positive_crossing_once_per_track() -> None:
    counter = LineCounter(CountLine(orientation="horizontal", position_ratio=0.5))
    frame_shape = (100, 100, 3)

    assert counter.update([detection_with_center(40)], frame_shape, frame_index=1) == []
    events = counter.update([detection_with_center(60)], frame_shape, frame_index=2)
    repeated = counter.update([detection_with_center(40)], frame_shape, frame_index=3)

    assert len(events) == 1
    assert events[0].direction == "positive"
    assert counter.counts() == {"person": 1}
    assert repeated == []


def test_line_counter_respects_direction_filter() -> None:
    counter = LineCounter(
        CountLine(orientation="horizontal", position_ratio=0.5, direction="negative")
    )
    frame_shape = (100, 100, 3)

    counter.update([detection_with_center(40)], frame_shape, frame_index=1)
    events = counter.update([detection_with_center(60)], frame_shape, frame_index=2)

    assert events == []
    assert counter.total() == 0
