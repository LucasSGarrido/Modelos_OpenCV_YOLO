import pytest

from object_counter.counting.roi import RegionOfInterest
from object_counter.detection.detector import Detection


def test_roi_filters_detections_by_center() -> None:
    roi = RegionOfInterest(0.25, 0.25, 0.75, 0.75)
    inside = Detection(40, 40, 60, 60, 0.9, 0, "person")
    outside = Detection(80, 80, 95, 95, 0.9, 0, "person")

    assert roi.filter([inside, outside], (100, 100, 3)) == [inside]


def test_roi_pixel_bounds() -> None:
    roi = RegionOfInterest(0.1, 0.2, 0.8, 0.9)

    assert roi.pixel_bounds((200, 100, 3)) == (10, 40, 80, 180)


def test_roi_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        RegionOfInterest(0.8, 0.2, 0.1, 0.9)
