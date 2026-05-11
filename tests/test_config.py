from object_counter.config import COCO_CLASSES, PEOPLE_VEHICLE_CLASSES


def test_coco_classes_include_people_vehicle_defaults() -> None:
    assert set(PEOPLE_VEHICLE_CLASSES).issubset(set(COCO_CLASSES))


def test_coco_classes_has_expected_size() -> None:
    assert len(COCO_CLASSES) == 80
