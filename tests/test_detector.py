import numpy as np
import pytest

from det_depth_distance.detector import DRIVING_CLASSES, Detections, resolve_classes

COCO_LIKE = [*DRIVING_CLASSES, "dog", "toothbrush"]


def test_resolve_classes():
    assert resolve_classes("all", COCO_LIKE) is None
    assert resolve_classes("driving", COCO_LIKE) == frozenset(DRIVING_CLASSES)
    assert resolve_classes(" dog, car ,", COCO_LIKE) == frozenset({"dog", "car"})


def test_resolve_classes_rejects_unknown():
    with pytest.raises(ValueError, match="tractor"):
        resolve_classes("car,tractor", COCO_LIKE)
    with pytest.raises(ValueError):  # custom model without the driving classes
        resolve_classes("driving", ["weed", "crop"])


def test_detections_select():
    det = Detections(
        np.arange(12, dtype=np.float32).reshape(3, 4),
        np.array([0.9, 0.8, 0.7], dtype=np.float32),
        np.array([3, 1, 3]),
        ("car", "person", "car"),
    )
    kept = det.select(np.array([True, False, True]))
    assert len(kept) == 2
    assert kept.class_name == ("car", "car")
    assert kept.xyxy[1].tolist() == [8, 9, 10, 11]
    assert len(det.select(np.zeros(3, dtype=bool))) == 0
