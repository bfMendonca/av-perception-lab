import pytest

from det_depth_distance.core.labels import (
    COCO,
    OBJECTS365,
    identify_vocabulary,
    resolve_classes,
)
from det_depth_distance.core.options import ConfigError

COCO_LIKE = [*COCO.presets["driving"], "dog", "toothbrush"]
O365_LIKE = ["sneakers", "chair", *OBJECTS365.presets["driving"]]


def test_resolve_classes():
    assert resolve_classes("all", COCO_LIKE) is None
    assert resolve_classes("driving", COCO_LIKE) == frozenset(COCO.presets["driving"])
    assert resolve_classes(" dog, car ,", COCO_LIKE) == frozenset({"dog", "car"})
    assert resolve_classes(("dog", "car"), COCO_LIKE) == frozenset({"dog", "car"})


def test_resolve_classes_rejects_unknown():
    with pytest.raises(ConfigError, match="tractor"):
        resolve_classes("car,tractor", COCO_LIKE)
    with pytest.raises(ConfigError, match="preset 'driving' undefined"):
        resolve_classes("driving", ["weed", "crop"])  # custom model
    with pytest.raises(ConfigError, match="Car"):
        resolve_classes(["Car"], COCO_LIKE)  # names are case-sensitive


def test_objects365_preset():
    assert len(OBJECTS365.presets["driving"]) == 21
    assert resolve_classes("driving", O365_LIKE) == frozenset(
        OBJECTS365.presets["driving"]
    )


def test_identify_vocabulary():
    assert identify_vocabulary(O365_LIKE) is OBJECTS365  # also contains the COCO preset
    assert identify_vocabulary(COCO_LIKE) is COCO
    assert identify_vocabulary(["weed"]) is None
