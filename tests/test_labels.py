from pathlib import Path

import pytest
import yaml

from det_depth_distance.core.labels import (
    COCO,
    OBJECTS365,
    VOCABULARIES,
    LabelMap,
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


LABEL_MAPS = sorted((Path(__file__).parents[1] / "configs" / "labels").glob("*.yaml"))
MAP = {
    "vocabulary": "coco",
    "canonical": {"vehicle": ["car", "truck"], "person": ["person"]},
}


def _map(**changes):
    return LabelMap.from_dict({**MAP, **changes}, name="m", path="labels")


def test_label_map_inverts_canonical():
    label_map = _map()
    assert label_map.mapping == {
        "car": "vehicle",
        "truck": "vehicle",
        "person": "person",
    }
    assert label_map.canonical == {"vehicle", "person"}
    assert label_map.unmapped == "keep" and label_map.vocabulary == "coco"
    assert label_map.describe() == {
        "name": "m",
        "vocabulary": "coco",
        "unmapped": "keep",
        "canonical": {"vehicle": ["car", "truck"], "person": ["person"]},
    }


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        (
            {"canonical": {"vehicle": ["truck"], "truck": ["truck"]}},
            r"^labels: 'truck' mapped to both 'vehicle' and 'truck'",
        ),
        ({"canonical": {}}, r"^labels\.canonical: expected a non-empty mapping"),
        ({"canonical": {"vehicle": "car"}}, r"^labels\.canonical\.vehicle: expected"),
        ({"canonical": {"vehicle": []}}, r"^labels\.canonical\.vehicle: expected"),
        ({"unmaped": "drop"}, r"^labels\.unmaped: unknown key"),
        ({"unmapped": "ignore"}, r"^labels\.unmapped: 'ignore' is not one of"),
        ({"vocabulary": "imagenet"}, r"^labels\.vocabulary: 'imagenet'"),
    ],
)
def test_label_map_schema_errors(changes, match):
    with pytest.raises(ConfigError, match=match):
        _map(**changes)


def test_label_map_apply_keep_and_drop():
    labels, keep = _map().apply(["car", "dog", "person"])
    assert labels == ("vehicle", "dog", "person") and keep.tolist() == [True] * 3
    labels, keep = _map(unmapped="drop").apply(["car", "dog"])
    assert labels == ("vehicle", "dog") and keep.tolist() == [True, False]
    labels, keep = _map().apply([])
    assert labels == () and keep.shape == (0,)


def test_label_map_check():
    _map().check(COCO_LIKE, "coco")
    with pytest.raises(ConfigError, match=r"^m: raw names not known.*'suv'"):
        _map(canonical={"vehicle": ["suv"]}, vocabulary=None).check(COCO_LIKE, "coco")
    with pytest.raises(ConfigError, match="written for 'coco'.*'objects365'"):
        _map().check(O365_LIKE, "objects365")  # "truck" exists there with another sense


def test_label_map_load(tmp_path):
    with pytest.raises(ConfigError, match="label map not found"):
        LabelMap.load(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("canonical: [")
    with pytest.raises(ConfigError, match="invalid YAML"):
        LabelMap.load(bad)
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(MAP))
    assert LabelMap.load(good).name == "good"


@pytest.mark.parametrize("path", LABEL_MAPS, ids=lambda p: p.name)
def test_versioned_label_maps_use_their_driving_preset(path):
    label_map = LabelMap.load(path)
    vocabulary = next(v for v in VOCABULARIES if v.name == label_map.vocabulary)
    assert set(label_map.mapping) <= set(vocabulary.presets["driving"])


def test_versioned_label_maps_share_canonical_names():
    maps = {p.stem: LabelMap.load(p) for p in LABEL_MAPS}
    assert set(maps) == {"coco_driving", "objects365_driving"}
    # Fusion by label needs the COCO names to exist on the Objects365 side.
    assert maps["coco_driving"].canonical <= maps["objects365_driving"].canonical
