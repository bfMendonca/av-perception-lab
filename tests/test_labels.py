import re
from pathlib import Path

import pytest
import yaml

from av_perception_lab.core.labels import (
    COCO,
    OBJECTS365,
    SEMANTIC_NAME_PATTERN,
    VOCABULARIES,
    LabelMap,
    group_semantic_labels,
    identify_vocabulary,
    resolve_classes,
    resolve_semantic_classes,
)
from av_perception_lab.core.options import ConfigError

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


NAMES = ("scene", "lanes")


def test_group_semantic_labels():
    labels = ["lanes.lane", "scene.road", "scene.traffic light", "scene.road"]
    assert group_semantic_labels(labels, NAMES) == {
        "scene": ("road", "traffic light"),  # given order, no duplicates
        "lanes": ("lane",),
    }
    # Comma-separated string form (as `classes`), spaces trimmed.
    assert group_semantic_labels(" scene.road, scene.sky ,", NAMES) == {
        "scene": ("road", "sky"),
        "lanes": (),
    }
    # Split at the first dot only: the class may contain dots.
    assert group_semantic_labels(["a.b.c"], ["a"]) == {"a": ("b.c",)}
    assert group_semantic_labels([], NAMES) == {"scene": (), "lanes": ()}
    assert group_semantic_labels("", ()) == {}


@pytest.mark.parametrize("label", ["road", ".sky", "scene.", "scene. "])
def test_group_semantic_labels_needs_a_namespace(label):
    with pytest.raises(ConfigError, match="expected '<segmenter>.<class>'"):
        group_semantic_labels([label], NAMES)


def test_group_semantic_labels_unknown_segmenter():
    with pytest.raises(ConfigError, match=r"unknown segmenter 'sky' \(scene, lanes\)"):
        group_semantic_labels(["sky.sky"], NAMES)
    with pytest.raises(ConfigError, match="no semantic segmenter configured"):
        group_semantic_labels("scene.road", ())


@pytest.mark.parametrize(
    ("name", "valid"),
    [("scene", True), ("lanes_2", True), ("a-b", True), ("0x", True)]
    + [("Scene", False), ("a.b", False), ("a,b", False), ("a b", False)]
    + [("_a", False), ("", False)],
)
def test_semantic_name_pattern(name, valid):
    assert bool(re.fullmatch(SEMANTIC_NAME_PATTERN, name)) is valid


def test_resolve_none_means_all():
    assert resolve_semantic_classes(None, {"scene": ["road"]}) is None


def test_resolve_empty_list_selects_nothing():
    assert resolve_semantic_classes([], {"scene": ["road"]}) == {"scene": frozenset()}


def test_resolve_splits_on_first_dot():
    got = resolve_semantic_classes(["scene.a.b"], {"scene": ["a.b", "c"]})
    assert got == {"scene": frozenset({"a.b"})}


def test_resolve_unknown_class_lists_valid_names():
    with pytest.raises(
        ConfigError,
        match=r"^render\.semantic_classes: scene\.x: "
        r"unknown class; scene classes: road, sky$",
    ):
        resolve_semantic_classes(["scene.x"], {"scene": ["road", "sky"]})


def test_resolve_unknown_map():
    with pytest.raises(
        ConfigError, match=r"unknown semantic map 'nope'; configured: scene"
    ):
        resolve_semantic_classes(["nope.road"], {"scene": ["road"]})


def test_resolve_same_class_in_two_maps():
    got = resolve_semantic_classes(["scene.car"], {"scene": ["car"], "other": ["car"]})
    assert got == {"scene": frozenset({"car"}), "other": frozenset()}
