"""Versioned label maps against the real checkpoints (marked gpu by conftest)."""

from pathlib import Path

from av_perception_lab.core import registry
from av_perception_lab.core.labels import VOCABULARIES

LABELS_DIR = Path(__file__).parents[2] / "configs" / "labels"
LABEL_MAPS = {
    "coco": LABELS_DIR / "coco_driving.yaml",
    "objects365": LABELS_DIR / "objects365_driving.yaml",
}


def test_label_map_matches_the_checkpoint(detector_spec, frame0, load_ctx):
    spec, vocabulary = detector_spec
    spec = {**spec, "labels": str(LABEL_MAPS[vocabulary])}
    detector = registry.build("detector", spec, path="detector")
    detector.load(load_ctx)  # LabelMap.check: raw names and vocabulary
    try:
        out = detector(frame0)
    finally:
        detector.close()
    label_map = detector.label_map
    preset = next(v for v in VOCABULARIES if v.name == vocabulary)
    unmapped = set(preset.presets["driving"]) - set(label_map.mapping)
    assert set(out.label) <= label_map.canonical | unmapped
    assert all(
        label_map.mapping.get(raw, raw) == label
        for raw, label in zip(out.raw_label, out.label, strict=True)
    )
