from pathlib import Path

import pytest
import yaml

from det_depth_distance.config import (
    PIPELINE_DISTANCE_KEYS,
    AgreementConfig,
    apply_overrides,
    load_config,
)
from det_depth_distance.core import registry
from det_depth_distance.core.options import ConfigError
from det_depth_distance.distance import DistanceConfig

REPO = Path(__file__).parents[1]
CONFIGS = sorted((REPO / "configs").glob("*.yaml"))
FUSED = REPO / "configs" / "multi_detector_fused.yaml"
MINIMAL = {"detectors": [{"type": "rfdetr"}], "depth": {"type": "da3"}}


def write(tmp_path, data) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.distance == DistanceConfig() and cfg.exclude_sky
    assert cfg.aggregation == {"type": "concat"}
    assert cfg.agreement == AgreementConfig(enabled=True, iou_thr=0.5, class_aware=True)
    assert cfg.camera.focal_px is None and cfg.camera.estimate.model == "da3-large-1.1"
    assert cfg.render.layout == "overlay" and cfg.output.out_root == Path(
        "_testbed/outputs"
    )
    assert cfg.device == "cuda:0" and cfg.source.video is None


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_example_configs_build_without_frameworks(path, monkeypatch):
    monkeypatch.chdir(REPO)  # label map paths are relative to the repository
    cfg = load_config(path)
    for i, spec in enumerate(cfg.detectors):
        assert not registry.build("detector", spec, path=f"detectors[{i}]").is_loaded
    registry.build("depth", cfg.depth, path="depth")
    registry.build("aggregator", cfg.aggregation, path="aggregation")
    assert cfg.source.video == Path("_testbed/data/IMG_4021.MOV")


def test_example_configs_cover_the_planned_runs():
    names = {p.name for p in CONFIGS}
    assert names == {
        "rfdetr_da3.yaml",
        "yolo_o365_da3.yaml",
        "multi_detector.yaml",
        "multi_detector_fused.yaml",
    }
    multi = load_config(CONFIGS[0].with_name("multi_detector.yaml"))
    assert [d["type"] for d in multi.detectors] == ["rfdetr", "yolo"]


def test_overrides(tmp_path):
    data = {**MINIMAL, "detectors": [{"type": "rfdetr"}, {"type": "rfdetr"}]}
    cfg = load_config(
        write(tmp_path, data),
        [
            "source.max_frames=60",
            "detectors.1.threshold=0.3",
            "distance.outlier_k=null",
            "distance.stat=p30",
            "render.layout=side-by-side",
        ],
    )
    assert cfg.source.max_frames == 60
    assert cfg.detectors[1]["threshold"] == 0.3 and "threshold" not in cfg.detectors[0]
    assert cfg.distance.outlier_k is None and cfg.distance.stat == "p30"
    assert cfg.render.layout == "side-by-side"


def test_mapping_override_replaces_the_section():
    data = {"aggregation": {"type": "concat", "extra": 1}, "detectors": [{"type": "a"}]}
    apply_overrides(data, ["aggregation={type: concat}", "detectors=[{type: b}]"])
    assert data == {"aggregation": {"type": "concat"}, "detectors": [{"type": "b"}]}


@pytest.mark.parametrize(
    "override", ["source", "detectors.5.threshold=1", "detectors.x.threshold=1"]
)
def test_bad_overrides(tmp_path, override):
    with pytest.raises(ConfigError, match="--set"):
        load_config(write(tmp_path, MINIMAL), [override])


def test_override_into_scalar_is_an_error():
    with pytest.raises(ConfigError, match="scalar"):
        apply_overrides({"device": "cpu"}, ["device.index=1"])


def test_unknown_key_has_dotted_path(tmp_path):
    with pytest.raises(ConfigError, match=r"^source\.vidoe: unknown option"):
        load_config(write(tmp_path, MINIMAL), ["source.vidoe=x.mov"])


@pytest.mark.parametrize(
    "override",
    [
        "source.stride=0",
        "source.max_frames=-1",
        "source.start=-1",
        "camera.focal_px=-1400",
        "camera.estimate.frames=0",
        "render.max_depth=0",
    ],
)
def test_invalid_numbers_rejected_before_loading(tmp_path, override):
    section = override.split(".")[0]
    with pytest.raises(ConfigError, match=rf"^{section}"):
        load_config(write(tmp_path, MINIMAL), [override])


def test_depth_spec_is_not_validated_by_load_config(tmp_path):
    # The 7th ported check (process_res >= 14) belongs to DA3Depth.Options; see
    # test_registry.py and test_cli.py.
    cfg = load_config(write(tmp_path, MINIMAL), ["depth.process_res=10"])
    assert cfg.depth["process_res"] == 10


def test_positional_video_overrides_source(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL), video=Path("clip.mov"))
    assert cfg.source.video == Path("clip.mov")


def test_pipeline_distance_keys_go_to_run_config(tmp_path):
    assert PIPELINE_DISTANCE_KEYS == ("exclude_sky",)
    cfg = load_config(write(tmp_path, MINIMAL), ["distance.exclude_sky=false"])
    assert cfg.exclude_sky is False and not hasattr(cfg.distance, "exclude_sky")
    with pytest.raises(ConfigError, match="under the distance section"):
        load_config(write(tmp_path, {**MINIMAL, "exclude_sky": False}))


def test_to_dict_restores_yaml_layout(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL), ["distance.exclude_sky=false"])
    data = cfg.to_dict()
    assert data["distance"]["exclude_sky"] is False and "exclude_sky" not in data
    assert data["output"]["out_root"] == "_testbed/outputs"
    assert isinstance(data["detectors"], list)
    assert load_config(write(tmp_path, data)) == cfg  # round trip


def test_aggregation_options_are_validated_by_the_registry(tmp_path):
    data = {**MINIMAL, "aggregation": {"type": "concat"}}
    cfg = load_config(write(tmp_path, data), ["aggregation.iou=0.5"])
    with pytest.raises(ConfigError, match=r"^aggregation\.iou: unknown option"):
        registry.build("aggregator", cfg.aggregation, path="aggregation")


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"depth": {"type": "da3"}}, r"^detectors: missing required option"),
        ({**MINIMAL, "detectors": []}, r"^detectors: at least one"),
        ({**MINIMAL, "render": {"cmap": "nope"}}, r"^render: unknown OpenCV colormap"),
    ],
)
def test_run_config_errors(tmp_path, data, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, data))


def test_missing_or_invalid_file(tmp_path):
    with pytest.raises(ConfigError, match="config not found"):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("detectors: [")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)


def test_fused_config_switches_aggregation(monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = load_config(FUSED)
    assert cfg.aggregation["type"] == "wbf" and cfg.agreement.class_aware
    detectors = [
        registry.build("detector", spec, path=f"detectors[{i}]")
        for i, spec in enumerate(cfg.detectors)
    ]
    assert [d.label_map.name for d in detectors] == [
        "coco_driving",
        "objects365_driving",
    ]
    wbf = registry.build("aggregator", cfg.aggregation, path="aggregation")
    wbf.bind_sources([d.name for d in detectors])  # weights name both detectors
    nms = load_config(FUSED, ["aggregation.type=nms"]).aggregation
    assert registry.build("aggregator", nms, path="aggregation").type_name == "nms"
    concat = load_config(FUSED, ["aggregation={type: concat}"]).aggregation
    assert concat == {"type": "concat"}
    registry.build("aggregator", concat, path="aggregation")


@pytest.mark.parametrize("override", ["agreement.iou_thr=1", "agreement.iou_thr=0"])
def test_agreement_iou_thr_range(tmp_path, override):
    with pytest.raises(ConfigError, match=r"^agreement: iou_thr must be in"):
        load_config(write(tmp_path, MINIMAL), [override])
