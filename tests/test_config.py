from pathlib import Path

import pytest
import yaml

from av_perception_lab.adapters.semantic.twinlitenet import TWINLITENET_CLASSES
from av_perception_lab.config import (
    PIPELINE_DISTANCE_KEYS,
    AgreementConfig,
    CalibrationConfig,
    apply_overrides,
    load_config,
)
from av_perception_lab.core import registry
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.pipeline import Pipeline
from av_perception_lab.distance import DistanceConfig
from av_perception_lab.motion import MotionConfig, build_motion
from av_perception_lab.motion.config import (
    CameraMount,
    MeasurementConfig,
    PredictionConfig,
)
from av_perception_lab.temporal import SmoothingConfig

REPO = Path(__file__).parents[1]
# configs/presets.yaml is the web presets manifest, not a run config
CONFIGS = [
    p for p in sorted((REPO / "configs").glob("*.yaml")) if p.name != "presets.yaml"
]
FUSED = REPO / "configs" / "multi_detector_fused.yaml"
MINIMAL = {"detectors": [{"type": "rfdetr"}], "depth": {"type": "da3"}}


def write(tmp_path, data) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.distance == DistanceConfig() and cfg.exclude_sky
    assert cfg.region == "mask" and cfg.distance.mask_erode == 1
    assert cfg.render.mask_alpha == 0.35
    assert cfg.aggregation == {"type": "concat"}
    assert cfg.agreement == AgreementConfig(enabled=True, iou_thr=0.5, class_aware=True)
    assert cfg.camera.focal_px is None and cfg.camera.estimate.model == "da3-large-1.1"
    assert cfg.render.layout == "overlay" and cfg.output.out_root == Path(
        "_testbed/outputs"
    )
    assert cfg.device == "cuda:0" and cfg.source.video is None
    # task_0006: tracking and smoothing are off by default.
    assert cfg.tracking is None and cfg.smoothing is None
    # task_0007: no calibration by default (focal_px / estimate as before).
    assert cfg.camera.calibration is None
    assert cfg.render.show_smoothed and cfg.render.show_speed
    # task_0009: no motion stage by default; its drawing options are inert.
    assert cfg.motion is None
    assert cfg.render.show_trajectory and cfg.render.show_motion_speed
    assert not cfg.render.bev_inset


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_example_configs_build_without_frameworks(path, monkeypatch):
    monkeypatch.chdir(REPO)  # label map paths are relative to the repository
    cfg = load_config(path)
    for i, spec in enumerate(cfg.detectors):
        assert not registry.build("detector", spec, path=f"detectors[{i}]").is_loaded
    registry.build("depth", cfg.depth, path="depth")
    registry.build("aggregator", cfg.aggregation, path="aggregation")
    for i, spec in enumerate(cfg.refiners):
        assert not registry.build("refiner", spec, path=f"refiners[{i}]").is_loaded
    for i, spec in enumerate(cfg.semantic):
        assert not registry.build("semantic", spec, path=f"semantic[{i}]").is_loaded
    if cfg.tracking is not None:
        assert not registry.build("tracker", cfg.tracking, path="tracking").is_loaded
    if cfg.motion is not None:
        build_motion(cfg.motion, distance_type=cfg.distance.distance_type)
    assert cfg.source.video == Path("_testbed/data/IMG_4021.MOV")


def test_example_configs_cover_the_planned_runs():
    names = {p.name for p in CONFIGS}
    assert names == {
        "rfdetr_da3.yaml",
        "yolo_o365_da3.yaml",
        "multi_detector.yaml",
        "multi_detector_fused.yaml",
        "rfdetr_seg_da3.yaml",
        "yolo_seg_da3.yaml",
        "yolo_o365_seg_da3.yaml",
        "rfdetr_sam2_da3.yaml",
        "yolo_o365_sam2_da3.yaml",
        "rfdetr_semantic_da3.yaml",
        "rfdetr_lanes_scene_da3.yaml",
        "rfdetr_da3_track.yaml",
        "rfdetr_da3_calibrated.yaml",
        "yolo_light_da3.yaml",
        "rfdetr_da3_motion.yaml",
        "rfdetr_sam2_lanes_motion_da3.yaml",
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
    assert PIPELINE_DISTANCE_KEYS == ("exclude_sky", "region", "exclude_semantic")
    cfg = load_config(write(tmp_path, MINIMAL), ["distance.exclude_sky=false"])
    assert cfg.exclude_sky is False and not hasattr(cfg.distance, "exclude_sky")
    with pytest.raises(ConfigError, match="under the distance section"):
        load_config(write(tmp_path, {**MINIMAL, "exclude_sky": False}))


def test_to_dict_restores_yaml_layout(tmp_path):
    cfg = load_config(
        write(tmp_path, MINIMAL),
        ["distance.exclude_sky=false", "distance.region=both"],
    )
    data = cfg.to_dict()
    assert data["distance"]["exclude_sky"] is False and "exclude_sky" not in data
    assert data["distance"]["region"] == "both" and "region" not in data
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


def test_region_and_mask_options(tmp_path):
    cfg = load_config(
        write(tmp_path, MINIMAL),
        ["distance.region=both", "distance.mask_erode=2", "render.mask_alpha=0"],
    )
    assert cfg.region == "both" and not hasattr(cfg.distance, "region")
    assert cfg.distance.mask_erode == 2 and cfg.render.mask_alpha == 0.0
    with pytest.raises(ConfigError, match="under the distance section"):
        load_config(write(tmp_path, {**MINIMAL, "region": "box"}))


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("distance.region=diagonal", r"^region: 'diagonal' is not one of"),
        ("distance.mask_erode=-1", r"^distance: mask_erode must be >= 0"),
        ("distance.mask_erode=1.5", r"^distance\.mask_erode: expected int"),
        ("render.mask_alpha=1.5", r"^render: mask_alpha must be in \[0, 1\]"),
        ("render.mask_alpha=-0.1", r"^render: mask_alpha must be in \[0, 1\]"),
    ],
)
def test_region_and_mask_options_are_validated(tmp_path, override, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, MINIMAL), [override])


@pytest.mark.parametrize(
    "name", ["rfdetr_seg_da3.yaml", "yolo_seg_da3.yaml", "yolo_o365_seg_da3.yaml"]
)
def test_seg_configs_compare_mask_and_box(name, monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = load_config(REPO / "configs" / name)
    assert cfg.region == "both" and cfg.distance.mask_erode == 1
    assert cfg.render.layout == "side-by-side"


# task_0005: refiners, semantic segmenters and namespaced exclusions.
SEMANTIC = REPO / "configs" / "rfdetr_semantic_da3.yaml"
LANES_SCENE = REPO / "configs" / "rfdetr_lanes_scene_da3.yaml"
TWO_MAPS = {
    **MINIMAL,
    "semantic": [
        {"name": "scene", "type": "segformer"},
        {"name": "lanes", "type": "twinlitenet"},
    ],
}


def test_semantic_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert (cfg.refiners, cfg.semantic, cfg.exclude_semantic) == ((), (), ())
    assert cfg.render.semantic_overlay is None and cfg.render.semantic_alpha == 0.4
    explicit = load_config(write(tmp_path, {**MINIMAL, "semantic": [], "refiners": []}))
    assert explicit == cfg
    assert cfg.to_dict()["distance"]["exclude_semantic"] == []


@pytest.mark.parametrize(
    "value",
    [
        "[scene.road,scene.sky]",
        "scene.road, scene.sky",
        "[ scene.road, scene.sky, scene.road]",
    ],
)
def test_exclude_semantic_list_or_string(tmp_path, value):
    cfg = load_config(write(tmp_path, TWO_MAPS), [f"distance.exclude_semantic={value}"])
    assert cfg.exclude_semantic == ("scene.road", "scene.sky")
    data = {**TWO_MAPS, "distance": {"exclude_semantic": "scene.road,scene.sky"}}
    assert load_config(write(tmp_path, data)).exclude_semantic == cfg.exclude_semantic
    assert load_config(write(tmp_path, cfg.to_dict())) == cfg  # round trip


def test_exclude_semantic_canonical_order(tmp_path):
    cfg = load_config(
        write(tmp_path, TWO_MAPS), ["distance.exclude_semantic=[lanes.lane, scene.sky]"]
    )
    assert cfg.exclude_semantic == ("scene.sky", "lanes.lane")  # order of `semantic`


def test_semantic_list_overrides(monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = load_config(
        SEMANTIC,
        ["semantic=[{name: lanes, type: twinlitenet}]", "distance.exclude_semantic=[]"],
    )
    assert [s["name"] for s in cfg.semantic] == ["lanes"]
    single = load_config(SEMANTIC, ["semantic.0.size=[1024,1792]"])
    assert single.semantic[0]["size"] == [1024, 1792]
    built = registry.build("semantic", single.semantic[0], path="semantic[0]")
    assert built.options.size == (1024, 1792)
    combined = load_config(LANES_SCENE, ["semantic.1.size=[1024,1792]"])
    lanes, scene = combined.semantic
    assert scene["size"] == [1024, 1792] and "size" not in lanes
    assert load_config(LANES_SCENE).semantic[0] == lanes  # TwinLiteNet+ untouched
    assert combined.exclude_semantic == ("lanes.lane", "scene.sky", "scene.vegetation")
    assert combined.render.semantic_overlay == ("lanes", "scene")
    with pytest.raises(ConfigError, match="--set semantic.2.size"):
        load_config(LANES_SCENE, ["semantic.2.size=[1,1]"])
    off = load_config(
        LANES_SCENE,
        ["semantic=[]", "distance.exclude_semantic=[]", "render.semantic_overlay=null"],
    )
    assert off.semantic == () and off.exclude_semantic == ()


@pytest.mark.parametrize(
    ("data", "overrides", "match"),
    [
        (
            {**MINIMAL, "semantic": [{"type": "segformer"}]},
            [],
            r"^semantic\[0\]\.name is required",
        ),
        (
            TWO_MAPS,
            ["semantic.1.name=scene"],
            r"^semantic\[1\]\.name: duplicate name 'scene'",
        ),
        (
            TWO_MAPS,
            ["semantic.0.name=scene.x"],
            r"^semantic\[0\]\.name: 'scene\.x' must match",
        ),
        (
            TWO_MAPS,
            ["semantic.0.name=Scene"],
            r"^semantic\[0\]\.name: 'Scene' must match",
        ),
        (
            TWO_MAPS,
            ["semantic.0.name='a,b'"],
            r"^semantic\[0\]\.name: 'a,b' must match",
        ),
        (
            TWO_MAPS,
            ["distance.exclude_semantic=[road]"],
            r"^distance\.exclude_semantic: 'road'",
        ),
        (
            TWO_MAPS,
            ["distance.exclude_semantic=[sky.sky]"],
            r"^distance\.exclude_semantic: .*unknown segmenter 'sky'",
        ),
        (
            MINIMAL,
            ["distance.exclude_semantic=[scene.road]"],
            r"no semantic segmenter configured",
        ),
        (
            TWO_MAPS,
            ["render.semantic_overlay=[x]"],
            r"^render\.semantic_overlay: unknown semantic map 'x'",
        ),
        (
            MINIMAL,
            ["render.semantic_alpha=1.5"],
            r"^render: semantic_alpha must be in \[0, 1\]",
        ),
    ],
)
def test_semantic_config_errors(tmp_path, data, overrides, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, data), overrides)


@pytest.mark.parametrize(
    ("kind", "spec", "match"),
    [
        (
            "refiner",
            {"type": "sam2", "treshold": 1},
            r"^refiners\[0\]\.treshold: unknown option",
        ),
        (
            "semantic",
            {"type": "segformer", "colour": 1},
            r"^semantic\[1\]\.colour: unknown option",
        ),
        (
            "refiner",
            {"type": "sam_hq", "score": "object"},
            r"^refiners\[0\]\.score: sam_hq has no object score",
        ),
        (
            "refiner",
            {
                "type": "ultralytics_sam",
                "weights": "x/mobile_sam.pt",
                "min_object_score": 0.5,
            },
            r"^refiners\[0\]\.min_object_score: ultralytics_sam has no object score",
        ),
        (
            "refiner",
            {"type": "sam", "hq_token_only": True},
            r"^refiners\[0\]\.hq_token_only: unknown option",
        ),
    ],
)
def test_refiner_and_semantic_specs_are_validated(tmp_path, kind, spec, match):
    family = "refiners" if kind == "refiner" else "semantic"
    data = {
        **MINIMAL,
        family: [spec]
        if kind == "refiner"
        else [{"name": "a", "type": "segformer"}, {**spec, "name": "b"}],
    }
    cfg = load_config(write(tmp_path, data))
    specs = cfg.refiners if kind == "refiner" else cfg.semantic
    with pytest.raises(ConfigError, match=match):
        for i, s in enumerate(specs):
            registry.build(kind, s, path=f"{family}[{i}]")


def test_object_score_options_need_the_signal():
    edgetam = registry.build(
        "refiner",
        {"type": "edgetam", "min_object_score": 0.5, "score": "object"},
        path="refiners[0]",
    )
    assert edgetam.options.min_object_score == 0.5 and edgetam.name == "edgetam-hf"


def test_two_unnamed_refiners_of_one_checkpoint_clash():
    specs = [{"type": "sam2"}, {"type": "sam2"}]
    refiners = [
        registry.build("refiner", s, path=f"refiners[{i}]") for i, s in enumerate(specs)
    ]
    detector = registry.build("detector", {"type": "rfdetr"}, path="detectors[0]")
    depth = registry.build("depth", {"type": "da3"}, path="depth")
    aggregator = registry.build("aggregator", {"type": "concat"}, path="aggregation")
    with pytest.raises(
        ConfigError, match=r"^refiners: duplicate name 'sam2\.1-hiera-large'"
    ):
        Pipeline([detector], depth, aggregator, DistanceConfig(), refiners=refiners)


TRACK = REPO / "configs" / "rfdetr_da3_track.yaml"


def test_track_config_builds_the_tracking_pipeline(monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = load_config(TRACK)
    assert cfg.smoothing == SmoothingConfig()  # the YAML spells out the defaults
    assert cfg.render.layout == "side-by-side"
    tracker = registry.build("tracker", cfg.tracking, path="tracking")
    assert tracker.describe()["options"] == {
        "iou_thr": 0.3,
        "max_age_s": 0.5,
        "motion": "constant_velocity",
        "velocity_alpha": 0.5,
        "class_aware": False,
    }
    pipeline = Pipeline(
        [registry.build("detector", cfg.detectors[0], path="detectors[0]")],
        registry.build("depth", cfg.depth, path="depth"),
        registry.build("aggregator", cfg.aggregation, path="aggregation"),
        cfg.distance,
        tracker=tracker,
        smoothing=cfg.smoothing,
    )
    assert pipeline.describe()["pipeline"]["smoothing"]["filter"] == "kalman"
    assert cfg.to_dict()["tracking"] == dict(cfg.tracking)


def test_tracking_and_smoothing_overrides(tmp_path):
    cfg = load_config(
        write(tmp_path, MINIMAL),
        ["tracking={type: iou, iou_thr: 0.5}", "smoothing.filter=ema"],
    )
    assert cfg.tracking == {"type": "iou", "iou_thr": 0.5}
    assert cfg.smoothing == SmoothingConfig(filter="ema")
    assert (
        registry.build("tracker", cfg.tracking, path="tracking").options.iou_thr == 0.5
    )
    off = load_config(
        write(tmp_path, {**MINIMAL, "tracking": {"type": "iou"}}), ["tracking=null"]
    )
    assert off.tracking is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (["smoothing.filter=foo"], r"^smoothing\.filter: 'foo' is not one of"),
        (["smoothing.tau_s=0"], r"^smoothing: tau_s must be > 0"),
        (["tracking=3"], r"^tracking: expected"),
    ],
)
def test_tracking_and_smoothing_errors(tmp_path, overrides, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, MINIMAL), overrides)


def test_tracking_options_are_validated_by_the_registry(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL), ["tracking={type: iou, iou: 0.5}"])
    with pytest.raises(ConfigError, match=r"^tracking\.iou: unknown option"):
        registry.build("tracker", cfg.tracking, path="tracking")
    with pytest.raises(ConfigError, match=r"^tracking\.type: unknown tracker 'sort'"):
        registry.build("tracker", {"type": "sort"}, path="tracking")


# task_0010: light stack for a 4 GB GPU.
LIGHT = REPO / "configs" / "yolo_light_da3.yaml"


def test_light_config_is_the_planned_light_stack(monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = load_config(LIGHT)
    (detector,) = cfg.detectors
    assert detector["type"] == "yolo"
    assert Path(detector["weights"]).name == "yolo26s-seg.pt"
    assert cfg.refiners == ()  # the masks come from the detector
    assert cfg.depth["model"] == "da3metric-large"
    assert "hf_cache" not in cfg.depth  # default: <weights root>/hf
    assert cfg.camera.estimate.model == "da3-large-1.1"
    assert cfg.camera.estimate.frames == 16
    assert cfg.camera.estimate.hf_cache is None
    track = load_config(TRACK)
    assert cfg.tracking == track.tracking
    assert cfg.smoothing == track.smoothing
    lanes, scene = (
        registry.build("semantic", spec, path=f"semantic[{i}]")
        for i, spec in enumerate(cfg.semantic)
    )
    assert (lanes.name, lanes.options.variant) == ("lanes", "small")
    assert scene.name == "scene"
    assert scene.options.model == "nvidia/segformer-b0-finetuned-cityscapes-640-1280"
    assert scene.options.size == (640, 1152)
    assert cfg.region == "both" and cfg.distance.mask_erode == 1
    assert cfg.exclude_semantic == ("lanes.lane", "scene.sky", "scene.vegetation")
    assert "lane" in TWINLITENET_CLASSES  # load_config only checks the prefix
    assert cfg.render.semantic_overlay == ("lanes", "scene")


# task_0009: per-track motion estimation.
MOTION = REPO / "configs" / "rfdetr_da3_motion.yaml"


def test_motion_config_builds_the_motion_pipeline(monkeypatch, capsys):
    monkeypatch.chdir(REPO)
    cfg = load_config(MOTION)
    track = load_config(TRACK)
    assert (cfg.tracking, cfg.smoothing) == (track.tracking, track.smoothing)
    assert cfg.detectors[0]["labels"] == "configs/labels/coco_driving.yaml"
    m = cfg.motion
    assert dict(m.classes) == {"vehicle": "vehicle", "person": "pedestrian"}
    # Without ego-motion every class uses constant velocity (no ctrv warning).
    assert {spec["type"] for spec in m.models.values()} == {"cv"}
    assert m.ego_motion is None and m.default == "cv"
    # The YAML spells out the section defaults (models aside).
    assert (m.measurement, m.prediction, m.camera_mount) == (
        MeasurementConfig(),
        PredictionConfig(),
        CameraMount(),
    )
    assert (m.reset_after_s, m.nis_gate) == (
        MotionConfig().reset_after_s,
        MotionConfig().nis_gate,
    )
    assert cfg.render.bev_inset and cfg.render.layout == "overlay"
    estimator = build_motion(m, distance_type=cfg.distance.distance_type)
    assert capsys.readouterr().err == ""
    pipeline = Pipeline(
        [registry.build("detector", cfg.detectors[0], path="detectors[0]")],
        registry.build("depth", cfg.depth, path="depth"),
        registry.build("aggregator", cfg.aggregation, path="aggregation"),
        cfg.distance,
        tracker=registry.build("tracker", cfg.tracking, path="tracking"),
        motion=estimator,
    )
    described = pipeline.describe()["pipeline"]["motion"]
    assert set(described["models"]) == {"vehicle", "pedestrian", "cv"}
    assert cfg.to_dict()["motion"]["classes"] == dict(m.classes)


def test_motion_overrides(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL), ["motion={}"])
    assert cfg.motion == MotionConfig()
    cfg = load_config(
        write(tmp_path, MINIMAL),
        ["motion.prediction.horizon_s=5", "motion.classes={person: cv}"],
    )
    assert cfg.motion.prediction.horizon_s == 5.0
    assert dict(cfg.motion.classes) == {"person": "cv"}
    off = load_config(write(tmp_path, {**MINIMAL, "motion": {}}), ["motion=null"])
    assert off.motion is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            ["motion.classes={car: truck}"],
            r"^motion: classes\.car: unknown model 'truck'",
        ),
        (["motion.default=fast"], r"^motion: default: unknown model 'fast'"),
        (["motion.models={}"], r"^motion: models: at least one model"),
        (["motion.nis_gate=0"], r"^motion: reset_after_s and nis_gate must be > 0"),
        (["motion.prediction.step_s=4"], r"^motion\.prediction: step_s must be in"),
        (["motion.measurement.pixel_std=0"], r"^motion\.measurement: pixel_std"),
        (["motion.measurement.reference_point=top"], r"reference_point: 'top' is not"),
        (["motion.camera_mount.z_m=1"], r"^motion\.camera_mount\.z_m: unknown option"),
        (["render.bev_inset=true"], r"^render\.bev_inset: needs the motion section"),
        (["render.trajectory_alpha=2"], r"^render: trajectory_alpha must be in"),
        (["render.bev_size=0"], r"^render: bev_size must be in"),
        (["render.bev_range_m=0"], r"^render: bev_range_m and bev_width_m"),
    ],
)
def test_motion_and_render_errors(tmp_path, overrides, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, MINIMAL), overrides)


CALIBRATED = REPO / "configs" / "rfdetr_da3_calibrated.yaml"


def test_calibrated_config():
    cfg = load_config(
        CALIBRATED
    )  # the calibration file itself is checked by build_camera
    assert cfg.camera.calibration == CalibrationConfig(
        path=Path("configs/calibration/iphone14promax_main_1080p30.yaml"),
        cross_check=True,
    )
    assert cfg.camera.focal_px is None and cfg.camera.estimate is not None
    assert cfg.to_dict()["camera"]["calibration"]["check_lens"] == "error"
    over = load_config(CALIBRATED, ["camera.calibration.undistort=true"])
    assert over.camera.calibration.undistort


def test_calibration_overrides(tmp_path):
    cfg = load_config(
        write(tmp_path, MINIMAL), ["camera.calibration={path: c.yaml, alpha: 1}"]
    )
    assert cfg.camera.calibration == CalibrationConfig(path=Path("c.yaml"), alpha=1.0)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            ["camera.calibration={path: c.yaml, alpha: 1.5}"],
            r"^camera\.calibration: alpha must be in \[0, 1\]",
        ),
        (
            ["camera.calibration={path: c.yaml, check_lens: strict}"],
            r"^camera\.calibration\.check_lens: 'strict' is not one of",
        ),
        (
            ["camera.calibration={undistort: true}"],
            r"^camera\.calibration: set path or from_source",
        ),
        (
            ["camera.calibration={path: c.yaml}", "camera.focal_px=1400"],
            r"^camera: calibration and focal_px are exclusive",
        ),
        (
            [
                "camera.calibration={path: c.yaml, cross_check: true}",
                "camera.estimate=null",
            ],
            r"^camera: calibration\.cross_check needs the estimate section",
        ),
    ],
)
def test_calibration_config_errors(tmp_path, overrides, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, MINIMAL), overrides)


MCAP_SRC = {"mcap": "a.mcap", "topic": "/cam"}


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (
            {
                "source": MCAP_SRC,
                "camera": {"calibration": {"from_source": True, "path": "c.yaml"}},
            },
            r"^camera\.calibration: .*exclusive",
        ),
        (
            {"source": MCAP_SRC, "camera": {"calibration": {}}},
            r"^camera\.calibration: .*path or from_source",
        ),
        (
            {
                "source": MCAP_SRC,
                "camera": {"calibration": {"from_source": True, "undistort": True}},
            },
            r"^camera\.calibration: .*rectify",
        ),
        (
            {
                "source": {"video": "v.mp4"},
                "camera": {"calibration": {"from_source": True}},
            },
            r"^camera\.calibration\.from_source: needs source\.mcap",
        ),
        (
            {"source": {"video": "v.mp4"}, "rectify": {"enabled": True}},
            r"^rectify\.enabled: needs source\.mcap",
        ),
        ({"source": MCAP_SRC, "rectify": {"alpha": 1.5}}, r"^rectify: alpha"),
        ({"source": MCAP_SRC, "rectify": {"alpha": -0.1}}, r"^rectify: alpha"),
    ],
)
def test_mcap_config_errors(tmp_path, data, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, {**MINIMAL, **data}))


def test_mcap_source_with_from_source_and_rectify_loads(tmp_path):
    data = {
        **MINIMAL,
        "source": MCAP_SRC,
        "camera": {"calibration": {"from_source": True}},
        "rectify": {"enabled": True, "alpha": 0.5},
    }
    cfg = load_config(write(tmp_path, data))
    assert cfg.camera.calibration.from_source and cfg.camera.calibration.path is None
    assert cfg.rectify.enabled and cfg.rectify.alpha == 0.5
    assert cfg.to_dict()["rectify"] == {"enabled": True, "alpha": 0.5}


def test_rectify_defaults_off(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert not cfg.rectify.enabled and cfg.rectify.alpha == 0.0
