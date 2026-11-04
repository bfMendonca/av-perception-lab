import dataclasses
from dataclasses import dataclass

import numpy as np
import pytest
from test_components import FakeRefiner, FakeSemantic, FakeTracker

from av_perception_lab.camera import IntrinsicsProvider, IntrinsicsResult
from av_perception_lab.config import AgreementConfig
from av_perception_lab.core.aggregate import Concat, Nms
from av_perception_lab.core.components import (
    DepthEstimator,
    Detector,
    DetectorOptions,
)
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.pipeline import Pipeline
from av_perception_lab.core.tracking import IouTracker
from av_perception_lab.core.types import (
    CameraIntrinsics,
    DepthResult,
    Frame,
    Instances,
    MapGeometry,
    Provenance,
)
from av_perception_lab.distance import (
    DistanceConfig,
    box_depth_stats,
    map_eligibility,
)
from av_perception_lab.io.source import FrameSource
from av_perception_lab.motion import MotionConfig, MotionEstimator, build_motion
from av_perception_lab.temporal import SmoothingConfig

FRAME = Frame(0, 0.0, np.zeros((100, 100, 3), np.uint8))
GEOMETRY = MapGeometry(100, 100, 100, 100, 1.0, 1.0)


class FakeDetector(Detector):
    type_name = "fake"
    Options = DetectorOptions

    def __init__(self, instances, log, *, name):
        super().__init__(DetectorOptions(classes="all"), name=name)
        self.instances, self.log = instances, log

    def _load(self, ctx):
        self.log.append(f"load {self.name}")
        self.ctx = ctx

    def _close(self):
        self.log.append(f"close {self.name}")

    @property
    def class_names(self):
        return ["car", "person"]

    def _predict(self, rgb):
        return self.instances


class FakeDepth(DepthEstimator):
    """Object at 10 m in columns 20..80 over a 40 m background; columns 0..20 are sky."""

    type_name = "fake-depth"

    @dataclass(frozen=True, slots=True)
    class Options:
        pass

    def __init__(self, log, *, needs_intrinsics=True):
        super().__init__(self.Options(), name="depth")
        self.log, self._needs = log, needs_intrinsics

    @property
    def needs_intrinsics(self):
        return self._needs

    def _load(self, ctx):
        self.log.append("load depth")
        self.ctx = ctx
        self._geometry = GEOMETRY

    def _close(self):
        self.log.append("close depth")

    def _predict(self, rgb):
        depth = np.full((100, 100), 40.0, dtype=np.float32)
        depth[:, 20:80] = 10.0
        sky = np.zeros((100, 100), dtype=bool)
        sky[:, :20] = True  # the "person" box is all sky
        k = CameraIntrinsics(50.0, 50.0, 50.0, 50.0)
        return DepthResult(depth, k, sky, GEOMETRY)


class SpyProvider(IntrinsicsProvider):
    def __init__(self, log):
        self.log = log

    def resolve(self, source, *, device):
        self.log.append("resolve")
        return IntrinsicsResult(CameraIntrinsics.centered(70.0, 100, 100), "spy")


class SpyConcat(Concat):
    def __init__(self):
        super().__init__(Concat.Options())
        self.bound = []

    def bind_sources(self, names):
        self.bound.append(list(names))


class Source(FrameSource):
    frame_size = (100, 100)
    name = "src"

    def __iter__(self):
        return iter([FRAME])

    def sample(self, n):
        return []

    def describe(self):
        return {}


@pytest.fixture
def scene(make_instances):
    """Two boxes: a car on the object, a person entirely over sky."""
    log = []  # shared record of setup/close events, in order
    instances = make_instances(
        [[0, 0, 100, 100], [0, 0, 20, 100]],
        labels=["car", "person"],
        score=[0.9, 0.6],
        source="det",
    )
    return log, FakeDetector(instances, log, name="det")


def make_pipeline(
    detectors,
    depth,
    *,
    exclude_sky=True,
    provider=None,
    aggregator=None,
    agreement=None,
    region="mask",
    **stages,
):
    return Pipeline(
        detectors,
        depth,
        aggregator or SpyConcat(),
        DistanceConfig(box_shrink=0.0),
        exclude_sky=exclude_sky,
        intrinsics=provider,
        device="cpu",
        agreement=agreement,
        region=region,
        **stages,  # refiners, semantic, exclude_semantic (0005); tracker, smoothing (0006)
    )


@pytest.mark.parametrize(("exclude_sky", "person_m"), [(True, None), (False, 40.0)])
def test_process_excludes_sky(scene, exclude_sky, person_m):
    log, det = scene
    pipeline = make_pipeline(
        [det], FakeDepth(log, needs_intrinsics=False), exclude_sky=exclude_sky
    )
    pipeline.setup(Source())
    result = pipeline.process(FRAME)
    assert len(result.stats) == 2
    assert result.stats[0].distance == pytest.approx(10.0)
    assert result.stats[1].distance == (pytest.approx(person_m) if person_m else None)
    assert list(result.timings_ms) == ["det", "det.det", "aggregate", "depth", "dist"]
    assert result.frame is FRAME


def test_two_detectors_concat_with_source(scene, make_instances):
    log, det = scene
    other = FakeDetector(
        make_instances([[10, 10, 30, 30]], labels=["car"], source="yolo"),
        log,
        name="yolo",
    )
    pipeline = make_pipeline([det, other], FakeDepth(log, needs_intrinsics=False))
    assert pipeline.aggregator.bound == [["det", "yolo"]]  # once, in YAML order
    pipeline.setup(Source())
    result = pipeline.process(FRAME)
    assert result.instances.source == ("det", "det", "yolo")
    t = result.timings_ms
    assert {"det.det", "det.yolo", "aggregate"} <= set(t)
    assert t["det"] == pytest.approx(t["det.det"] + t["det.yolo"])


def test_intrinsics_resolved_before_any_load(scene):
    log, det = scene
    depth = FakeDepth(log)
    info = make_pipeline([det], depth, provider=SpyProvider(log)).setup(Source())
    assert log == ["resolve", "load det", "load depth"]
    assert info.focal_frame_px == 70.0 and info.focal_source == "spy"
    assert depth.ctx.intrinsics == CameraIntrinsics(70.0, 70.0, 50.0, 50.0)
    assert det.ctx.frame_size == (100, 100) and det.ctx.device == "cpu"


class DetailedProvider(SpyProvider):
    def resolve(self, source, *, device):
        result = super().resolve(source, device=device)
        return dataclasses.replace(result, details={"estimated_focal_frame_px": 75.0})


def test_intrinsics_details_reach_setup_info(scene):
    log, det = scene
    info = make_pipeline([det], FakeDepth(log), provider=SpyProvider(log)).setup(
        Source()
    )
    assert info.intrinsics_details == {}  # providers without details (task_0006 path)
    detailed = make_pipeline([det], FakeDepth(log), provider=DetailedProvider(log))
    assert detailed.resolve_intrinsics(Source()).intrinsics_details == {
        "estimated_focal_frame_px": 75.0
    }


def test_depth_without_intrinsics_need_skips_the_provider(scene):
    log, det = scene
    info = make_pipeline(
        [det], FakeDepth(log, needs_intrinsics=False), provider=SpyProvider(log)
    ).setup(Source())
    assert "resolve" not in log
    assert info.focal_frame_px is None
    assert info.focal_source == "predicted per frame by the depth model"


def test_missing_provider_is_a_config_error(scene):
    log, det = scene
    with pytest.raises(
        ConfigError, match="set camera.calibration, camera.focal_px or camera.estimate"
    ):
        make_pipeline([det], FakeDepth(log)).setup(Source())
    assert log == []  # nothing loaded


def test_close_closes_everything(scene):
    log, det = scene
    with make_pipeline([det], FakeDepth(log, needs_intrinsics=False)) as pipeline:
        pipeline.setup(Source())
    assert log[-2:] == ["close det", "close depth"]


def test_detector_names_must_be_unique(scene):
    log, det = scene
    with pytest.raises(ConfigError, match="duplicate name 'det'"):
        make_pipeline([det, det], FakeDepth(log))
    with pytest.raises(ConfigError, match="at least one"):
        make_pipeline([], FakeDepth(log))


def test_single_detector_concat_without_labels_is_unchanged(scene):
    # Regression invariant of task_0003: one detector, concat and no label map
    # (with the default agreement section on) behave exactly as in task_0002.
    log, det = scene
    pipeline = make_pipeline(
        [det],
        FakeDepth(log, needs_intrinsics=False),
        aggregator=Concat(Concat.Options()),
        agreement=AgreementConfig(),
    )
    pipeline.setup(Source())
    result = pipeline.process(FRAME)
    inst, raw = result.instances, det.instances
    for name in ("xyxy", "score", "class_id"):
        np.testing.assert_array_equal(getattr(inst, name), getattr(raw, name))
    assert (inst.label, inst.raw_label, inst.source) == (
        raw.label,
        raw.raw_label,
        raw.source,
    )
    assert inst.masks is None and inst.sources is None and result.agreement is None
    assert inst.provenance() == (
        (Provenance("det", "car", pytest.approx(0.9)),),
        (Provenance("det", "person", pytest.approx(0.6)),),
    )
    assert list(result.timings_ms) == ["det", "det.det", "aggregate", "depth", "dist"]
    desc = pipeline.describe()
    assert desc["components"]["aggregator"] == {"type": "concat", "options": {}}
    assert desc["components"]["detectors"][0]["labels"] is None
    assert desc["pipeline"] == {
        "exclude_sky": True,
        "agreement": None,
        "region": "mask",
        "mask_erode": 1,
        "exclude_semantic": {},
        "smoothing": None,
        "motion": None,
    }
    assert (desc["components"]["refiners"], desc["components"]["semantic"]) == ([], [])
    # task_0006: no tracker nor smoothing -> no track ids, no temporal estimates.
    assert desc["components"]["tracker"] is None
    assert inst.track_id is None and result.temporal is None
    # task_0005: no refiner nor segmenter -> no mask fields, no semantic map.
    assert inst.mask_source is None and inst.mask_score is None
    assert result.semantic == {}
    # task_0004: box-only instances keep the box statistics under the default policy.
    assert all(not s.used_mask for s in result.stats) and result.box_stats is None


def test_describe_separates_components_and_policy(scene):
    log, det = scene
    pipeline = make_pipeline(
        [det], FakeDepth(log, needs_intrinsics=False), exclude_sky=False
    )
    pipeline.setup(Source())
    desc = pipeline.describe()
    assert set(desc) == {"components", "pipeline"}
    assert set(desc["components"]) == {
        "detectors",
        "depth",
        "aggregator",
        "refiners",
        "semantic",
        "tracker",
    }
    assert desc["components"]["detectors"][0]["name"] == "det"
    assert desc["pipeline"] == {
        "exclude_sky": False,
        "agreement": None,
        "region": "mask",
        "mask_erode": 1,
        "exclude_semantic": {},
        "smoothing": None,
        "motion": None,
    }


@pytest.fixture
def two_detectors(scene, make_instances):
    """`det` (car + person) and `yolo` (the same car, slightly shifted)."""
    log, det = scene
    yolo = FakeDetector(
        make_instances([[2, 0, 100, 100]], labels=["car"], source="yolo"),
        log,
        name="yolo",
    )
    return log, det, yolo


def test_agreement_with_two_detectors(two_detectors):
    log, det, yolo = two_detectors
    pipeline = make_pipeline(
        [det, yolo],
        FakeDepth(log, needs_intrinsics=False),
        agreement=AgreementConfig(iou_thr=0.5),
    )
    pipeline.setup(Source())
    result = pipeline.process(FRAME)
    (pair,) = result.agreement.pairs
    assert (pair.a, pair.b, pair.matched, pair.only_a, pair.only_b) == (
        "det",
        "yolo",
        1,
        1,
        0,
    )
    assert list(result.timings_ms)[3:5] == ["agree", "aggregate"]
    desc = pipeline.describe()
    assert desc["pipeline"]["agreement"] == {
        "enabled": True,
        "iou_thr": 0.5,
        "class_aware": True,
    }
    assert "agreement" not in desc["components"]


@pytest.mark.parametrize("agreement", [None, AgreementConfig(enabled=False)])
def test_agreement_off(two_detectors, agreement):
    log, det, yolo = two_detectors
    pipeline = make_pipeline(
        [det, yolo], FakeDepth(log, needs_intrinsics=False), agreement=agreement
    )
    pipeline.setup(Source())
    result = pipeline.process(FRAME)
    assert result.agreement is None and "agree" not in result.timings_ms


def test_agreement_needs_two_detectors(scene):
    log, det = scene
    pipeline = make_pipeline(
        [det], FakeDepth(log, needs_intrinsics=False), agreement=AgreementConfig()
    )
    pipeline.setup(Source())
    assert pipeline.process(FRAME).agreement is None
    assert pipeline.describe()["pipeline"]["agreement"] is None


def test_fusion_through_the_pipeline(two_detectors):
    log, det, yolo = two_detectors
    pipeline = make_pipeline(
        [det, yolo],
        FakeDepth(log, needs_intrinsics=False),
        aggregator=Nms(Nms.Options(weights={"yolo": 0.5})),
    )
    pipeline.setup(Source())
    result = pipeline.process(FRAME)
    assert result.instances.label == ("car", "person")
    assert [m.source for m in result.instances.sources[0]] == ["det", "yolo"]
    assert len(result.stats) == 2  # distance of the fused instances
    assert pipeline.describe()["components"]["aggregator"]["weights"] == {
        "det": 1.0,
        "yolo": 0.5,
    }


def test_unknown_weight_is_a_config_error(scene):
    log, det = scene
    with pytest.raises(ConfigError, match="aggregation.weights: unknown detector"):
        make_pipeline(
            [det],
            FakeDepth(log),
            aggregator=Nms(Nms.Options(weights={"yolo": 2.0})),
        )


class OtherVocabulary(FakeDetector):
    @property
    def class_names(self):
        return ["suv", "pedestrian"]


def test_warns_when_fused_detectors_share_no_label(scene, capsys):
    log, det = scene
    other = OtherVocabulary(Instances.empty(), log, name="o365")
    for class_agnostic, warned in [(False, True), (True, False)]:
        pipeline = make_pipeline(
            [det, other],
            FakeDepth(log, needs_intrinsics=False),
            aggregator=Nms(Nms.Options(class_agnostic=class_agnostic)),
        )
        pipeline.setup(Source())
        err = capsys.readouterr().err
        assert ("[warn] nms fuses by label, but det and o365" in err) is warned
    same_vocabulary = FakeDetector(Instances.empty(), log, name="b")
    make_pipeline(
        [det, same_vocabulary],
        FakeDepth(log, needs_intrinsics=False),
        aggregator=Nms(Nms.Options()),
    ).setup(Source())
    assert capsys.readouterr().err == ""


def _background_mask() -> np.ndarray:
    """(1, 100, 100) frame mask over columns 85..100, the 40 m background of FakeDepth."""
    masks = np.zeros((1, 100, 100), dtype=bool)
    masks[0, :, 85:] = True
    return masks


@pytest.fixture
def masked_scene(make_instances):
    """`seg` sees the full-frame box with a mask over the background (box 10 m,
    mask 40 m) and the person box without a mask."""
    log = []
    masks = np.concatenate([_background_mask(), np.zeros((1, 100, 100), bool)])
    instances = make_instances(
        [[0, 0, 100, 100], [0, 0, 20, 100]],
        labels=["car", "person"],
        score=[0.9, 0.6],
        source="seg",
        masks=masks,
    )
    return log, FakeDetector(instances, log, name="seg")


@pytest.mark.parametrize(
    ("region", "car_m", "used_mask"),
    [("mask", 40.0, True), ("box", 10.0, False), ("both", 40.0, True)],
)
def test_region_policy(masked_scene, region, car_m, used_mask):
    log, det = masked_scene
    pipeline = make_pipeline(
        [det], FakeDepth(log, needs_intrinsics=False), exclude_sky=False, region=region
    )
    pipeline.setup(Source())
    result = pipeline.process(FRAME)
    car, person = result.stats
    assert car.distance == pytest.approx(car_m) and car.used_mask is used_mask
    assert not person.used_mask  # all-False mask: box statistics
    if region == "both":
        assert result.box_stats[0].distance == pytest.approx(10.0)
        assert result.box_stats[1] is None
    else:
        assert result.box_stats is None
    assert list(result.timings_ms)[-1] == "dist"


@pytest.mark.parametrize("region", ["mask", "box", "both"])
def test_box_only_detector_is_unchanged_by_the_region_policy(scene, region):
    log, det = scene
    pipeline = make_pipeline(
        [det], FakeDepth(log, needs_intrinsics=False), region=region
    )
    pipeline.setup(Source())
    result = pipeline.process(FRAME)
    depth = result.depth
    expected = tuple(
        box_depth_stats(
            depth.depth_m,
            box,
            depth.geometry,
            pipeline.distance,
            intrinsics=depth.intrinsics,
            mask=~depth.sky,
        )
        for box in result.instances.xyxy
    )
    assert result.stats == expected
    assert result.box_stats == ((None, None) if region == "both" else None)


def test_detector_without_masks_falls_back_to_the_box(make_instances):
    log = []
    seg = FakeDetector(
        make_instances([[0, 0, 100, 100]], source="seg", masks=_background_mask()),
        log,
        name="seg",
    )
    det = FakeDetector(
        make_instances([[0, 0, 100, 100]], source="det"), log, name="det"
    )
    pipeline = make_pipeline([seg, det], FakeDepth(log, needs_intrinsics=False))
    pipeline.setup(Source())
    result = pipeline.process(FRAME)
    assert result.instances.masks.shape == (2, 100, 100)  # concat filled zeros
    assert not result.instances.masks[1].any()
    assert [s.used_mask for s in result.stats] == [True, False]
    assert result.stats[0].distance == pytest.approx(40.0)
    assert result.stats[1].distance == pytest.approx(10.0)


def test_region_policy_is_validated_and_described(scene):
    log, det = scene
    with pytest.raises(
        ConfigError, match=r"^distance\.region: 'diagonal' is not one of"
    ):
        make_pipeline([det], FakeDepth(log), region="diagonal")
    desc = make_pipeline([det], FakeDepth(log), region="both").describe()
    assert (desc["pipeline"]["region"], desc["pipeline"]["mask_erode"]) == ("both", 1)
    assert "region" not in desc["components"]


# task_0005: mask refiners and semantic segmenters.
# Semantic doubles on the 100x100 FRAME: "scene" (road | building) with road on
# columns 20..70, the left part of FakeDepth's 10 m object; "lanes" (background |
# lane) with lane on the top third. Id 1 means building in one and lane in the other.
SCENE = ("road", "building")
LANES = ("background", "lane")


def _scene_labels() -> np.ndarray:
    labels = np.ones((100, 100), np.int32)
    labels[:, 20:70] = 0
    return labels


def _lane_labels() -> np.ndarray:
    labels = np.zeros((100, 100), np.int32)
    labels[:33] = 1
    return labels


def _segmenters(log=None):
    return [
        FakeSemantic(_scene_labels(), SCENE, name="scene", log=log),
        FakeSemantic(_lane_labels(), LANES, name="lanes", log=log),
    ]


class LoggingDepth(FakeDepth):
    def _predict(self, rgb):
        self.log.append("depth")
        return super()._predict(rgb)


class LoggingConcat(SpyConcat):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def __call__(self, parts):
        self.log.append("aggregate")
        return super().__call__(parts)


class LoggingRefiner(FakeRefiner):
    def __init__(self, log, **kwargs):
        super().__init__(**kwargs)
        self.log = log

    def _refine(self, rgb, instances):
        self.log.append(f"refine {self.name}")
        return super()._refine(rgb, instances)

    def _close(self):
        self.log.append(f"close {self.name}")


def test_stage_order_timings_and_close(scene):
    log, det = scene
    pipeline = make_pipeline(
        [det],
        LoggingDepth(log, needs_intrinsics=False),
        aggregator=LoggingConcat(log),
        refiners=[LoggingRefiner(log)],
        semantic=_segmenters(log),
    )
    pipeline.setup(Source())
    log.clear()
    result = pipeline.process(FRAME)
    assert log == [
        "aggregate",
        "refine sam",
        "depth",
        "semantic scene",
        "semantic lanes",
    ]
    assert list(result.semantic) == ["scene", "lanes"]
    t = result.timings_ms
    assert list(t) == [
        "det",
        "det.det",
        "aggregate",
        "refine",
        "refine.sam",
        "depth",
        "semantic",
        "semantic.scene",
        "semantic.lanes",
        "dist",
    ]
    assert t["semantic"] == pytest.approx(t["semantic.scene"] + t["semantic.lanes"])
    assert t["refine"] == pytest.approx(t["refine.sam"])
    pipeline.close()
    assert log[-5:] == [
        "close det",
        "close depth",
        "close sam",
        "close scene",
        "close lanes",
    ]


def test_box_only_detector_with_a_refiner_uses_masks(scene):
    log, det = scene
    pipeline = make_pipeline(
        [det], FakeDepth(log, needs_intrinsics=False), refiners=[FakeRefiner()]
    )
    pipeline.setup(Source())
    result = pipeline.process(FRAME)
    assert all(s.used_mask for s in result.stats)
    assert result.instances.mask_sources() == ("sam", "sam")


def _expected_stats(result, pipeline, mask):
    depth = result.depth
    return tuple(
        box_depth_stats(
            depth.depth_m,
            box,
            depth.geometry,
            pipeline.distance,
            intrinsics=depth.intrinsics,
            mask=mask,
        )
        for box in result.instances.xyxy
    )


def test_semantic_exclusion_changes_the_distance(scene):
    log, det = scene
    plain = make_pipeline([det], FakeDepth(log, needs_intrinsics=False))
    plain.setup(Source())
    before = plain.process(FRAME).stats[0]
    pipeline = make_pipeline(
        [det],
        FakeDepth(log, needs_intrinsics=False),
        semantic=_segmenters()[:1],
        exclude_semantic=("scene.road",),
    )
    pipeline.setup(Source())
    after = pipeline.process(FRAME).stats[0]
    # The car's eligible pixels lose columns 20..70 (road): 10 m object -> 40 m.
    assert after.n_valid < before.n_valid
    assert before.distance == pytest.approx(10.0)
    assert after.distance == pytest.approx(40.0)


def test_two_segmenters_exclude_the_union(scene):
    log, det = scene
    pipeline = make_pipeline(
        [det],
        FakeDepth(log, needs_intrinsics=False),
        semantic=_segmenters(),
        exclude_semantic="scene.road, lanes.lane",
    )
    pipeline.setup(Source())
    result = pipeline.process(FRAME)
    sky = result.depth.sky
    union = map_eligibility(
        GEOMETRY,
        sky=sky,
        exclusions=[(_scene_labels(), [0]), (_lane_labels(), [1])],
    )
    assert result.stats == _expected_stats(result, pipeline, union)
    road_only = map_eligibility(GEOMETRY, sky=sky, exclusions=[(_scene_labels(), [0])])
    assert union.sum() < road_only.sum()
    # scene.road alone does not exclude id 1 of the lanes map.
    only_road = make_pipeline(
        [det],
        FakeDepth(log, needs_intrinsics=False),
        semantic=_segmenters(),
        exclude_semantic=["scene.road"],
    )
    only_road.setup(Source())
    out = only_road.process(FRAME)
    assert out.stats == _expected_stats(out, only_road, road_only)
    assert {n: list(m.class_names) for n, m in out.semantic.items()} == {
        "scene": list(SCENE),
        "lanes": list(LANES),
    }
    desc = pipeline.describe()
    assert desc["pipeline"]["exclude_semantic"] == {
        "scene": ["road"],
        "lanes": ["lane"],
    }
    assert [s["name"] for s in desc["components"]["semantic"]] == ["scene", "lanes"]
    assert only_road.describe()["pipeline"]["exclude_semantic"] == {"scene": ["road"]}


@pytest.mark.parametrize("semantic", [None, ()])
def test_without_refiners_and_segmenters_nothing_changes(scene, semantic):
    # Regression invariant: no refiner and no segmenter (absent or explicitly
    # empty) gives the task_0004 result field by field.
    log, det = scene
    stages = {} if semantic is None else {"semantic": semantic, "refiners": ()}
    pipeline = make_pipeline([det], FakeDepth(log, needs_intrinsics=False), **stages)
    pipeline.setup(Source())
    result = pipeline.process(FRAME)
    assert result.semantic == {}
    assert not any(k.startswith(("semantic", "refine")) for k in result.timings_ms)
    assert result.stats == _expected_stats(result, pipeline, ~result.depth.sky)
    assert result.instances.mask_source is None


def test_unknown_semantic_class_fails_at_load(scene):
    log, det = scene
    pipeline = make_pipeline(
        [det],
        FakeDepth(log, needs_intrinsics=False),
        semantic=_segmenters(),
        exclude_semantic=["scene.roda"],
    )
    with pytest.raises(
        ConfigError,
        match=r"^distance\.exclude_semantic: scene\.roda: unknown class; "
        r"scene classes: road, building",
    ):
        pipeline.setup(Source())


@pytest.mark.parametrize(
    ("semantic", "exclude", "match"),
    [
        ((), ["scene.road"], r"^distance\.exclude_semantic: .*no semantic segmenter"),
        (True, ["sky.sky"], r"^distance\.exclude_semantic: .*unknown segmenter 'sky'"),
        (True, ["road"], r"^distance\.exclude_semantic: .*'<segmenter>\.<class>'"),
    ],
)
def test_exclude_semantic_namespace_fails_at_init(scene, semantic, exclude, match):
    log, det = scene
    segmenters = _segmenters() if semantic is True else semantic
    with pytest.raises(ConfigError, match=match):
        make_pipeline(
            [det], FakeDepth(log), semantic=segmenters, exclude_semantic=exclude
        )


def test_component_names_unique_and_semantic_names_valid(scene):
    log, det = scene
    twice = [FakeSemantic(_scene_labels(), SCENE, name="scene")] * 2
    with pytest.raises(ConfigError, match=r"^semantic: duplicate name 'scene'"):
        make_pipeline([det], FakeDepth(log), semantic=twice)
    with pytest.raises(ConfigError, match=r"^refiners: duplicate name 'det'"):
        make_pipeline([det], FakeDepth(log), refiners=[FakeRefiner(name="det")])
    bad = [FakeSemantic(_scene_labels(), SCENE, name="Scene.1")]
    with pytest.raises(ConfigError, match=r"^semantic: name 'Scene\.1' must match"):
        make_pipeline([det], FakeDepth(log), semantic=bad)


def test_refiners_are_described(scene):
    log, det = scene
    pipeline = make_pipeline(
        [det], FakeDepth(log, needs_intrinsics=False), refiners=[FakeRefiner()]
    )
    (refiner,) = pipeline.describe()["components"]["refiners"]
    assert (refiner["name"], refiner["options"]) == ("sam", {"targets": "missing"})


def test_fused_mask_source_end_to_end(make_instances):
    # F2: a box-only detector wins the NMS cluster, the mask comes from `seg`.
    log = []
    box = FakeDetector(
        make_instances([[0, 0, 100, 100]], source="box", score=[0.9]), log, name="box"
    )
    masks = np.zeros((1, 100, 100), bool)
    masks[0, :, 85:] = True
    seg = FakeDetector(
        make_instances([[2, 0, 100, 100]], source="seg", score=[0.6], masks=masks),
        log,
        name="seg",
    )
    pipeline = make_pipeline(
        [box, seg],
        FakeDepth(log, needs_intrinsics=False),
        aggregator=Nms(Nms.Options()),
    )
    pipeline.setup(Source())
    inst = pipeline.process(FRAME).instances
    assert inst.source == ("box",) and inst.mask_sources() == ("seg",)


# task_0006: tracker and per-track smoothing.


def test_tracker_runs_after_refiners_and_before_depth(scene):
    log, det = scene
    tracker = FakeTracker(log=log)
    pipeline = make_pipeline(
        [det],
        LoggingDepth(log, needs_intrinsics=False),
        aggregator=LoggingConcat(log),
        refiners=[LoggingRefiner(log)],
        tracker=tracker,
        smoothing=SmoothingConfig(),
    )
    pipeline.setup(Source())
    assert tracker.is_loaded and tracker.resets == 1
    log.clear()
    result = pipeline.process(FRAME)
    assert log == ["aggregate", "refine sam", "track", "depth"]
    assert list(result.timings_ms) == [
        "det",
        "det.det",
        "aggregate",
        "refine",
        "refine.sam",
        "track",
        "depth",
        "dist",
        "smooth",
    ]
    assert result.instances.track_id.tolist() == [1, 2]
    pipeline.close()
    assert log[-1] == "close tracker" and not tracker.is_loaded


def test_tracker_without_smoothing(scene):
    log, det = scene
    pipeline = make_pipeline(
        [det], FakeDepth(log, needs_intrinsics=False), tracker=FakeTracker()
    )
    pipeline.setup(Source())
    result = pipeline.process(FRAME)
    assert result.instances.track_id.tolist() == [1, 2]
    assert result.temporal is None and "smooth" not in result.timings_ms
    assert pipeline.describe()["pipeline"]["smoothing"] is None


def test_smoothing_needs_a_tracker(scene):
    log, det = scene
    with pytest.raises(ConfigError, match=r"^smoothing: needs tracking"):
        make_pipeline([det], FakeDepth(log), smoothing=SmoothingConfig())


class RecedingDepth(FakeDepth):
    """FakeDepth whose 10 m object moves 0.1 m away on every call."""

    def _load(self, ctx):
        super()._load(ctx)
        self.offset = 0.0

    def _predict(self, rgb):
        result = super()._predict(rgb)
        depth = result.depth_m.copy()
        depth[:, 20:80] += self.offset
        self.offset += 0.1
        return DepthResult(depth, result.intrinsics, result.sky, result.geometry)


def test_smoothing_follows_each_track(scene):
    # 0.1 m per frame at 30 fps = moving away at 3 m/s; the person box is all
    # sky (no distance), so its track never gets an estimate.
    log, det = scene
    pipeline = make_pipeline(
        [det],
        RecedingDepth(log, needs_intrinsics=False),
        tracker=IouTracker(IouTracker.Options()),
        smoothing=SmoothingConfig(),
    )
    pipeline.setup(Source())
    results = [pipeline.process(Frame(k, k / 30, FRAME.rgb)) for k in range(60)]
    assert {tuple(r.instances.track_id.tolist()) for r in results} == {(1, 2)}
    last = results[-1]
    car, person = last.temporal
    assert person is None and last.stats[1].distance is None
    assert last.stats[0].distance == pytest.approx(10.0 + 5.9)
    assert car.distance == pytest.approx(last.stats[0].distance, abs=0.05)
    assert car.rel_speed == pytest.approx(3.0, abs=0.1)  # > 0: moving away
    assert car.n_updates == 60
    desc = pipeline.describe()
    assert desc["components"]["tracker"]["type"] == "iou"
    assert desc["pipeline"]["smoothing"] == {
        "filter": "kalman",
        "accel_std": 2.0,
        "std_scale": 0.5,
        "rel_floor": 0.01,
        "init_speed_std": 5.0,
        "tau_s": 0.1,
        "window_s": 0.15,
        "reset_after_s": 1.0,
    }
    # A new run (load after close) starts with fresh tracks and filters.
    pipeline.close()
    pipeline.setup(Source())
    again = pipeline.process(Frame(0, 0.0, FRAME.rgb))
    assert again.instances.track_id.tolist() == [1, 2]
    assert again.temporal[0].n_updates == 1


def test_without_tracker_and_smoothing_the_result_is_unchanged(scene):
    # Regression invariant: the task_0005 FrameResult field by field.
    log, det = scene
    pipeline = make_pipeline([det], FakeDepth(log, needs_intrinsics=False))
    pipeline.setup(Source())
    result = pipeline.process(FRAME)
    inst, raw = result.instances, det.instances
    for f in dataclasses.fields(Instances):
        a, b = getattr(inst, f.name), getattr(raw, f.name)
        assert (a is None and b is None) or np.array_equal(a, b), f.name
    assert result.instances.track_id is None and result.temporal is None
    assert list(result.timings_ms) == ["det", "det.det", "aggregate", "depth", "dist"]
    assert result.stats == _expected_stats(result, pipeline, ~result.depth.sky)


# task_0009: per-track motion estimation.
def motion_stage(**cfg) -> MotionEstimator:
    return build_motion(MotionConfig(**cfg), distance_type="z")


def test_motion_needs_a_tracker(scene):
    log, det = scene
    with pytest.raises(ConfigError, match=r"^motion: needs tracking"):
        make_pipeline([det], FakeDepth(log), motion=motion_stage())


class SidewaysDetector(FakeDetector):
    """One car box moving 1 px to the right per call, inside the 10 m object."""

    def _load(self, ctx):
        super()._load(ctx)
        self.calls = 0

    def _predict(self, rgb):
        x0 = 25.0 + self.calls
        self.calls += 1
        return dataclasses.replace(
            self.instances, xyxy=np.array([[x0, 40.0, x0 + 20, 60.0]], np.float32)
        )


def test_motion_follows_a_sideways_track(make_instances):
    # 1 px per frame at 30 fps, z = 10 m, fx = 50 px: 6 m/s to the right.
    log = []
    det = SidewaysDetector(
        make_instances([[25, 40, 45, 60]], labels=["car"]), log, name="det"
    )
    pipeline = make_pipeline(
        [det],
        FakeDepth(log, needs_intrinsics=False),
        tracker=IouTracker(IouTracker.Options()),
        motion=motion_stage(),
    )
    pipeline.setup(Source())
    results = [pipeline.process(Frame(k, k / 30, FRAME.rgb)) for k in range(30)]
    last = results[-1]
    assert list(last.timings_ms)[-1] == "motion"
    (m,) = last.motion
    assert m.model == "cv" and m.n_updates == 30
    assert m.vx == pytest.approx(6.0, abs=0.2) and abs(m.vz) < 0.2
    assert m.heading_deg == pytest.approx(90.0, abs=3.0)
    desc = pipeline.describe()["pipeline"]["motion"]
    assert desc["default"] == "cv" and desc["ego_motion"] is None
    # A new run (load after close) starts with fresh motion filters.
    pipeline.close()
    pipeline.setup(Source())
    (m,) = pipeline.process(Frame(0, 0.0, FRAME.rgb)).motion
    assert m.n_updates == 1


def test_warns_when_no_detector_emits_a_motion_class(scene, capsys):
    log, det = scene  # emits car and person
    stage = motion_stage(classes={"vehicle": "cv", "person": "cv"})
    make_pipeline(
        [det],
        FakeDepth(log, needs_intrinsics=False),
        tracker=FakeTracker(),
        motion=stage,
    ).setup(Source())
    err = capsys.readouterr().err
    assert "[warn] motion.classes.vehicle: no detector emits this label" in err
    assert "motion.classes.person" not in err
