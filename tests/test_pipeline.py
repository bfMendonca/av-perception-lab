from dataclasses import dataclass

import numpy as np
import pytest

from det_depth_distance.camera import IntrinsicsProvider, IntrinsicsResult
from det_depth_distance.config import AgreementConfig
from det_depth_distance.core.aggregate import Concat, Nms
from det_depth_distance.core.components import (
    DepthEstimator,
    Detector,
    DetectorOptions,
)
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.pipeline import Pipeline
from det_depth_distance.core.types import (
    CameraIntrinsics,
    DepthResult,
    Frame,
    Instances,
    MapGeometry,
    Provenance,
)
from det_depth_distance.distance import DistanceConfig, box_depth_stats
from det_depth_distance.io.source import FrameSource

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
    with pytest.raises(ConfigError, match="set camera.focal_px or camera.estimate"):
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
    }
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
    assert set(desc["components"]) == {"detectors", "depth", "aggregator"}
    assert desc["components"]["detectors"][0]["name"] == "det"
    assert desc["pipeline"] == {
        "exclude_sky": False,
        "agreement": None,
        "region": "mask",
        "mask_erode": 1,
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
