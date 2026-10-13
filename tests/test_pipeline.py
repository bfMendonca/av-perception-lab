from dataclasses import dataclass

import numpy as np
import pytest

from det_depth_distance.camera import IntrinsicsProvider, IntrinsicsResult
from det_depth_distance.core.aggregate import Concat
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
    MapGeometry,
)
from det_depth_distance.distance import DistanceConfig
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


def make_pipeline(detectors, depth, *, exclude_sky=True, provider=None):
    return Pipeline(
        detectors,
        depth,
        SpyConcat(),
        DistanceConfig(box_shrink=0.0),
        exclude_sky=exclude_sky,
        intrinsics=provider,
        device="cpu",
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
    assert desc["pipeline"] == {"exclude_sky": False}
