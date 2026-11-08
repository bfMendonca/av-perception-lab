import os
import stat

import pytest

from av_perception_lab.core import class_cache
from av_perception_lab.core.components import (
    ClassCatalog,
    Detector,
    DetectorOptions,
    LoadContext,
)
from av_perception_lab.core.labels import COCO
from av_perception_lab.core.requirements import set_weights_root

CTX = LoadContext("cpu", (100, 50), None)


@pytest.fixture
def root(tmp_path):
    set_weights_root(tmp_path / "root")
    (tmp_path / "root").mkdir()
    yield tmp_path / "root"
    set_weights_root(None)


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"abc")
    return path


def test_write_then_read_roundtrip(root, weights):
    class_cache.write(weights, ["car", "person"])
    assert class_cache.read(weights) == ("car", "person")


def test_replaced_weights_file_is_a_miss(root, weights):
    class_cache.write(weights, ["car"])
    weights.write_bytes(b"abcdef")
    assert class_cache.read(weights) is None


def test_touched_mtime_is_a_miss(root, weights):
    class_cache.write(weights, ["car"])
    st = weights.stat()
    os.utime(weights, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert class_cache.read(weights) is None


def test_corrupt_json_is_a_miss(root, weights):
    class_cache.write(weights, ["car", "person"])
    (file,) = (root / ".classes").glob("*.json")
    file.write_text(file.read_text()[:5])
    assert class_cache.read(weights) is None


def test_read_only_root_does_not_raise(root, weights):
    root.chmod(0o500)
    try:
        class_cache.write(weights, ["car"])
    finally:
        root.chmod(stat.S_IRWXU)
    assert class_cache.read(weights) is None


def test_missing_weights_file_has_no_key(root, tmp_path):
    assert class_cache.cache_key(tmp_path / "nope.pt") is None
    assert class_cache.read(tmp_path / "nope.pt") is None


class CachedDetector(Detector):
    type_name = "cached-fake"
    Options = DetectorOptions
    weights = None
    names = (*COCO.presets["driving"], "dog")

    @classmethod
    def weights_file(cls, options):
        return cls.weights

    @property
    def class_names(self):
        return self.names

    def _load(self, ctx):
        pass

    def _predict(self, rgb):
        raise NotImplementedError


def test_detector_load_writes_cache_and_declares_it(root, weights, monkeypatch):
    monkeypatch.setattr(CachedDetector, "weights", weights)
    assert CachedDetector.declared_classes(DetectorOptions()) is None
    CachedDetector(DetectorOptions()).load(CTX)
    assert CachedDetector.declared_classes(DetectorOptions()) == ClassCatalog(
        (*COCO.presets["driving"], "dog"), "cache"
    )


def test_cache_does_not_override_loaded_names(root, weights, monkeypatch):
    monkeypatch.setattr(CachedDetector, "weights", weights)
    class_cache.write(weights, ["stale"])
    det = CachedDetector(DetectorOptions())
    det.load(CTX)
    assert tuple(det.class_names) == (*COCO.presets["driving"], "dog")


def test_detector_without_weights_file_declares_nothing(root):
    assert CachedDetector.declared_classes(DetectorOptions()) is None
