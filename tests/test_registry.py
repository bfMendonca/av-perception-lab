import subprocess
import sys

import pytest

from det_depth_distance.adapters.depth.da3 import DA3Depth
from det_depth_distance.adapters.detectors.rfdetr import RFDETRDetector
from det_depth_distance.adapters.detectors.ultralytics_yolo import YoloDetector
from det_depth_distance.core import registry
from det_depth_distance.core.aggregate import Concat, Nms, Wbf
from det_depth_distance.core.options import ConfigError


def test_available_lists_every_family():
    found = registry.available()
    assert found["detector"] == {"rfdetr": RFDETRDetector, "yolo": YoloDetector}
    assert found["depth"] == {"da3": DA3Depth}
    assert found["aggregator"] == {"concat": Concat, "nms": Nms, "wbf": Wbf}
    assert registry.available("depth") == {"depth": {"da3": DA3Depth}}


def test_duplicate_registration_is_an_error():
    with pytest.raises(ValueError, match="already registered"):
        registry.register("detector", "rfdetr")(type("Other", (), {}))


def test_unknown_type_lists_available():
    with pytest.raises(ConfigError, match=r"^detectors\[0\]\.type: .*rfdetr, yolo"):
        registry.build("detector", {"type": "detr"}, path="detectors[0]")
    with pytest.raises(ConfigError, match=r"^depth\.type"):
        registry.build("depth", {"model": "da3metric-large"}, path="depth")


def test_build_validates_options():
    det = registry.build("detector", {"type": "rfdetr", "variant": "nano"}, path="d")
    assert det.name == "rfdetr-nano" and not det.is_loaded
    with pytest.raises(ConfigError, match=r"^d\.treshold: unknown option"):
        registry.build("detector", {"type": "rfdetr", "treshold": 0.3}, path="d")
    with pytest.raises(ConfigError, match=r"^d\.variant: 'base' is not one of"):
        registry.build("detector", {"type": "rfdetr", "variant": "base"}, path="d")


def test_rfdetr_seg_variants():
    det = registry.build(
        "detector", {"type": "rfdetr", "variant": "seg-large"}, path="d"
    )
    assert det.name == "rfdetr-seg-large" and not det.is_loaded
    with pytest.raises(
        ConfigError, match=r"^d\.variant: 'seg-huge' is not one of .*'seg-2xlarge'"
    ):
        registry.build("detector", {"type": "rfdetr", "variant": "seg-huge"}, path="d")


def test_name_only_for_components():
    det = registry.build("detector", {"type": "rfdetr", "name": "main"}, path="d")
    assert det.name == "main"
    depth = registry.build("depth", {"type": "da3", "name": "z"}, path="depth")
    assert depth.name == "z"
    with pytest.raises(
        ConfigError, match=r"^aggregation\.name: aggregators have no name"
    ):
        registry.build(
            "aggregator", {"type": "concat", "name": "x"}, path="aggregation"
        )


def test_depth_process_res_is_validated_by_da3_options():
    # Ported from the old --process-res 10 check: load_config accepts the raw
    # depth spec; DA3Depth.Options rejects it when the component is built.
    with pytest.raises(ConfigError, match=r"^depth: process_res must be >= 14"):
        registry.build("depth", {"type": "da3", "process_res": 10}, path="depth")


def test_relative_depth_model_refused_before_loading():
    with pytest.raises(ConfigError, match=r"^depth\.model: .*relative"):
        registry.build("depth", {"type": "da3", "model": "da3-large"}, path="depth")


def test_load_adapters_imports_no_framework():
    code = (
        "import sys; from det_depth_distance.core import registry; "
        "registry.load_adapters(); "
        "print(sorted({'torch', 'ultralytics', 'rfdetr', 'depth_anything_3'} & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert out.stdout.strip() == "[]", out.stderr
