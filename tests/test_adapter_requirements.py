"""Every adapter's `requirements(options)` (task_0021), without torch or network.

The weights root is a temporary folder (`DDD_WEIGHTS_ROOT`); framework modules
are replaced by stand-ins in `sys.modules` where a `_load` is exercised, and
the provisioning sources are made to fail on any network access.
"""

import sys
import threading
import types
from pathlib import Path

import pytest
import yaml

from det_depth_distance import cli
from det_depth_distance.adapters._hf_hub import HF_FACTS
from det_depth_distance.adapters._ultralytics import ULTRALYTICS_ASSETS
from det_depth_distance.adapters.depth import da3
from det_depth_distance.adapters.detectors import rfdetr as rfdetr_adapter
from det_depth_distance.adapters.semantic.twinlitenet import TWINLITENET_REPO
from det_depth_distance.config import load_config
from det_depth_distance.core import registry
from det_depth_distance.core.components import Component, LoadContext
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.provision import prepare, sources
from det_depth_distance.core.requirements import (
    CodeRepo,
    HuggingFaceAsset,
    Requirements,
    UrlAsset,
    UserProvided,
    check,
    set_weights_root,
)

REPO = Path(__file__).parents[1]
CONFIGS = sorted((REPO / "configs").glob("*.yaml"))
CTX = LoadContext("cpu", (100, 50), None)

# Minimal options for the types that have a required option.
REQUIRED = {
    ("detector", "yolo"): {"weights": "yolo26l.pt"},
    ("refiner", "ultralytics_sam"): {"weights": "mobile_sam.pt"},
}
# Framework-free types implemented in core/: nothing to declare.
EMPTY = {("tracker", "iou")}


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A fresh weights root, no network, adapters registered."""
    weights = tmp_path / "weights"
    set_weights_root(None)
    monkeypatch.setenv("DDD_WEIGHTS_ROOT", str(weights))

    def no_network(*args, **kwargs):
        raise AssertionError("network access in a unit test")

    monkeypatch.setattr(sources, "_open", no_network)
    registry.load_adapters()
    return weights


def _build(kind, type_name, **options):
    spec = {"type": type_name, **REQUIRED.get((kind, type_name), {}), **options}
    return registry.build(kind, spec, path=f"{kind}[0]")


def _components():
    registry.load_adapters()
    return [
        (kind, name, cls)
        for kind in ("detector", "depth", "refiner", "semantic", "tracker")
        for name, cls in registry.available(kind)[kind].items()
    ]


def _fake_hf_cache(cache: Path, repo: str, weights_file: str) -> Path:
    """The `models--org--name` layout of huggingface_hub (refs/main + snapshot)."""
    repo_dir = cache / f"models--{repo.replace('/', '--')}"
    snap = repo_dir / "snapshots" / "4010e39f"
    snap.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("4010e39f")
    (snap / "config.json").write_text("{}")
    with (snap / weights_file).open("wb") as handle:  # sparse, at the declared size
        handle.truncate(HF_FACTS.get(repo, (1,))[0])
    return snap


@pytest.mark.parametrize(("kind", "type_name", "cls"), _components())
def test_every_registered_adapter_declares_or_explicitly_empty_requirements(
    root, kind, type_name, cls
):
    component = _build(kind, type_name)
    req = type(component).requirements(component.options)
    assert isinstance(req, Requirements)
    if (kind, type_name) in EMPTY:
        assert req == Requirements()
        return
    # Adapters override the hook and need the infer stack.
    assert cls.requirements.__func__ is not Component.requirements.__func__
    assert req.extra == "infer" and "torch" in req.packages
    assert req.assets or req.code
    for asset in req.assets:
        assert not Path(asset.target).is_absolute() and ".." not in asset.target
        asset.path(root)  # confined to the weights root (raises otherwise)
        if isinstance(asset, UrlAsset):  # the engine refuses unverifiable files
            assert asset.size_bytes is not None or asset.sha256 is not None
    status = check(req, root)
    assert (
        status.state == "missing" and status.layers["assets" if req.assets else "code"]
    )


def test_da3_requirements_use_hf_cache_layout_and_no_runtime_download(
    root, monkeypatch
):
    depth = _build("depth", "da3")
    req = type(depth).requirements(depth.options)
    (asset,) = req.assets
    assert isinstance(asset, HuggingFaceAsset)
    assert (asset.target, asset.repo) == ("hf", "depth-anything/DA3METRIC-LARGE")
    assert asset.files == ("model.safetensors",) and asset.patterns == ("*.json",)
    assert (asset.size_bytes, asset.license) == (1_336_734_448, "apache-2.0")
    assert set(req.packages) == {"torch", "depth_anything_3"}

    # Missing: a ConfigError naming --prepare, raised before depth_anything_3 is
    # imported (the module is made unimportable), so nothing can be downloaded.
    monkeypatch.setitem(sys.modules, "depth_anything_3", None)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", None)
    with pytest.raises(ConfigError, match=r"^model: .*DA3METRIC-LARGE.*--prepare"):
        da3._load_da3("depth-anything/DA3METRIC-LARGE", "cpu", None)

    # An existing cache in the hub layout (refs/main + snapshot) is ready as is,
    # and the network is loaded from it with local_files_only=True.
    _fake_hf_cache(root / "hf", "depth-anything/DA3METRIC-LARGE", "model.safetensors")
    assert asset.present(root).state == "ready"
    calls = []

    class FakeDA3:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            calls.append((repo, kwargs))
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    api = types.ModuleType("depth_anything_3.api")
    api.DepthAnything3 = FakeDA3
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", api)
    da3._load_da3("depth-anything/DA3METRIC-LARGE", "cpu", None)
    assert calls == [
        (
            "depth-anything/DA3METRIC-LARGE",
            {"cache_dir": str(root / "hf"), "local_files_only": True},
        )
    ]

    # The focal model of camera.estimate is declared for --prepare too.
    focal = da3.da3_requirements("da3-large-1.1", None).assets[0]
    assert (focal.repo, focal.target) == ("depth-anything/DA3-LARGE-1.1", "hf")


def test_rfdetr_loads_from_explicit_checkpoint_path(root, monkeypatch):
    assert set(rfdetr_adapter.RFDETR_CHECKPOINTS) == set(rfdetr_adapter.DET_MODELS)
    detector = _build("detector", "rfdetr", variant="large")
    (asset,) = type(detector).requirements(detector.options).assets
    assert isinstance(asset, UrlAsset)
    assert asset.target == "rfdetr/rf-detr-large-2026.pth"
    assert asset.url == "https://repo.roboflow.com/rfdetr/rf-detr-large-2026.pth"
    assert (asset.size_bytes, asset.license) == (135_954_129, "Apache-2.0")

    built = []

    class RFDETRLarge:
        def __init__(self, **kwargs):
            built.append(kwargs)

    fake = types.ModuleType("rfdetr")
    fake.RFDETRLarge = RFDETRLarge
    monkeypatch.setitem(sys.modules, "rfdetr", fake)
    # Missing: refused before the constructor, which would download the file.
    with pytest.raises(ConfigError, match=r"^variant: .*rf-detr-large-2026.*--prepare"):
        detector._load(CTX)
    assert built == []
    path = root / asset.target
    path.parent.mkdir(parents=True)
    path.write_bytes(b"ckpt")
    detector._load(CTX)
    assert built == [{"pretrain_weights": str(path), "device": "cpu"}]


def test_yolo_custom_pt_is_user_provided(root):
    known = _build("detector", "yolo", weights="yolo26l-objv1-150.pt")
    (url,) = type(known).requirements(known.options).assets
    assert (
        isinstance(url, UrlAsset) and url.target == "ultralytics/yolo26l-objv1-150.pt"
    )
    assert url.url.endswith("/v8.4.0/yolo26l-objv1-150.pt")
    assert url.size_bytes == ULTRALYTICS_ASSETS["yolo26l-objv1-150.pt"][0]

    custom = _build("detector", "yolo", weights="my_custom.pt")
    req = type(custom).requirements(custom.options)
    (asset,) = req.assets
    assert isinstance(asset, UserProvided) and not asset.downloadable
    assert asset.target == "ultralytics/my_custom.pt"
    # Prepare never downloads it (the network is an AssertionError): it says
    # "provide this file" and the model stays not ready.
    status = prepare(req, root, lambda *a: None, threading.Event())
    assert status.state == "failed" and "provide" in status.message
    with pytest.raises(ConfigError, match=r"^weights: file not found at .*--prepare"):
        custom._load(CTX)

    # Ultralytics SAM: a release name is a URL asset, an unpublished one is not.
    sam = _build("refiner", "ultralytics_sam", weights="mobile_sam.pt")
    assert isinstance(type(sam).requirements(sam.options).assets[0], UrlAsset)
    sam_h = _build("refiner", "ultralytics_sam", weights="sam_h.pt")
    assert isinstance(type(sam_h).requirements(sam_h.options).assets[0], UserProvided)


def test_twinlitenet_declares_pinned_clone(root):
    lanes = _build("semantic", "twinlitenet", variant="small")
    req = type(lanes).requirements(lanes.options)
    (code,) = req.code
    assert isinstance(code, CodeRepo) and code == TWINLITENET_REPO
    assert code.url == "https://github.com/chequanghuy/TwinLiteNetPlus"
    assert code.commit.startswith("90f1b86") and len(code.commit) == 40
    assert code.path(root) == root / "_code" / "TwinLiteNetPlus"
    (weights,) = req.assets  # not in the git repo upstream: user-provided
    assert isinstance(weights, UserProvided)
    assert weights.target == "_code/TwinLiteNetPlus/pretrained/small.pth"
    assert "drive.usercontent.google.com" in weights.hint
    assert weights.hint.startswith("manual download")
    large = _build("semantic", "twinlitenet")  # variant large: the folder link
    (large_weights,) = type(large).requirements(large.options).assets
    assert large_weights.hint.startswith("manual download")
    assert (
        "https://drive.google.com/drive/folders/1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz"
        in large_weights.hint
    )
    assert check(req, root).layers["code"] == "missing"
    with pytest.raises(
        ConfigError, match=r"^repo: no TwinLiteNetPlus clone.*--prepare"
    ):
        lanes._load(CTX)
    assert "torch" not in sys.modules


def test_existing_explicit_paths_in_yaml_still_override_defaults(
    root, tmp_path, monkeypatch
):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    (home / "w").mkdir(parents=True)
    (home / "w" / "yolo26l.pt").write_bytes(b"pt")
    config = tmp_path / "explicit.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "source": {"video": "absent.mov"},
                "detectors": [{"type": "yolo", "weights": "~/w/yolo26l.pt"}],
                "depth": {"type": "da3", "hf_cache": "~/hf"},
                "camera": {"estimate": {"hf_cache": "~/hf"}},
                "semantic": [
                    {"name": "lanes", "type": "twinlitenet", "repo": "~/TwinLite"}
                ],
            }
        )
    )
    rows = {
        path: req for path, _, _, req in cli.config_requirements(load_config(config))
    }
    # Locations outside the weights root are the user's: nothing is declared,
    # only the packages; the adapters read exactly the given paths.
    for path in ("detectors[0]", "depth", "camera.estimate", "semantic[0]"):
        assert rows[path].assets == () and rows[path].code == (), path
        assert rows[path].packages
    yolo = registry.build("detector", load_config(config).detectors[0], path="d")
    assert yolo.options.weights == home / "w" / "yolo26l.pt"
    from det_depth_distance.adapters._ultralytics import resolve_weights

    assert resolve_weights(yolo.options.weights) == home / "w" / "yolo26l.pt"

    # An explicit location inside the weights root is declared like a default.
    inside = _build("depth", "da3", hf_cache=str(root / "other"))
    assert type(inside).requirements(inside.options).assets[0].target == "other"
    # Not a file name: refused at option validation.
    for kind, type_name in (("detector", "yolo"), ("refiner", "ultralytics_sam")):
        with pytest.raises(ConfigError, match=r"weights: '\.\.' is not a file name"):
            _build(kind, type_name, weights="..")


@pytest.mark.parametrize(
    ("kind", "type_name", "name"),
    [
        ("detector", "yolo", "yolo26s.pt"),
        ("refiner", "ultralytics_sam", "mobile_sam.pt"),
    ],
)
def test_ultralytics_weights_bare_name_under_root_other_paths_unchanged(
    root, tmp_path, monkeypatch, kind, type_name, name
):
    from det_depth_distance.adapters._ultralytics import resolve_weights

    work = tmp_path / "work"
    (work / "_testbed" / "weights").mkdir(parents=True)
    monkeypatch.chdir(work)

    # A bare file name resolves under <weights root>/ultralytics/ and is declared.
    bare = _build(kind, type_name, weights=name)
    assert resolve_weights(bare.options.weights) == root / "ultralytics" / name
    (asset,) = type(bare).requirements(bare.options).assets
    assert isinstance(asset, UrlAsset) and asset.target == f"ultralytics/{name}"

    # A relative path with a directory part keeps its old meaning: relative to
    # the working directory; outside the root it is not declared (no UrlAsset
    # under the root), and a missing file is a plain "file not found".
    rel = Path("_testbed/weights") / name
    relative = _build(kind, type_name, weights=str(rel))
    assert resolve_weights(relative.options.weights) == rel
    assert type(relative).requirements(relative.options).assets == ()
    with pytest.raises(ConfigError, match=r"^weights: file not found: _testbed"):
        relative._load(CTX)
    assert "ultralytics" not in sys.modules
    (work / rel).write_bytes(b"pt")
    assert resolve_weights(relative.options.weights).resolve() == work / rel

    # An absolute path is untouched (and not declared outside the root).
    absolute = _build(kind, type_name, weights=str(work / rel))
    assert resolve_weights(absolute.options.weights) == work / rel
    assert type(absolute).requirements(absolute.options).assets == ()

    # A relative path with a directory part that lands inside the root is
    # declared at its real location (the same file the adapter opens).
    monkeypatch.chdir(root.parent)
    inside = _build(kind, type_name, weights=f"{root.name}/custom/{name}")
    (declared,) = type(inside).requirements(inside.options).assets
    assert declared.target == f"custom/{name}"


def test_every_shipped_config_loads_and_checks_without_importing_torch(
    root, monkeypatch
):
    assert CONFIGS
    monkeypatch.chdir(REPO)  # relative YAML paths start at the repo root
    for config in CONFIGS:
        cfg = load_config(config)
        rows = cli.config_requirements(cfg)
        assert rows, config
        for path, _, _, req in rows:
            assert not isinstance(req, str), (config, path, req)
            status = check(req, root)  # fresh root: nothing there yet
            expected = "ready" if req == Requirements() else "missing"  # tracker
            assert status.state == expected, (config, path, status)
    assert not {"torch", "rfdetr", "ultralytics", "transformers"} & set(sys.modules)


def test_no_shipped_config_contains_a_home_directory_path():
    for config in CONFIGS:
        text = config.read_text()
        assert "~/" not in text and "/home/" not in text, config
        assert ".roboflow" not in text, config
