"""POST /api/classes: declared classes of a spec, never loading a model."""

from dataclasses import dataclass

import pytest

pytest.importorskip("fastapi")

from av_perception_lab.core import registry
from av_perception_lab.core import requirements as rq
from av_perception_lab.core.components import Detector, DetectorOptions

URL = "/api/classes"


@dataclass(frozen=True, slots=True, kw_only=True)
class _Opts(DetectorOptions):
    pass


class _Silent(Detector):
    """No declaration; loading is a bug."""

    Options = _Opts
    type_name = "silent"

    def _load(self, ctx):
        raise AssertionError("the endpoint must not load a model")

    @property
    def class_names(self):
        return ()

    def _predict(self, frame):
        raise AssertionError


@pytest.fixture
def silent(monkeypatch, tmp_path):
    monkeypatch.setattr(rq, "_explicit_root", tmp_path / "weights")
    registry.load_adapters()
    monkeypatch.setitem(registry._REGISTRY["detector"], "silent", _Silent)


def test_twinlitenet_returns_static_names(client):
    r = client.post(
        URL, json={"kind": "semantic", "spec": {"type": "twinlitenet", "name": "t"}}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "static" and body["names"]


def test_unknown_classes_returns_null(client, silent):
    r = client.post(URL, json={"kind": "detector", "spec": {"type": "silent"}})
    assert r.status_code == 200
    assert r.json() == {"names": None, "source": None}


def test_never_loads_a_model(client, silent, monkeypatch):
    monkeypatch.setattr(_Silent, "load", lambda self, *a, **k: pytest.fail("load"))
    r = client.post(URL, json={"kind": "detector", "spec": {"type": "silent"}})
    assert r.status_code == 200


def test_invalid_options_is_422(client):
    r = client.post(
        URL, json={"kind": "semantic", "spec": {"type": "twinlitenet", "bogus": 1}}
    )
    assert r.status_code == 422 and r.json()["errors"]


def test_unknown_type_and_kind_are_422(client):
    r = client.post(URL, json={"kind": "semantic", "spec": {"type": "nope"}})
    assert r.status_code == 422 and r.json()["errors"][0]["path"] == "spec.type"
    r = client.post(URL, json={"kind": "depth", "spec": {"type": "da3"}})
    assert r.status_code == 422


def test_extra_body_field_is_422(client):
    body = {"kind": "semantic", "spec": {"type": "twinlitenet"}, "x": 1}
    assert client.post(URL, json=body).status_code == 422


def test_path_outside_allowed_roots_is_rejected(client, web_paths, tmp_path):
    inside = web_paths.library_root / "evil"
    inside.mkdir(parents=True)
    spec = {"type": "segformer", "hf_cache": str(inside)}
    r = client.post(URL, json={"kind": "semantic", "spec": spec})
    assert r.status_code == 422
    assert r.json()["errors"][0]["path"] == "spec.hf_cache"


def test_local_model_dir_outside_weights_is_rejected(client, tmp_path, monkeypatch):
    monkeypatch.setattr(rq, "_explicit_root", tmp_path / "weights")
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "config.json").write_text('{"id2label": {"0": "a"}}')
    spec = {"type": "segformer", "model": str(other)}
    r = client.post(URL, json={"kind": "semantic", "spec": spec})
    assert r.status_code == 422
    assert r.json()["errors"][0]["path"] == "spec.model"


def test_local_model_dir_in_weights_reads_metadata(client, tmp_path, monkeypatch):
    root = tmp_path / "weights"
    monkeypatch.setattr(rq, "_explicit_root", root)
    ck = root / "mine"
    ck.mkdir(parents=True)
    (ck / "config.json").write_text('{"id2label": {"0": "road", "1": "car"}}')
    spec = {"type": "segformer", "model": str(ck)}
    r = client.post(URL, json={"kind": "semantic", "spec": spec})
    assert r.json() == {"names": ["road", "car"], "source": "metadata"}


def _asset_client(tmp_path, web_paths):
    import dataclasses

    from conftest import LOCAL
    from fastapi.testclient import TestClient

    from av_perception_lab.web.app import create_app

    assets = tmp_path / "assets"
    assets.mkdir(exist_ok=True)
    paths = dataclasses.replace(web_paths, asset_roots=(assets,))
    return TestClient(base_url=LOCAL, app=create_app(paths))


def test_labels_outside_roots_leaks_nothing(web_paths, tmp_path, monkeypatch):
    monkeypatch.setattr(rq, "_explicit_root", tmp_path / "weights")
    api = _asset_client(tmp_path, web_paths)
    secret = tmp_path / "secret.txt"
    secret.write_text("zanzibar-secret")
    from av_perception_lab.core import labels

    monkeypatch.setattr(
        labels.LabelMap, "load", classmethod(lambda *a, **k: pytest.fail("read"))
    )
    bodies = []
    for target in (secret, tmp_path / "missing.yaml"):
        spec = {"type": "rfdetr", "labels": str(target)}
        r = api.post(URL, json={"kind": "detector", "spec": spec})
        assert r.status_code == 422
        assert "zanzibar" not in r.text
        bodies.append(r.json()["errors"][0]["path"])
    assert bodies[0] == bodies[1] == "spec.labels"


def test_segformer_model_escapes_are_rejected(web_paths, tmp_path, monkeypatch):
    root = tmp_path / "weights"
    monkeypatch.setattr(rq, "_explicit_root", root)
    api = _asset_client(tmp_path, web_paths)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_text('{"id2label": {"0": "a"}}')
    root.mkdir()
    (root / "link").symlink_to(outside)
    for model in (f"{root}/../outside", str(root / "link")):
        spec = {"type": "segformer", "model": model}
        r = api.post(URL, json={"kind": "semantic", "spec": spec})
        assert r.status_code == 422
        assert r.json()["errors"][0]["path"] == "spec.model"
