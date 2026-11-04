"""HTTP routes of the config presets (`/api/presets`)."""

from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")

from conftest import LOCAL, REPO
from fastapi.testclient import TestClient

from det_depth_distance.web import app as web_app
from det_depth_distance.web.app import create_app
from det_depth_distance.web.paths import WebPaths

VALID = yaml.safe_load((REPO / "configs/rfdetr_da3.yaml").read_text())
PIPELINE = {k: v for k, v in VALID.items() if k not in ("source", "output")}


@pytest.fixture
def builtin_dir(tmp_path: Path) -> Path:
    d = tmp_path / "builtin"
    d.mkdir()
    (d / "alpha.yaml").write_text(
        yaml.safe_dump({**VALID, "mystery": 1}, sort_keys=False), encoding="utf-8"
    )
    (d / "presets.yaml").write_text(
        yaml.safe_dump(
            {"presets": [{"file": "alpha.yaml", "name": "Alpha", "description": "A"}]}
        ),
        encoding="utf-8",
    )
    return d


@pytest.fixture
def user_root(tmp_path: Path) -> Path:
    return tmp_path / "presets"


@pytest.fixture
def preset_client(tmp_path, builtin_dir, user_root) -> TestClient:
    paths = WebPaths(
        tmp_path / "out",
        tmp_path / "library",
        (tmp_path / "data",),
        presets_root=user_root,
    )
    return TestClient(
        base_url=LOCAL, app=create_app(paths, builtin_presets_dir=builtin_dir)
    )


def _put(client, name="Mine", config=None, **kw):
    body = {"name": name, "config": PIPELINE if config is None else config, **kw}
    return client.put("/api/presets", json=body)


def test_list_returns_builtin_and_user(preset_client):
    assert _put(preset_client).status_code == 200
    rows = preset_client.get("/api/presets").json()
    assert rows == [
        {"id": "builtin:alpha", "name": "Alpha", "description": "A", "kind": "builtin"},
        {"id": "user:mine", "name": "Mine", "description": "", "kind": "user"},
    ]


def test_list_works_when_presets_root_missing_or_none(
    preset_client, tmp_path, builtin_dir
):
    assert [r["id"] for r in preset_client.get("/api/presets").json()] == [
        "builtin:alpha"
    ]
    paths = WebPaths(tmp_path / "out", tmp_path / "library")
    c = TestClient(
        base_url=LOCAL, app=create_app(paths, builtin_presets_dir=builtin_dir)
    )
    assert [r["id"] for r in c.get("/api/presets").json()] == ["builtin:alpha"]
    assert _put(c).status_code == 404


def test_get_builtin_strips_source_output_and_splits_extras(preset_client):
    body = preset_client.get("/api/presets/builtin:alpha").json()
    assert set(body) == {"config", "extras"}
    assert "source" not in body["config"] and "output" not in body["config"]
    assert body["extras"] == {"mystery": 1}
    assert "detectors" in body["config"]


def test_put_saves_and_get_roundtrips_without_source_output(preset_client, user_root):
    config = {**PIPELINE, "source": {"video": "x"}, "output": {"out_root": "y"}}
    resp = _put(preset_client, config=config, extras={"mystery": 2})
    assert resp.status_code == 200
    assert resp.json()["id"] == "user:mine"
    text = (user_root / "mine.yaml").read_text(encoding="utf-8")
    assert text.startswith("# name: Mine\n")
    on_disk = yaml.safe_load(text)
    assert "source" not in on_disk and "output" not in on_disk
    got = preset_client.get("/api/presets/user:mine").json()
    assert got["config"] == PIPELINE
    assert got["extras"] == {"mystery": 2}


def test_put_empty_slug_name_422_and_no_file(preset_client, user_root):
    resp = _put(preset_client, name="!!!")
    assert resp.status_code == 422
    assert isinstance(resp.json()["detail"], str)
    assert not user_root.exists() or not list(user_root.iterdir())


def test_put_invalid_config_422_lists_errors(preset_client, user_root):
    resp = _put(preset_client, config={})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail and {"path", "message"} <= set(detail[0])
    assert not user_root.exists() or not list(user_root.iterdir())


def test_put_existing_409_then_overwrite_200(preset_client):
    assert _put(preset_client).status_code == 200
    assert _put(preset_client).status_code == 409
    assert _put(preset_client, overwrite=True).status_code == 200


def test_put_builtin_name_409(preset_client, user_root):
    assert _put(preset_client, name="Alpha").status_code == 409
    assert _put(preset_client, name="alpha").status_code == 409
    assert not user_root.exists() or not list(user_root.iterdir())


def test_put_oversize_413(preset_client, monkeypatch):
    monkeypatch.setattr(web_app, "MAX_YAML_CHARS", 50)
    assert _put(preset_client).status_code == 413


def test_put_unrepresentable_data_is_422(preset_client, user_root, monkeypatch):
    # No JSON body can carry a non-YAML object, so make the dump fail.
    def boom(*a, **k):
        raise yaml.representer.RepresenterError("cannot represent")

    monkeypatch.setattr(yaml, "safe_dump", boom)
    resp = _put(preset_client)
    assert resp.status_code == 422
    assert not user_root.exists() or not list(user_root.glob("*.yaml"))


def test_put_dump_failure_inside_save_is_422(preset_client, monkeypatch):
    def boom(*a, **k):
        raise yaml.representer.RepresenterError("cannot represent")

    monkeypatch.setattr(web_app.presets, "save_preset", boom)
    assert _put(preset_client).status_code == 422


def test_get_and_delete_traversal_ids_never_leave_root(
    preset_client, tmp_path, user_root
):
    canary = tmp_path / "x.yaml"
    canary.write_text("a: 1\n")
    _put(preset_client)
    for pid in ("user:..%2Fx", "user:%2e%2e", "user:%2Fetc%2Fpasswd", "user:../x"):
        assert preset_client.get(f"/api/presets/{pid}").status_code in (404, 405, 422)
        assert preset_client.delete(f"/api/presets/{pid}").status_code in (
            404,
            405,
            422,
        )
    assert canary.read_text() == "a: 1\n"
    assert (user_root / "mine.yaml").exists()


def test_delete_user_204_then_404(preset_client):
    _put(preset_client)
    assert preset_client.delete("/api/presets/user:mine").status_code == 204
    assert preset_client.delete("/api/presets/user:mine").status_code == 404
    assert preset_client.get("/api/presets/user:mine").status_code == 404


def test_delete_builtin_403(preset_client, builtin_dir):
    assert preset_client.delete("/api/presets/builtin:alpha").status_code == 403
    assert (builtin_dir / "alpha.yaml").exists()


def test_invalid_user_file_listed_and_get_422(preset_client, user_root):
    user_root.mkdir()
    (user_root / "bad.yaml").write_text("[1, 2\n", encoding="utf-8")
    ids = [r["id"] for r in preset_client.get("/api/presets").json()]
    assert "user:bad" in ids
    resp = preset_client.get("/api/presets/user:bad")
    assert resp.status_code == 422
    assert "bad.yaml" in resp.json()["detail"]


def test_state_changing_routes_require_local_origin(preset_client):
    _put(preset_client)
    evil = {"Origin": "http://evil.example"}
    body = {"name": "Other", "config": PIPELINE}
    assert preset_client.put("/api/presets", json=body, headers=evil).status_code == 403
    assert (
        preset_client.delete("/api/presets/user:mine", headers=evil).status_code == 403
    )
    assert preset_client.get("/api/presets/user:mine").status_code == 200


def test_put_invalid_known_key_in_extras_is_422(preset_client, user_root):
    resp = _put(preset_client, extras={"tracking": {"bogus_option": 1}})
    assert resp.status_code == 422
    assert isinstance(resp.json()["detail"], list)
    assert not user_root.exists() or not list(user_root.glob("*.yaml"))


def test_put_unknown_extras_key_is_saved_and_roundtrips(preset_client):
    assert _put(preset_client, extras={"mystery": {"a": 1}}).status_code == 200
    got = preset_client.get("/api/presets/user:mine").json()
    assert got["extras"] == {"mystery": {"a": 1}}
