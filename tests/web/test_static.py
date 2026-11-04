"""The built front-end (`web/dist`) is served at `/` with an SPA fallback."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from conftest import LOCAL
from fastapi.testclient import TestClient

from av_perception_lab.web.app import create_app


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><div id=root></div>")
    (root / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("outside")
    return root


def test_index_served_at_root(web_paths, dist):
    client = TestClient(base_url=LOCAL, app=create_app(web_paths, static_dir=dist))
    r = client.get("/")
    assert r.status_code == 200 and "id=root" in r.text
    assert r.headers["content-type"].startswith("text/html")


def test_assets_served_and_unknown_routes_fall_back_to_index(web_paths, dist):
    client = TestClient(base_url=LOCAL, app=create_app(web_paths, static_dir=dist))
    assert client.get("/assets/app.js").text == "console.log(1)"
    r = client.get("/runs/some-run")
    assert r.status_code == 200 and "id=root" in r.text


def test_api_routes_never_fall_back(web_paths, dist):
    client = TestClient(base_url=LOCAL, app=create_app(web_paths, static_dir=dist))
    assert client.get("/api/catalog").status_code == 200
    for path in ("/api/nope", "/api", "/api/runs/x/y/z"):
        r = client.get(path)
        assert r.status_code == 404
        assert r.headers["content-type"].startswith("application/json")
        assert "detail" in r.json()


def test_missing_asset_is_404_not_index(web_paths, dist):
    client = TestClient(base_url=LOCAL, app=create_app(web_paths, static_dir=dist))
    for path in ("/assets/index-stale.js", "/assets/x", "/favicon.ico"):
        r = client.get(path)
        assert r.status_code == 404, path
        assert "id=root" not in r.text


def test_static_traversal_does_not_escape_dist(web_paths, dist):
    client = TestClient(base_url=LOCAL, app=create_app(web_paths, static_dir=dist))
    for path in ("/..%2Fsecret.txt", "/assets/..%2F..%2Fsecret.txt"):
        r = client.get(path)
        assert r.status_code == 404
        assert "outside" not in r.text


def test_server_static_dir_flag(monkeypatch, tmp_path):
    import uvicorn

    from av_perception_lab.web import app as app_module
    from av_perception_lab.web import server

    seen = {}

    def fake_create_app(paths, **kwargs):
        seen.update(kwargs)
        return "app"

    monkeypatch.setattr(app_module, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    server.main(["--static-dir", str(tmp_path / "ui")])
    assert seen["static_dir"] == tmp_path / "ui" and seen["app"] == "app"
    assert seen["timeout_graceful_shutdown"] == 3  # Ctrl-C with an open SSE stream
    seen.clear()
    server.main([])
    assert "static_dir" not in seen  # default: create_app's own DEFAULT_STATIC_DIR
    assert seen["allowed_hosts"] == ["127.0.0.1", "localhost", "[::1]"]


def test_without_dist_only_the_api_is_served(web_paths, tmp_path):
    client = TestClient(
        base_url=LOCAL, app=create_app(web_paths, static_dir=tmp_path / "missing")
    )
    assert client.get("/api/catalog").status_code == 200
    assert client.get("/").status_code == 404
