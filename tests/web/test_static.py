"""The built front-end (`web/dist`) is served at `/` with an SPA fallback."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from det_depth_distance.web.app import create_app


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><div id=root></div>")
    (root / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("outside")
    return root


def test_index_served_at_root(web_paths, dist):
    client = TestClient(create_app(web_paths, static_dir=dist))
    r = client.get("/")
    assert r.status_code == 200 and "id=root" in r.text
    assert r.headers["content-type"].startswith("text/html")


def test_assets_served_and_unknown_routes_fall_back_to_index(web_paths, dist):
    client = TestClient(create_app(web_paths, static_dir=dist))
    assert client.get("/assets/app.js").text == "console.log(1)"
    r = client.get("/runs/some-run")
    assert r.status_code == 200 and "id=root" in r.text


def test_api_routes_never_fall_back(web_paths, dist):
    client = TestClient(create_app(web_paths, static_dir=dist))
    assert client.get("/api/catalog").status_code == 200
    assert client.get("/api/nope").status_code == 404
    assert client.get("/api").status_code == 404


def test_static_traversal_does_not_escape_dist(web_paths, dist):
    client = TestClient(create_app(web_paths, static_dir=dist))
    for path in ("/..%2Fsecret.txt", "/assets/..%2F..%2Fsecret.txt"):
        r = client.get(path)
        assert "outside" not in r.text


def test_without_dist_only_the_api_is_served(web_paths, tmp_path):
    client = TestClient(create_app(web_paths, static_dir=tmp_path / "missing"))
    assert client.get("/api/catalog").status_code == 200
    assert client.get("/").status_code == 404
