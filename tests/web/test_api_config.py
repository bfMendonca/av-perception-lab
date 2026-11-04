import os

import pytest
import yaml

pytest.importorskip("fastapi")

from av_perception_lab.config import load_config
from av_perception_lab.web.paths import safe_join


def test_catalog_endpoint(client):
    body = client.get("/api/catalog").json()
    assert body["catalog_version"] == 1 and body["stages"]


def test_validate_endpoint_422_free(client):
    for bad in (
        {},
        {"detectors": "x"},
        {"detectors": [1], "depth": []},
        {"detectors": [{"type": "nope"}], "source": 3},
    ):
        r = client.post("/api/config/validate", json={"config": bad})
        assert r.status_code == 200
        assert r.json()["errors"]


def test_validate_endpoint_accepts_valid(client, recommended_yaml):
    r = client.post(
        "/api/config/validate", json={"config": yaml.safe_load(recommended_yaml)}
    )
    assert r.json() == {"errors": []}


def test_parse_keeps_unknown_keys_as_extras(client, recommended_yaml):
    text = recommended_yaml + "\nmy_notes: {a: 1}\n"
    body = client.post("/api/config/parse", json={"yaml": text}).json()
    assert body["extras"] == {"my_notes": {"a": 1}}
    assert "detectors" in body["config"] and "my_notes" not in body["config"]


def test_parse_rejects_bad_yaml(client):
    assert client.post("/api/config/parse", json={"yaml": "a: [1"}).status_code == 422
    assert client.post("/api/config/parse", json={"yaml": "- 1"}).status_code == 422


def test_dump_roundtrips_through_load_config(client, recommended_yaml, tmp_path):
    parsed = client.post("/api/config/parse", json={"yaml": recommended_yaml}).json()
    dumped = client.post("/api/config/dump", json={"config": parsed["config"]}).json()
    out = tmp_path / "dumped.yaml"
    out.write_text(dumped["yaml"])
    original = tmp_path / "orig.yaml"
    original.write_text(recommended_yaml)
    assert load_config(out) == load_config(original)


def test_lifted_distance_keys_round_trip(client, recommended_yaml):
    data = yaml.safe_load(recommended_yaml)
    data["distance"] = {
        **(data.get("distance") or {}),
        "exclude_sky": False,
        "region": "box",
        "exclude_semantic": [],
    }
    r = client.post("/api/config/validate", json={"config": data})
    assert r.json() == {"errors": []}
    text = client.post("/api/config/dump", json={"config": data}).json()["yaml"]
    parsed = client.post("/api/config/parse", json={"yaml": text}).json()
    assert parsed["config"]["distance"]["region"] == "box"
    assert parsed["config"]["distance"]["exclude_sky"] is False
    assert parsed["extras"] == {}


def test_safe_join_rejects_traversal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside)
    (root / "ok").mkdir()
    assert safe_join(root, "ok") == (root / "ok").resolve()
    for rel in ("../x", "ok/../../x", str(outside), "link", "link/f"):
        with pytest.raises(ValueError):
            safe_join(root, rel)
    assert os.path.exists(outside)
