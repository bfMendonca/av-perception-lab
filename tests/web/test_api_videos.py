"""Video routes: upload (streamed multipart), import, listing, files, and the
`video_id` check of POST /api/runs (fake worker, tiny synthetic clips)."""

import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from conftest import LOCAL
from fastapi.testclient import TestClient

from av_perception_lab.web.app import create_app

FAKE = Path(__file__).with_name("fake_worker.py")
VALID = {"detectors": [{"type": "rfdetr"}], "depth": {"type": "da3"}}


@pytest.fixture
def api(web_paths):
    app = create_app(web_paths, lambda p: [sys.executable, str(FAKE), str(p)])
    with TestClient(base_url=LOCAL, app=app) as client:
        yield client


def files_under(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


def upload(api, path: Path, name: str | None = None, **extra):
    files = {"file": (name or path.name, path.read_bytes(), "video/mp4")}
    for field, (fname, data) in extra.items():
        files[field] = (fname, data, "application/octet-stream")
    return api.post("/api/videos", files=files)


def wait_preview(api, video_id: str, timeout: float = 60) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = api.get(f"/api/videos/{video_id}").json()
        if record["preview_status"] != "pending":
            return record
        time.sleep(0.1)
    raise AssertionError("preview not ready")


def test_upload_endpoint_returns_record(api, video_factory):
    clip = video_factory("road.mp4", seconds=2, fps=10, size="64x48")
    resp = upload(api, clip)
    assert resp.status_code == 200, resp.text
    record = resp.json()
    assert record["name"] == "road.mp4"
    assert (record["width"], record["height"]) == (64, 48)
    assert record["fps"] == pytest.approx(10)
    assert record["duration_s"] == pytest.approx(2.0, abs=0.05)
    assert record["size_bytes"] == clip.stat().st_size
    assert Path(record["original"]).is_absolute()
    assert record["urls"]["file"] == f"/api/videos/{record['id']}/file"
    assert [r["id"] for r in api.get("/api/videos").json()] == [record["id"]]
    assert upload(api, clip, "copy.mp4").json()["id"] == record["id"]  # dedupe
    file = api.get(f"/api/videos/{record['id']}/file")
    assert file.status_code == 200 and file.content == clip.read_bytes()
    part = api.get(f"/api/videos/{record['id']}/file", headers={"Range": "bytes=0-9"})
    assert part.status_code == 206 and len(part.content) == 10
    thumb = api.get(f"/api/videos/{record['id']}/thumb")
    assert thumb.status_code == 200 and thumb.content[:2] == b"\xff\xd8"
    ready = wait_preview(api, record["id"])
    assert ready["preview_status"] == "ready" and ready["preview"]
    assert api.get(f"/api/videos/{record['id']}/preview").status_code == 200


def test_upload_rejects_non_video_without_leftovers(api, web_paths, tmp_path):
    text = tmp_path / "notes.mp4"
    text.write_text("not a video\n" * 50)
    resp = upload(api, text)
    assert resp.status_code == 422
    assert "video" in resp.json()["detail"].lower()
    assert api.get("/api/videos").json() == []
    assert files_under(web_paths.library_root) == []


def test_upload_truncated_video_rejected(api, web_paths, video_factory, tmp_path):
    data = video_factory("t.mp4", seconds=3).read_bytes()
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(data[: len(data) // 2])
    assert upload(api, cut).status_code == 422
    assert files_under(web_paths.library_root) == []


def test_upload_cut_multipart_body_is_400_without_leftovers(api, web_paths):
    boundary = "XyZ"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="a.mp4"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
    ).encode() + b"\x00" * 5000  # the closing boundary never arrives
    resp = api.post(
        "/api/videos",
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert resp.status_code == 400
    assert files_under(web_paths.library_root) == []


def test_upload_without_file_field_is_400(api):
    assert api.post("/api/videos", files={"other": ("a", b"x")}).status_code == 400
    resp = api.post("/api/videos", content=b"x", headers={"Content-Type": "text/plain"})
    assert resp.status_code == 400


def test_upload_h264_with_meta(api, video_factory, tmp_path):
    raw = video_factory("0000.h264", seconds=0.5, fmt="h264")
    t0 = 56216714000
    meta = (
        "["
        + ",".join(f'{{"SensorTimestamp": {t0 + i * 41_000_000}}}' for i in range(5))
        + "]"
    )
    resp = upload(api, raw, meta=("0000.meta.json", meta.encode()))
    assert resp.status_code == 200, resp.text
    record = resp.json()
    assert record["original"].endswith(".mkv")
    assert 0.164 <= record["duration_s"] <= 0.4
    alone = upload(api, video_factory("1111.h264", seconds=0.3, fmt="h264"))
    assert alone.status_code == 422 and "meta.json" in alone.json()["detail"]


def test_import_endpoint(api, web_paths, video_factory, tmp_path):
    root = web_paths.data_roots[0]
    inside = video_factory(root / "s1" / "a.mp4")
    outside = video_factory(tmp_path / "secret" / "b.mp4")
    (root / "link.mp4").symlink_to(outside)
    resp = api.post("/api/videos/import", json={"path": str(inside)})
    assert resp.status_code == 200, resp.text
    assert resp.json()["original"] == str(inside.resolve())
    for bad in (str(outside), f"{root}/../secret/b.mp4", str(root / "link.mp4")):
        assert api.post("/api/videos/import", json={"path": bad}).status_code == 403
    # relative paths start from the server's working directory (the repo root)
    assert api.post("/api/videos/import", json={"path": "a.mp4"}).status_code == 403
    missing = api.post("/api/videos/import", json={"path": str(root / "none.mp4")})
    assert missing.status_code == 404
    assert len(api.get("/api/videos").json()) == 1
    roots = api.get("/api/videos/roots").json()
    assert roots == {"roots": [str(root.resolve())], "cwd": str(Path.cwd().resolve())}


def test_import_relative_path_from_cwd(api, web_paths, video_factory, monkeypatch):
    root = web_paths.data_roots[0]
    video_factory(root / "s1" / "a.mp4")
    video_factory(root.parent / "secret" / "b.mp4")
    monkeypatch.chdir(root.parent)  # the server's working directory
    resp = api.post("/api/videos/import", json={"path": "data/s1/a.mp4"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["original"] == str((root / "s1" / "a.mp4").resolve())
    escape = api.post("/api/videos/import", json={"path": "data/../secret/b.mp4"})
    assert escape.status_code == 403


def test_video_file_routes_reject_bad_ids(api):
    for vid in ("0123456789ab", "..", "%2e%2e", "..%2fsecret"):
        assert api.get(f"/api/videos/{vid}").status_code == 404
        assert api.get(f"/api/videos/{vid}/file").status_code == 404
    assert api.get("/api/videos/0123456789ab/other").status_code == 404


def test_runs_require_a_library_video_id(api, web_paths, video_factory):
    resp = api.post("/api/runs", json={"config": VALID, "video_id": "0123456789ab"})
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["path"] == "video_id"
    record = upload(api, video_factory("v.mp4")).json()
    resp = api.post("/api/runs", json={"config": VALID, "video_id": record["id"]})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    job = api.app.state.jobs.get(job_id)
    # A config without source.video runs the library video.
    assert job.config["source"]["video"] == record["original"]
    # source.video is bound to video_id: a client path is overridden.
    other = {**VALID, "source": {"video": "/elsewhere.mp4", "start": 1.0}}
    job_id = api.post(
        "/api/runs", json={"config": other, "video_id": record["id"]}
    ).json()["job_id"]
    source = api.app.state.jobs.get(job_id).config["source"]
    assert source == {"video": record["original"], "start": 1.0}


def test_runs_refuse_a_missing_library_video(api, web_paths, video_factory):
    root = web_paths.data_roots[0]
    clip = video_factory(root / "m.mp4")
    record = api.post("/api/videos/import", json={"path": str(clip)}).json()
    clip.unlink()
    listed = api.get(f"/api/videos/{record['id']}").json()
    assert listed["missing"] is True
    resp = api.post("/api/runs", json={"config": VALID, "video_id": record["id"]})
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["path"] == "video_id"
