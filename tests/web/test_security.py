"""Hardening of the HTTP surface (task_0018): Host/Origin guards, path
containment of run configs, upload limits, ffmpeg input guard, served-file
headers, YAML expansion bound (fake worker, tiny synthetic clips)."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from conftest import LOCAL
from fastapi.testclient import TestClient
from run_folders import NEW_HEADER, make_run

from av_perception_lab.web import app as app_module
from av_perception_lab.web import library, schema, security, uploads
from av_perception_lab.web.app import create_app
from av_perception_lab.web.paths import WebPaths

FAKE = Path(__file__).with_name("fake_worker.py")
VALID = {"detectors": [{"type": "rfdetr"}], "depth": {"type": "da3"}}


def make_api(web_paths, **kw):
    app = create_app(web_paths, lambda p: [sys.executable, str(FAKE), str(p)], **kw)
    return TestClient(base_url=LOCAL, app=app)


@pytest.fixture
def api(web_paths):
    with make_api(web_paths) as client:
        yield client


def upload(api, path: Path, name: str | None = None):
    files = {"file": (name or path.name, path.read_bytes(), "video/mp4")}
    return api.post("/api/videos", files=files)


# -- C1: source.video without video_id ------------------------------------------


def test_bare_source_video_must_be_inside_the_roots(api, web_paths, video_factory):
    outside = video_factory("outside.mp4")  # tmp/src: no import root
    missing = outside.with_name("nope.mp4")
    answers = []
    for video in (outside, missing, "/etc/passwd"):
        cfg = {**VALID, "source": {"video": str(video)}}
        resp = api.post("/api/runs", json={"config": cfg})
        assert resp.status_code == 422
        answers.append(resp.json()["errors"])
    # The same answer whether the path exists or not.
    assert answers[0] == answers[1] == answers[2]
    assert answers[0][0]["path"] == "source.video"
    inside = video_factory(web_paths.data_roots[0] / "in.mp4")
    cfg = {**VALID, "source": {"video": str(inside)}}
    assert api.post("/api/runs", json={"config": cfg}).status_code == 200
    # A symlink inside a root that leads outside is outside.
    link = web_paths.data_roots[0] / "link.mp4"
    link.symlink_to(outside)
    cfg = {**VALID, "source": {"video": str(link)}}
    assert api.post("/api/runs", json={"config": cfg}).status_code == 422
    # A library original may be named directly.
    record = upload(api, video_factory("u.mp4", seconds=1.0)).json()
    cfg = {**VALID, "source": {"video": record["original"]}}
    assert api.post("/api/runs", json={"config": cfg}).status_code == 200


# -- I1: Host and Origin ----------------------------------------------------------


def test_foreign_host_is_refused(web_paths):
    app = create_app(web_paths)
    assert TestClient(app=app).get("/api/catalog").status_code == 400  # testserver
    rebinding = TestClient(base_url="http://evil.example:8000", app=app)
    assert rebinding.get("/api/runs").status_code == 400
    for host in ("http://127.0.0.1:8000", "http://localhost:8000", "http://[::1]:8000"):
        assert TestClient(base_url=host, app=app).get("/api/catalog").status_code == 200
    lan = create_app(
        web_paths, allowed_hosts=security.allowed_hosts("0.0.0.0", ["192.168.0.7"])
    )
    assert (
        TestClient(base_url="http://192.168.0.7:8000", app=lan)
        .get("/api/catalog")
        .status_code
        == 200
    )


def test_allowed_hosts():
    assert security.allowed_hosts("127.0.0.1") == ["127.0.0.1", "localhost", "[::1]"]
    assert security.allowed_hosts("0.0.0.0", ["Box.lan"]) == [
        "127.0.0.1",
        "localhost",
        "[::1]",
        "box.lan",
    ]
    assert security.allowed_hosts("fe80::1")[-1] == "[fe80::1]"


def test_foreign_origin_on_unsafe_methods_is_403(api, video_factory):
    body = {"config": VALID}
    evil = {"Origin": "http://evil.example"}
    assert api.post("/api/config/validate", json=body, headers=evil).status_code == 403
    assert api.delete("/api/jobs/x", headers=evil).status_code == 403
    clip = video_factory("o.mp4")
    resp = api.post(
        "/api/videos",
        files={"file": ("o.mp4", clip.read_bytes(), "video/mp4")},
        headers=evil,
    )
    assert resp.status_code == 403
    assert (
        api.post(
            "/api/config/validate", json=body, headers={"Origin": "null"}
        ).status_code
        == 403
    )
    cross = {"Sec-Fetch-Site": "cross-site"}
    assert api.post("/api/config/validate", json=body, headers=cross).status_code == 403
    # Same origin, and no Origin at all (curl), pass; reads are never blocked.
    same = {"Origin": LOCAL}
    assert api.post("/api/config/validate", json=body, headers=same).status_code == 200
    assert api.post("/api/config/validate", json=body).status_code == 200
    assert api.get("/api/catalog", headers=evil).status_code == 200


def test_dev_origin_flag_allows_the_dev_server(web_paths):
    dev = {"Origin": "http://localhost:5173"}
    body = {"config": VALID}
    with make_api(web_paths) as plain:
        assert (
            plain.post("/api/config/validate", json=body, headers=dev).status_code
            == 403
        )
    with make_api(web_paths, dev_origins=["http://localhost:5173"]) as api:
        assert (
            api.post("/api/config/validate", json=body, headers=dev).status_code == 200
        )


# -- I2: other path fields --------------------------------------------------------


def test_config_paths_are_discovered_from_the_schemas():
    cfg = {
        "detectors": [
            {"type": "rfdetr", "weights": "w/a.pth", "labels": "l.yaml"},
        ],
        "depth": {"type": "da3"},
        "camera": {"calibration": {"path": "cal.yaml"}},
        "render": {"font": "f.ttf"},
    }
    found = {k: str(v) for k, v in schema.config_paths(cfg)}
    assert found["detectors[0].weights"] == "w/a.pth"
    assert found["detectors[0].labels"] == "l.yaml"
    assert found["camera.calibration.path"] == "cal.yaml"
    assert found["render.font"] == "f.ttf"
    assert "source.video" not in found  # None: nothing to check


def test_path_fields_cannot_point_into_client_writable_folders(api, web_paths):
    for bad in (
        web_paths.library_root / "abc" / "original.mp4",
        web_paths.out_root / "run" / "evil.pth",
        web_paths.library_root / ".incoming" / "x" / "file.part",
    ):
        cfg = {**VALID, "detectors": [{"type": "rfdetr", "weights": str(bad)}]}
        resp = api.post("/api/runs", json={"config": cfg})
        assert resp.status_code == 422, bad
        assert resp.json()["errors"][0]["path"] == "detectors[0].weights"
    cfg = {**VALID, "render": {"font": str(web_paths.out_root / "../out/f.ttf")}}
    resp = api.post("/api/runs", json={"config": cfg})
    assert resp.json()["errors"][0]["path"] == "render.font"
    ok = {**VALID, "detectors": [{"type": "rfdetr", "weights": "weights/x.pth"}]}
    assert api.post("/api/runs", json={"config": ok}).status_code == 200


def test_asset_roots_restrict_path_fields(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    paths = WebPaths(tmp_path / "out", tmp_path / "lib", (), (assets,))
    with make_api(paths) as api:
        inside = {
            **VALID,
            "detectors": [{"type": "rfdetr", "weights": str(assets / "w.pth")}],
        }
        assert api.post("/api/runs", json={"config": inside}).status_code == 200
        outside = {**VALID, "detectors": [{"type": "rfdetr", "weights": "/opt/w.pth"}]}
        resp = api.post("/api/runs", json={"config": outside})
        assert resp.status_code == 422
        assert "--asset-root" in resp.json()["errors"][0]["message"]


def test_lan_bind_without_asset_root_warns():
    from av_perception_lab.web import server

    assert server.startup_warnings("127.0.0.1", [], []) == []
    warnings = server.startup_warnings("0.0.0.0", [], [])
    assert any("--asset-root" in w for w in warnings)
    assert any("--allowed-host" in w for w in warnings)
    assert server.startup_warnings("0.0.0.0", ["box"], [Path("w")]) == []


# -- I3/I4: upload limits ---------------------------------------------------------


def _multipart(parts: list[tuple[list[str], bytes]]) -> tuple[bytes, str]:
    boundary = "XyZ"
    body = b""
    for headers, data in parts:
        body += f"--{boundary}\r\n".encode()
        body += "".join(f"{h}\r\n" for h in headers).encode() + b"\r\n" + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


FILE_HEADER = 'Content-Disposition: form-data; name="file"; filename="v.mp4"'


def test_multipart_header_and_part_limits(api, web_paths):
    huge = FILE_HEADER + "; x=" + "a" * (uploads.MAX_HEADER_BYTES + 10)
    cases = [
        [([huge], b"x")],
        [
            (
                [FILE_HEADER] + [f"X-H{i}: v" for i in range(uploads.MAX_HEADERS + 1)],
                b"x",
            )
        ],
        [
            ([f'Content-Disposition: form-data; name="p{i}"'], b"x")
            for i in range(uploads.MAX_PARTS + 1)
        ],
    ]
    for parts in cases:
        body, ctype = _multipart(parts)
        resp = api.post("/api/videos", content=body, headers={"Content-Type": ctype})
        assert resp.status_code == 400, resp.text
    assert not [p for p in web_paths.library_root.rglob("*") if p.is_file()]


def test_oversize_upload_is_413(api, monkeypatch, video_factory):
    clip = video_factory("big.mp4")
    monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 1000)
    assert upload(api, clip).status_code == 413
    # Declared too large: refused before reading.
    monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 8 << 30)
    body, ctype = _multipart([([FILE_HEADER], b"x")])
    resp = api.post(
        "/api/videos",
        content=body,
        headers={"Content-Type": ctype, "Content-Length": str(10 << 30)},
    )
    assert resp.status_code == 413


def test_meta_part_has_its_own_cap(api, monkeypatch, video_factory):
    monkeypatch.setattr(app_module, "MAX_META_BYTES", 10)
    clip = video_factory("m.h264", fmt="h264")
    files = {
        "file": ("m.h264", clip.read_bytes(), "video/h264"),
        "meta": ("m.meta.json", b"x" * 100, "application/json"),
    }
    assert api.post("/api/videos", files=files).status_code == 413


def test_upload_busy_and_full_disk(web_paths, monkeypatch, video_factory):
    clip = video_factory("d.mp4")
    monkeypatch.setattr(app_module, "_free_bytes", lambda _p: 0)
    with make_api(web_paths) as api:
        assert upload(api, clip).status_code == 507
    monkeypatch.undo()
    monkeypatch.setattr(app_module, "MAX_CONCURRENT_UPLOADS", 0)  # always busy
    with make_api(web_paths) as api:
        assert upload(api, clip).status_code == 429


# -- I5: ffmpeg input guard -------------------------------------------------------


def test_probe_refuses_playlists_and_concat_files(tmp_path, video_factory):
    clip = video_factory("c.mp4")
    concat = tmp_path / "c.mp4.txt"
    concat.write_text(f"ffconcat version 1.0\nfile {clip}\n")
    playlist = tmp_path / "p.m3u8"
    playlist.write_text(f"#EXTM3U\n#EXTINF:2,\n{clip}\n#EXT-X-ENDLIST\n")
    for path in (concat, playlist):
        with pytest.raises(library.InvalidVideo):
            library.probe(path)
    assert library.probe(clip).width == 64  # an accepted container still probes


def test_accepted_containers_probe(video_factory):
    # webm is the matroska demuxer (libx264 cannot go in webm); a raw .h264
    # is remuxed with -f h264 (test_library). AVI is not probed here: its
    # first frame at -ss 0 fails with or without the guard (unrelated).
    for name in ("a.mov", "a.mkv", "a.ts"):
        assert library.probe(video_factory(name)).height == 48, name


def test_preview_encode_times_out(tmp_path, monkeypatch, video_factory):
    clip = video_factory("t.mp4")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "ffmpeg"
    fake.write_text("#!/bin/sh\nexec sleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    with pytest.raises(RuntimeError, match="timed out"):
        library._make_preview(clip, tmp_path / "p.mp4", timeout=0.5)
    assert not list(tmp_path.glob(".p.tmp.mp4"))
    assert library.preview_timeout_s(10) == library.PREVIEW_TIMEOUT_MIN_S
    assert library.preview_timeout_s(3600) == 36000


# -- I6/I7: served files ------------------------------------------------------------


def test_upload_suffix_is_allow_listed_and_served_as_video(api, video_factory):
    clip = video_factory("s.mp4")
    record = upload(api, clip, name="page.html").json()
    assert Path(record["original"]).name == "original.bin"
    resp = api.get(record["urls"]["file"])
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["content-security-policy"] == "sandbox"
    assert "attachment" in resp.headers["content-disposition"]
    other = upload(api, video_factory("s2.mp4", seconds=1.0)).json()
    resp = api.get(other["urls"]["file"])
    assert resp.headers["content-type"] == "video/mp4"
    assert "content-disposition" not in resp.headers
    thumb = api.get(other["urls"]["thumb"])
    assert thumb.headers["content-type"] == "image/jpeg"
    assert thumb.headers["x-content-type-options"] == "nosniff"


def test_missing_original_is_not_served(api, web_paths, video_factory):
    clip = video_factory(web_paths.data_roots[0] / "gone.mp4")
    record = api.post("/api/videos/import", json={"path": str(clip)}).json()
    clip.unlink()
    assert api.get(record["urls"]["file"]).status_code == 404


def test_active_artifacts_are_downloads(api, web_paths):
    run = make_run(web_paths.out_root, "20260101_000000_r", NEW_HEADER, [], {})
    (run / "page.html").write_text("<script>alert(1)</script>")
    (run / "pic.svg").write_text("<svg/>")
    (run / "data.json").write_text("{}")
    for name in ("page.html", "pic.svg"):
        resp = api.get(f"/api/runs/{run.name}/files/{name}")
        assert resp.status_code == 200
        assert resp.headers["content-disposition"].startswith("attachment")
        assert resp.headers["content-security-policy"] == "sandbox"
        assert resp.headers["x-content-type-options"] == "nosniff"
    for url in (
        f"/api/runs/{run.name}/files/data.json",
        f"/api/runs/{run.name}/metrics",
    ):
        resp = api.get(url)
        assert resp.headers["content-type"] == "application/json"
        assert "content-disposition" not in resp.headers
        assert resp.headers["x-content-type-options"] == "nosniff"


def test_hidden_run_ids_do_not_resolve(api, web_paths):
    api.post("/api/runs", json={"config": VALID})  # writes .web_jobs/<id>.json
    assert (web_paths.out_root / ".web_jobs").is_dir()
    for path in (
        "/api/runs/.web_jobs",
        "/api/runs/.web_jobs/artifacts",
        "/api/runs/.web_jobs/config",
    ):
        assert api.get(path).status_code == 404, path


# -- I9: YAML and body size ------------------------------------------------------------


def test_yaml_alias_bomb_is_refused(api, recommended_yaml):
    lines = ["a0: &a0 [x, x, x, x, x, x, x, x, x, x]"]
    for i in range(1, 9):
        lines.append(f"a{i}: &a{i} [{', '.join([f'*a{i - 1}'] * 10)}]")
    resp = api.post("/api/config/parse", json={"yaml": "\n".join(lines)})
    assert resp.status_code == 422 and "expands" in resp.json()["detail"]
    recursive = api.post("/api/config/parse", json={"yaml": "a: &a [*a]"})
    assert recursive.status_code == 422
    # A real config sharing a value through an anchor is still accepted (the
    # shipped configs no longer share hf_cache, task_0021: anchor process_res).
    assert recommended_yaml.count("process_res: 504\n") == 2
    anchored = recommended_yaml.replace(
        "process_res: 504\n", "process_res: &res 504\n", 1
    ).replace("process_res: 504\n", "process_res: *res\n", 1)
    assert "&res" in anchored and "*res" in anchored
    ok = api.post("/api/config/parse", json={"yaml": anchored})
    assert ok.status_code == 200


def test_large_bodies_are_413(api):
    big = "x" * ((1 << 20) + 10)
    assert api.post("/api/config/parse", json={"yaml": big}).status_code == 413
    resp = api.post("/api/config/validate", json={"config": {"k": big}})
    assert resp.status_code == 413

    # Streamed without Content-Length (chunked): counted as it arrives.
    def chunks():
        yield b'{"config": {"k": "'
        for _ in range(20):
            yield b"x" * 65536
        yield b'"}}'

    resp = api.post(
        "/api/config/dump",
        content=chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413


# -- A2/A3: job listing, SSE resume, history titles --------------------------------------


def test_jobs_listing_and_event_resume(api):
    job_id = api.post("/api/runs", json={"config": VALID}).json()["job_id"]
    with api.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        lines = [line for line in resp.iter_lines() if line]
    ids = [int(line[4:]) for line in lines if line.startswith("id: ")]
    assert ids == list(range(len(ids))) and len(ids) >= 4
    listed = api.get("/api/jobs").json()
    assert [j["id"] for j in listed] == [job_id]
    assert listed[0] == api.get(f"/api/jobs/{job_id}").json()
    assert listed[0]["status"] == "done"
    # A reconnect with Last-Event-ID only gets what came after it.
    with api.stream(
        "GET", f"/api/jobs/{job_id}/events", headers={"Last-Event-ID": str(ids[1])}
    ) as resp:
        rest = [json.loads(x[6:]) for x in resp.iter_lines() if x.startswith("data: ")]
    assert rest[0]["event"] != "queued" and len(rest) == len(ids) - 2


def test_uploaded_videos_keep_their_names_in_history(api, video_factory):
    titles = set()
    for name, seconds in (("road.mp4", 1.0), ("city.mp4", 1.5)):
        record = upload(api, video_factory(name, seconds=seconds)).json()
        assert Path(record["original"]).name == "original.mp4"
        job_id = api.post(
            "/api/runs", json={"config": VALID, "video_id": record["id"]}
        ).json()["job_id"]
        with api.stream("GET", f"/api/jobs/{job_id}/events") as resp:
            list(resp.iter_lines())
        job = api.get(f"/api/jobs/{job_id}").json()
        assert job["video"] == name
        assert api.get(f"/api/runs/{job['run_id']}").json()["video"] == name
    titles = {r["video"] for r in api.get("/api/runs").json()}
    assert titles == {"road.mp4", "city.mp4"}
