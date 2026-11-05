"""ROS 2 MCAP inputs in the web library (task_0033): upload and import, the
topics route, run submission binding `source.mcap`, and the early checks of
`source.topic`, `rectify.enabled` and `camera.calibration.from_source`
(fake worker, tiny MCAPs written by tests/mcap_fixture.py)."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")

from conftest import LOCAL
from fastapi.testclient import TestClient
from mcap_fixture import CameraInfoSpec, ImageSpec, write_mcap
from run_folders import NEW_HEADER, make_run

from av_perception_lab.web.app import create_app
from av_perception_lab.web.library import Library

FAKE = Path(__file__).with_name("fake_worker.py")
VALID = {"detectors": [{"type": "rfdetr"}], "depth": {"type": "da3"}}
K = [100, 0, 8, 0, 100, 6, 0, 0, 1]
RAW_D = [-0.1, 0.01, 0.0, 0.0, 0.0]


@pytest.fixture
def api(web_paths):
    app = create_app(web_paths, lambda p: [sys.executable, str(FAKE), str(p)])
    with TestClient(base_url=LOCAL, app=app) as client:
        yield client


def frames(n: int, h: int = 12, w: int = 16, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 4, w // 4, 3), dtype=np.uint8)
    return [cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)] * n


def recording(path: Path, indexed: bool = True) -> Path:
    """Three image topics: /cam (raw CameraInfo), /front (jpeg, image_transport
    format, no CameraInfo) and /rect (already rectified CameraInfo)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_mcap(
        path,
        images=[
            ImageSpec(
                topic="/cam/image_raw",
                frames=frames(3),
                times_ns=[0, 500_000_000, 1_000_000_000],
            ),
            ImageSpec(
                topic="/front/image_raw/compressed",
                frames=frames(2, 8, 12, seed=1),
                times_ns=[0, 250_000_000],
                compressed="jpeg",
                format="rgb8; jpeg compressed bgr8",
            ),
            ImageSpec(
                topic="/rect/image_raw",
                frames=frames(2),
                times_ns=[0, 100_000_000],
                encoding="bgr8",
            ),
        ],
        camera_infos=[
            CameraInfoSpec(topic="/cam/camera_info", width=16, height=12, k=K, d=RAW_D),
            CameraInfoSpec(topic="/rect/camera_info", width=16, height=12, k=K),
        ],
        indexed=indexed,
    )


def upload(api, path: Path, name: str | None = None):
    files = {"file": (name or path.name, path.read_bytes(), "application/octet-stream")}
    return api.post("/api/videos", files=files)


def files_under(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


def by_name(topics: dict) -> dict[str, dict]:
    return {t["name"]: t for t in topics["images"]}


def test_upload_mcap_lists_topics(api, tmp_path, video_factory):
    rec = recording(tmp_path / "src" / "drive.mcap")
    resp = upload(api, rec)
    assert resp.status_code == 200, resp.text
    record = resp.json()
    assert record["kind"] == "mcap" and record["name"] == "drive.mcap"
    # size and rate of the first image topic (sorted by name: /cam)
    assert (record["width"], record["height"]) == (16, 12)
    assert record["fps"] == pytest.approx(2.0)
    assert record["duration_s"] == pytest.approx(1.0)
    assert record["preview"] is None and record["preview_status"] == "ready"
    assert record["original"].endswith("original.mcap")
    assert record["urls"]["topics"] == f"/api/videos/{record['id']}/topics"
    topics = api.get(f"/api/videos/{record['id']}/topics")
    assert topics.status_code == 200, topics.text
    data = topics.json()
    assert data == record["mcap"]
    assert data["duration_s"] == pytest.approx(1.0)
    assert data["camera_info_topics"] == ["/cam/camera_info", "/rect/camera_info"]
    cam, front, rect = (
        by_name(data)[n]
        for n in ("/cam/image_raw", "/front/image_raw/compressed", "/rect/image_raw")
    )
    assert [t["name"] for t in data["images"]] == [
        "/cam/image_raw",
        "/front/image_raw/compressed",
        "/rect/image_raw",
    ]
    assert cam == {
        "name": "/cam/image_raw",
        "schema": "sensor_msgs/msg/Image",
        "width": 16,
        "height": 12,
        "encoding": "rgb8",
        "rate_hz": pytest.approx(2.0),
        "count": 3,
        "duration_s": pytest.approx(1.0),
        "camera_info_topic": "/cam/camera_info",
        "has_camera_info": True,
        "has_distortion": True,
        "camera_info_error": None,
        "supported": True,
        "unsupported_reason": None,
    }
    assert front["schema"] == "sensor_msgs/msg/CompressedImage"
    assert (front["width"], front["height"], front["count"]) == (12, 8, 2)
    assert front["encoding"] == "jpeg"  # not the "rgb8" of the format string
    assert front["rate_hz"] == pytest.approx(4.0)
    assert front["camera_info_topic"] is None
    assert front["has_camera_info"] is False and front["has_distortion"] is False
    assert rect["camera_info_topic"] == "/rect/camera_info"
    assert rect["has_camera_info"] is True and rect["has_distortion"] is False
    # thumb (first frame of the first topic) and the original are served
    thumb = api.get(f"/api/videos/{record['id']}/thumb")
    assert thumb.status_code == 200 and thumb.content[:2] == b"\xff\xd8"
    file = api.get(f"/api/videos/{record['id']}/file")
    assert file.status_code == 200 and file.content == rec.read_bytes()
    assert file.headers["content-type"] == "application/octet-stream"
    assert api.get(f"/api/videos/{record['id']}/preview").status_code == 404
    assert api.get(f"/api/videos/{record['id']}").json() == record
    # dedupe, and the listing keeps both kinds
    assert upload(api, rec, "again.mcap").json()["id"] == record["id"]
    clip = video_factory("road.mp4")
    video = api.post(
        "/api/videos", files={"file": ("road.mp4", clip.read_bytes(), "video/mp4")}
    ).json()
    kinds = {r["id"]: r["kind"] for r in api.get("/api/videos").json()}
    assert kinds == {record["id"]: "mcap", video["id"]: "video"}
    # a restarted server loads the entry again, with no preview to schedule
    api.app.state.library.wait_previews(30)
    again = Library(api.app.state.paths.library_root).get(record["id"])
    assert again.kind == "mcap" and again.mcap == record["mcap"]


def test_import_mcap_by_path_under_roots_and_403_outside(api, web_paths, tmp_path):
    root = web_paths.data_roots[0]
    inside = recording(root / "s1" / "a.mcap")
    outside = recording(tmp_path / "secret" / "b.mcap")
    (root / "link.mcap").symlink_to(outside)
    resp = api.post("/api/videos/import", json={"path": str(inside)})
    assert resp.status_code == 200, resp.text
    record = resp.json()
    assert record["kind"] == "mcap" and record["origin"] == "import"
    assert record["original"] == str(inside.resolve())  # by reference, no copy
    assert not list((web_paths.library_root / record["id"]).glob("original*"))
    assert len(by_name(api.get(f"/api/videos/{record['id']}/topics").json())) == 3
    for bad in (str(outside), f"{root}/../secret/b.mcap", str(root / "link.mcap")):
        assert api.post("/api/videos/import", json={"path": bad}).status_code == 403
    missing = api.post("/api/videos/import", json={"path": str(root / "none.mcap")})
    assert missing.status_code == 404
    file = api.get(f"/api/videos/{record['id']}/file")
    assert file.status_code == 200 and file.content == inside.read_bytes()
    # a changed import is missing (and cannot run)
    inside.write_bytes(inside.read_bytes() + b"\0")
    assert api.get(f"/api/videos/{record['id']}").json()["missing"] is True


def test_mcap_without_image_topics_is_422(api, web_paths, tmp_path):
    only_info = write_mcap(
        tmp_path / "info.mcap",
        camera_infos=[
            CameraInfoSpec(topic="/cam/camera_info", width=16, height=12, k=K)
        ],
    )
    resp = upload(api, only_info)
    assert resp.status_code == 422
    assert "image topic" in resp.json()["detail"]
    assert api.get("/api/videos").json() == []
    assert files_under(web_paths.library_root) == []
    root = web_paths.data_roots[0]
    root.mkdir(parents=True, exist_ok=True)
    (root / "info.mcap").write_bytes(only_info.read_bytes())
    resp = api.post("/api/videos/import", json={"path": str(root / "info.mcap")})
    assert resp.status_code == 422 and "image topic" in resp.json()["detail"]


@pytest.mark.parametrize("indexed", [True, False])
def test_corrupt_or_truncated_mcap_is_422_not_500(api, web_paths, tmp_path, indexed):
    good = recording(tmp_path / "good.mcap", indexed=indexed)
    data = good.read_bytes()
    truncated = tmp_path / "cut.mcap"
    truncated.write_bytes(data[: len(data) // 2])
    garbage = tmp_path / "garbage.mcap"
    garbage.write_bytes(b"not an mcap at all\n" * 40)
    for bad in (truncated, garbage):
        resp = upload(api, bad)
        assert resp.status_code == 422, resp.text
        detail = resp.json()["detail"]
        assert "MCAP" in detail
        assert str(web_paths.library_root) not in detail  # no server paths
        assert api.get("/api/videos").json() == []
        assert files_under(web_paths.library_root) == []
        root = web_paths.data_roots[0]
        root.mkdir(parents=True, exist_ok=True)
        (root / bad.name).write_bytes(bad.read_bytes())
        resp = api.post("/api/videos/import", json={"path": str(root / bad.name)})
        assert resp.status_code == 422, resp.text


def test_unindexed_mcap_is_accepted(api, tmp_path):
    rec = recording(tmp_path / "plain.mcap", indexed=False)
    resp = upload(api, rec)
    assert resp.status_code == 200, resp.text
    record = resp.json()
    assert record["kind"] == "mcap"
    topics = by_name(api.get(f"/api/videos/{record['id']}/topics").json())
    assert topics["/cam/image_raw"]["count"] == 3
    assert topics["/cam/image_raw"]["has_distortion"] is True


def test_run_with_mcap_binds_source_mcap(api, web_paths, tmp_path):
    record = upload(api, recording(tmp_path / "drive.mcap")).json()
    cfg = {
        **VALID,
        "source": {"video": "/elsewhere.mp4", "topic": "/cam/image_raw", "start": 0.5},
    }
    resp = api.post("/api/runs", json={"config": cfg, "video_id": record["id"]})
    assert resp.status_code == 200, resp.text
    job = api.app.state.jobs.get(resp.json()["job_id"])
    assert job.video_id == record["id"]
    assert job.config["source"] == {
        "mcap": record["original"],
        "topic": "/cam/image_raw",
        "start": 0.5,
    }
    assert api.get(f"/api/jobs/{job.id}").json()["video"] == "drive.mcap"
    # the topic is required and must be an image topic of the file
    for source, needle in (
        ({}, "topic"),
        ({"topic": "/cam/camera_info"}, "/cam/camera_info"),
        ({"topic": "/nope"}, "/nope"),
    ):
        resp = api.post(
            "/api/runs",
            json={"config": {**VALID, "source": source}, "video_id": record["id"]},
        )
        assert resp.status_code == 422, resp.text
        error = resp.json()["errors"][0]
        assert error["path"] == "source.topic" and needle in error["message"]
        assert "/cam/image_raw" in error["message"]  # names the choices
    # a bare source.mcap follows the rule of a bare source.video
    bare = {**VALID, "source": {"mcap": str(tmp_path / "drive.mcap"), "topic": "/c"}}
    resp = api.post("/api/runs", json={"config": bare})
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["path"] == "source.mcap"
    inside = recording(web_paths.data_roots[0] / "in.mcap")
    bare = {**VALID, "source": {"mcap": str(inside), "topic": "/cam/image_raw"}}
    assert api.post("/api/runs", json={"config": bare}).status_code == 200


def test_run_with_rectify_but_topic_without_camera_info_is_422(api, tmp_path):
    record = upload(api, recording(tmp_path / "drive.mcap")).json()

    def submit(topic: str, **sections):
        cfg = {**VALID, "source": {"topic": topic}, **sections}
        return api.post("/api/runs", json={"config": cfg, "video_id": record["id"]})

    rectify = {"rectify": {"enabled": True}}
    from_source = {"camera": {"calibration": {"from_source": True}}}
    resp = submit("/front/image_raw/compressed", **rectify)
    assert resp.status_code == 422, resp.text
    error = resp.json()["errors"][0]
    assert error["path"] == "rectify.enabled"
    assert "CameraInfo" in error["message"]
    resp = submit("/rect/image_raw", **rectify)  # already rectified
    assert resp.status_code == 422
    error = resp.json()["errors"][0]
    assert error["path"] == "rectify.enabled" and "rectified" in error["message"]
    resp = submit("/front/image_raw/compressed", **from_source)
    assert resp.status_code == 422
    error = resp.json()["errors"][0]
    assert error["path"] == "camera.calibration.from_source"
    assert "CameraInfo" in error["message"]
    # topics with what each option needs are accepted
    assert submit("/cam/image_raw", **rectify).status_code == 200
    assert submit("/rect/image_raw", **from_source).status_code == 200


def test_video_routes_unchanged(api, video_factory):
    clip = video_factory("road.mp4")
    record = api.post(
        "/api/videos", files={"file": ("road.mp4", clip.read_bytes(), "video/mp4")}
    ).json()
    assert record["kind"] == "video" and record["mcap"] is None
    assert "topics" not in record["urls"]
    assert api.get(f"/api/videos/{record['id']}/topics").status_code == 404
    assert api.get("/api/videos/0123456789ab/topics").status_code == 404
    resp = api.post("/api/runs", json={"config": VALID, "video_id": record["id"]})
    assert resp.status_code == 200, resp.text
    job = api.app.state.jobs.get(resp.json()["job_id"])
    assert job.config["source"] == {"video": record["original"]}


def test_old_record_without_kind_loads_as_video(web_paths, video_factory):
    library = Library(web_paths.library_root)
    try:
        record = library.add(video_factory("old.mp4").open("rb"), "old.mp4")
        library.wait_previews(60)
        path = web_paths.library_root / record.id / "record.json"
        data = json.loads(path.read_text())
        assert data["kind"] == "video"
        del data["kind"]
        data.pop("mcap", None)
        path.write_text(json.dumps(data))
        old = library.get(record.id)
        assert old.kind == "video" and old.mcap is None and not old.missing
        assert [r.id for r in library.list()] == [record.id]
        # an unknown kind, or an mcap entry without its summary, is not loaded
        path.write_text(json.dumps({**data, "kind": "lidar"}))
        assert library.list() == []
        path.write_text(json.dumps({**data, "kind": "mcap"}))
        assert library.list() == []
    finally:
        library.close()


def test_rerun_and_reopen_of_an_mcap_run(api, web_paths, tmp_path):
    record = upload(api, recording(tmp_path / "drive.mcap")).json()
    cfg = {
        **VALID,
        "source": {"mcap": record["original"], "topic": "/cam/image_raw"},
        "output": {"tag": "web-a"},
    }
    run = make_run(web_paths.out_root, "20260101_000000_a", NEW_HEADER, [], {}, cfg)
    (run / "web_run.json").write_text(
        json.dumps({"status": "done", "video_id": record["id"]})
    )
    reopened = api.post(f"/api/runs/{run.name}/reopen").json()
    assert reopened["config"]["source"] == cfg["source"]
    resp = api.post(f"/api/runs/{run.name}/rerun")
    assert resp.status_code == 200, resp.text
    job = api.app.state.jobs.get(resp.json()["job_id"])
    assert job.video_id == record["id"]
    assert job.config["source"] == {
        "mcap": record["original"],
        "topic": "/cam/image_raw",
    }
    assert api.get(f"/api/runs/{run.name}").json()["video"] == "drive.mcap"
    # a run of a bare server MCAP reruns from an import folder only
    inside = recording(web_paths.data_roots[0] / "in.mcap")
    bare = {**cfg, "source": {"mcap": str(inside), "topic": "/cam/image_raw"}}
    (run / "config.yaml").write_text(json.dumps(bare))
    (run / "web_run.json").write_text(json.dumps({"status": "done"}))
    resp = api.post(f"/api/runs/{run.name}/rerun")
    assert resp.status_code == 200, resp.text
    job = api.app.state.jobs.get(resp.json()["job_id"])
    assert job.video_id is None and job.config["source"]["mcap"] == str(inside)
    outside = {**cfg, "source": {"mcap": str(tmp_path / "drive.mcap"), "topic": "/c"}}
    (run / "config.yaml").write_text(json.dumps(outside))
    resp = api.post(f"/api/runs/{run.name}/rerun")
    assert resp.status_code == 409 and "import" in resp.json()["detail"]


def realsense_like(path: Path, color: bool = True) -> Path:
    """A 16UC1 depth topic that sorts before the colour topic, a
    compressedDepth topic, and (optionally) the rgb8 colour topic."""
    depth = [np.full((12, 16), 1000, np.uint16)] * 2
    images = [
        ImageSpec(
            topic="/camera/aligned_depth_to_color/image_raw",
            frames=depth,
            times_ns=[0, 100_000_000],
            encoding="16UC1",
        ),
        ImageSpec(
            topic="/camera/aligned_depth_to_color/image_raw/compressedDepth",
            frames=frames(2),
            times_ns=[0, 100_000_000],
            compressed="png",
            format="16UC1; compressedDepth png",
            data_prefix=bytes(12),
        ),
    ]
    if color:
        images.append(
            ImageSpec(
                topic="/camera/color/image_raw",
                frames=frames(3, 8, 12),
                times_ns=[0, 500_000_000, 1_000_000_000],
            )
        )
    return write_mcap(path, images=images)


def test_unsupported_first_topic_does_not_reject_the_file(api, tmp_path):
    resp = upload(api, realsense_like(tmp_path / "rs.mcap"))
    assert resp.status_code == 200, resp.text
    record = resp.json()
    # the record and thumb come from the first supported topic (colour)
    assert (record["width"], record["height"]) == (12, 8)
    assert record["fps"] == pytest.approx(2.0)
    assert record["duration_s"] == pytest.approx(1.0)
    thumb = api.get(f"/api/videos/{record['id']}/thumb")
    assert thumb.status_code == 200 and thumb.content[:2] == b"\xff\xd8"
    topics = by_name(api.get(f"/api/videos/{record['id']}/topics").json())
    raw16 = topics["/camera/aligned_depth_to_color/image_raw"]
    assert raw16["supported"] is False
    assert "16UC1" in raw16["unsupported_reason"]
    comp = topics["/camera/aligned_depth_to_color/image_raw/compressedDepth"]
    assert comp["supported"] is False and comp["encoding"] == "compressedDepth"
    color = topics["/camera/color/image_raw"]
    assert color["supported"] is True and color["unsupported_reason"] is None
    # choosing an unsupported topic is refused early, with the reason
    for extra in ({}, {"rectify": {"enabled": True}}):
        cfg = {
            **VALID,
            "source": {"topic": "/camera/aligned_depth_to_color/image_raw"},
            **extra,
        }
        resp = api.post("/api/runs", json={"config": cfg, "video_id": record["id"]})
        assert resp.status_code == 422, resp.text
        error = resp.json()["errors"][0]
        assert error["path"] == "source.topic" and "16UC1" in error["message"]
    cfg = {**VALID, "source": {"topic": "/camera/color/image_raw"}}
    resp = api.post("/api/runs", json={"config": cfg, "video_id": record["id"]})
    assert resp.status_code == 200, resp.text


def test_mcap_with_only_unsupported_topics_is_422(api, web_paths, tmp_path):
    resp = upload(api, realsense_like(tmp_path / "depth.mcap", color=False))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "no decodable image topic" in detail and "16UC1" in detail
    assert api.get("/api/videos").json() == []
    assert files_under(web_paths.library_root) == []


def test_camera_info_of_another_size_is_reported_and_refused(api, tmp_path):
    rec = write_mcap(
        tmp_path / "size.mcap",
        images=[ImageSpec(topic="/cam/image_raw", frames=frames(2), times_ns=[0, 1])],
        camera_infos=[
            CameraInfoSpec(topic="/cam/camera_info", width=32, height=20, k=K, d=RAW_D)
        ],
    )
    record = upload(api, rec).json()
    cam = by_name(record["mcap"])["/cam/image_raw"]
    assert cam["has_camera_info"] is True and cam["has_distortion"] is False
    assert cam["camera_info_error"]
    for section, path in (
        ({"rectify": {"enabled": True}}, "rectify.enabled"),
        (
            {"camera": {"calibration": {"from_source": True}}},
            "camera.calibration.from_source",
        ),
    ):
        cfg = {**VALID, "source": {"topic": "/cam/image_raw"}, **section}
        resp = api.post("/api/runs", json={"config": cfg, "video_id": record["id"]})
        assert resp.status_code == 422, resp.text
        error = resp.json()["errors"][0]
        assert error["path"] == path and "cannot be used" in error["message"]


def test_single_message_topic_is_accepted_with_fps_zero(api, tmp_path):
    rec = write_mcap(
        tmp_path / "one.mcap",
        images=[ImageSpec(topic="/cam/image_raw", frames=frames(1), times_ns=[5])],
    )
    resp = upload(api, rec)
    assert resp.status_code == 200, resp.text
    record = resp.json()
    assert record["fps"] == 0.0 and record["duration_s"] == 0.0
    assert by_name(record["mcap"])["/cam/image_raw"]["count"] == 1


def test_oversized_compressed_image_is_422(api, tmp_path):
    def huge(data: bytes) -> bytes:  # IHDR declares 60000 x 60000
        return data[:16] + (60000).to_bytes(4, "big") * 2 + data[24:]

    path = write_mcap(
        tmp_path / "bomb.mcap",
        images=[
            ImageSpec(
                topic="/cam/image_raw/compressed",
                frames=frames(1),
                times_ns=[0],
                compressed="png",
                data_patch=huge,
            )
        ],
    )
    resp = upload(api, path)
    assert resp.status_code == 422, resp.text
    assert "above the cap" in resp.json()["detail"]
