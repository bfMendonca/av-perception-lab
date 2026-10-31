"""Worker: JSON lines, effective config, exit codes (in process, fake adapters)."""

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")

SMOKE = Path(__file__).resolve().parents[1] / "smoke"
sys.path.insert(0, str(SMOKE))

from smoke_fakes import DepthOracle, install_scene_fakes
from smoke_scene import SCENE, scene_frames, write_scene_video

from det_depth_distance.core.types import CameraIntrinsics
from det_depth_distance.web import worker

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def lines(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


@pytest.fixture
def scene_config(tmp_path, monkeypatch):
    focal = 725.0 / 3
    oracle = DepthOracle(SCENE, CameraIntrinsics(focal, focal, 160.0, 90.0))
    install_scene_fakes(monkeypatch, oracle)
    video = write_scene_video(SCENE, tmp_path / "scene.mp4", scene_frames(SCENE))
    data = {
        "device": "cpu",
        "source": {"video": str(video)},
        "detectors": [{"type": "scene_det", "name": "box", "pad_px": 1}],
        "depth": {"type": "da3", "model": "da3metric-large", "process_res": 154},
        "camera": {"focal_px": focal, "estimate": None},
        "output": {"out_root": str(tmp_path / "out"), "tag": "w"},
    }
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(data))
    return path, data, tmp_path / "out"


def test_worker_emits_json_lines_and_writes_effective_config(
    scene_config, capsys, monkeypatch
):
    path, _data, out_root = scene_config
    monkeypatch.setenv(worker.VIDEO_ID_ENV, "abc123")
    assert worker.main([str(path)]) == 0
    events = lines(capsys)  # stdout holds only JSON lines
    assert [e["event"] for e in events] == [
        "started",
        *["frame"] * SCENE.n_frames,
        "done",
    ]
    assert [e["n"] for e in events[1:-1]] == list(range(1, SCENE.n_frames + 1))
    run_dir = Path(events[0]["run_dir"])
    assert run_dir.parent == out_root
    assert events[-1] == {
        "event": "done",
        "run_dir": str(run_dir),
        "frames": SCENE.n_frames,
    }
    effective = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert effective["detectors"][0]["name"] == "box"
    assert effective["output"]["tag"] == "w"
    meta = json.loads((run_dir / "web_run.json").read_text())
    assert (meta["status"], meta["video_id"]) == ("done", "abc123")
    assert meta["started_at"]
    assert (run_dir / "preview.jpg").stat().st_size > 0
    assert (run_dir / "metrics.json").is_file()


def test_worker_exit_2_on_config_error(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("detectors: 3\n")
    assert worker.main([str(bad)]) == 2
    (event,) = lines(capsys)
    assert event["event"] == "error" and event["message"]


def test_worker_start_past_end_is_a_readable_error(scene_config, capsys):
    path, data, out_root = scene_config
    data["source"]["start"] = 3600.0
    path.write_text(yaml.safe_dump(data))
    assert worker.main([str(path)]) == 2
    events = lines(capsys)
    assert events[-1]["event"] == "error"
    assert "no frames decoded" in events[-1]["message"]
    assert "source.start" in events[-1]["message"]
    (run_dir,) = out_root.iterdir()  # kept, flagged as failed
    assert json.loads((run_dir / "web_run.json").read_text())["status"] == "failed"


def test_worker_unexpected_error_exits_1(scene_config, capsys, monkeypatch):
    path, _, _ = scene_config

    def boom(cfg, on_frame=None):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(worker.cli, "run", boom)
    assert worker.main([str(path)]) == 1
    (event,) = lines(capsys)
    assert event == {"event": "error", "message": "RuntimeError: CUDA out of memory"}
