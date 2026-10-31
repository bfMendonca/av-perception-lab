"""Fixtures of the web backend tests (need the `web` extra)."""

import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from det_depth_distance.web.app import create_app
from det_depth_distance.web.paths import WebPaths

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "out", tmp_path / "library", (tmp_path / "data",))


@pytest.fixture
def client(web_paths: WebPaths) -> TestClient:
    return TestClient(create_app(web_paths))


@pytest.fixture
def recommended_yaml() -> str:
    return (REPO / "configs/rfdetr_sam2_lanes_motion_da3.yaml").read_text()


def make_video(path: Path, seconds: float = 2.0, fps: int = 10, size="64x48", **kw):
    """Tiny synthetic H.264 clip (a few KB) written with the ffmpeg CLI.

    `fmt="h264"` writes a raw Annex B stream (as rpicam-vid does); `faststart`
    moves the moov atom to the front of an MP4.
    """
    fmt = kw.get("fmt")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi"]
    cmd += ["-i", f"testsrc=size={size}:rate={fps}", "-t", str(seconds)]
    cmd += ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", str(fps)]
    if kw.get("faststart"):
        cmd += ["-movflags", "+faststart"]
    if fmt == "h264":
        cmd += ["-bf", "0"]  # rpicam-vid streams have no B-frames
    if fmt:
        cmd += ["-f", fmt]
    subprocess.run([*cmd, str(path)], check=True, capture_output=True)
    return path


@pytest.fixture
def video_factory(tmp_path: Path):
    """`video_factory(name_or_path, **kw)` -> a tiny clip; a bare name goes
    under tmp/src/, a path is used as is (parents created)."""

    def make(target: str | Path = "clip.mp4", **kw) -> Path:
        path = Path(target)
        if not path.is_absolute():
            path = tmp_path / "src" / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return make_video(path, **kw)

    return make
