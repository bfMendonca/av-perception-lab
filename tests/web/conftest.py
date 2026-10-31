"""Fixtures of the web backend tests (need the `web` extra)."""

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
