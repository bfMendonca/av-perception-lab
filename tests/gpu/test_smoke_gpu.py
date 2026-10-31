"""Smoke of every real adapter through the CLI on 5 frames of IMG_4021
(marked gpu by conftest). One fresh interpreter per example config; a config
whose models are not all ready under the weights root skips naming
`--prepare` (nothing is downloaded by the tests; readiness comes from the
models' own declarations, task_0021); outputs checked by
tests/run_outputs.check_run_outputs.

The focal is fixed (camera.focal_px) so no estimation model runs: focal
estimation is covered by the task_0001 regression.
"""

from pathlib import Path

import pytest
from run_outputs import check_run_outputs, single_run_dir

REPO = Path(__file__).resolve().parents[2]
N_FRAMES = 5

# case -> (example config, extra --set overrides)
GPU_SMOKE_CASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "rfdetr_da3": ("rfdetr_da3.yaml", ()),
    "yolo_o365_da3": ("yolo_o365_da3.yaml", ()),
    "multi_detector_fused": ("multi_detector_fused.yaml", ()),
    # seg-nano: the small Seg checkpoint is enough for a smoke.
    "rfdetr_seg_da3": ("rfdetr_seg_da3.yaml", ("detectors.0.variant=seg-nano",)),
    "yolo_seg_da3": ("yolo_seg_da3.yaml", ()),
    "rfdetr_sam2_da3": ("rfdetr_sam2_da3.yaml", ()),
    "rfdetr_semantic_da3": ("rfdetr_semantic_da3.yaml", ()),
    "rfdetr_lanes_scene_da3": ("rfdetr_lanes_scene_da3.yaml", ()),
    "rfdetr_da3_track": ("rfdetr_da3_track.yaml", ()),
    "rfdetr_da3_motion": ("rfdetr_da3_motion.yaml", ()),
    "rfdetr_sam2_lanes_motion_da3": ("rfdetr_sam2_lanes_motion_da3.yaml", ()),
    # Light stack for a 4 GB GPU (task_0010).
    "yolo_light_da3": ("yolo_light_da3.yaml", ()),
}


@pytest.mark.parametrize("case", list(GPU_SMOKE_CASES))
def test_real_adapters_end_to_end(
    case, video, require_ready, run_cli_subprocess, tmp_path
):
    """Exit 0 and a structurally complete run folder (skips until every model
    of the config is ready under the weights root: run --prepare)."""
    config, overrides = GPU_SMOKE_CASES[case]
    overrides = (
        *overrides,
        f"source.max_frames={N_FRAMES}",
        "camera.focal_px=1390",
        f"output.out_root={tmp_path}",
    )
    require_ready(f"configs/{config}", overrides)
    argv = [str(video), "--config", f"configs/{config}"]
    for override in overrides:
        argv += ["--set", override]
    run_cli_subprocess(argv, cwd=REPO)
    out = check_run_outputs(single_run_dir(tmp_path))
    assert out.metrics["frames_processed"] == N_FRAMES
