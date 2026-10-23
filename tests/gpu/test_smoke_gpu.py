"""Smoke of every real adapter through the CLI on 5 frames of IMG_4021
(marked gpu by conftest). One fresh interpreter per example config, missing
weights skip with their download hint (nothing is downloaded), outputs
checked by tests/run_outputs.check_run_outputs.

The focal is fixed (camera.focal_px) so no estimation model runs: focal
estimation is covered by the task_0001 regression. Weights are checked at
the paths the example configs use (the run reads those, not the
DDD_* overrides of conftest).

The light example (yolo_light_da3, task_0010) needs its own weights, kept only
on the laptop (DA3 and SegFormer-B0 in weights/hf): it skips on gpu-host and,
on the laptop, runs with DDD_HF_CACHE=weights/hf.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from run_outputs import check_run_outputs, single_run_dir

REPO = Path(__file__).resolve().parents[2]
N_FRAMES = 5
YOLO = Path("~/dev/ultralytics/_testbed/weights")
SEGFORMER = ("nvidia/segformer-b5-finetuned-cityscapes-1024-1024", "pytorch_model.bin")
TWINLITENET = Path("~/dev/TwinLiteNetPlus/pretrained/large.pth")
TWINLITENET_SMALL = TWINLITENET.with_name("small.pth")
SEGFORMER_B0 = (
    "nvidia/segformer-b0-finetuned-cityscapes-640-1280",
    "pytorch_model.bin",
)
DA3_METRIC = ("depth-anything/DA3METRIC-LARGE", "model.safetensors")  # in weights/hf

# case -> (example config, extra --set overrides, assets the run needs besides DA3)
GPU_SMOKE_CASES: dict[
    str, tuple[str, tuple[str, ...], tuple[Mapping[str, Any], ...]]
] = {
    "rfdetr_da3": ("rfdetr_da3.yaml", (), ()),
    "yolo_o365_da3": (
        "yolo_o365_da3.yaml",
        (),
        ({"path": YOLO / "yolo26l-objv1-150.pt"},),
    ),
    "multi_detector_fused": (
        "multi_detector_fused.yaml",
        (),
        ({"path": YOLO / "yolo26l-objv1-150.pt"},),
    ),
    # seg-nano: never let rfdetr download the large Seg weights on its own.
    "rfdetr_seg_da3": (
        "rfdetr_seg_da3.yaml",
        ("detectors.0.variant=seg-nano",),
        ({"detector": {"type": "rfdetr", "variant": "seg-nano"}},),
    ),
    "yolo_seg_da3": ("yolo_seg_da3.yaml", (), ({"path": YOLO / "yolo26l-seg.pt"},)),
    "rfdetr_sam2_da3": ("rfdetr_sam2_da3.yaml", (), ({"refiner": "sam2"},)),
    "rfdetr_semantic_da3": ("rfdetr_semantic_da3.yaml", (), ({"hf": SEGFORMER},)),
    "rfdetr_lanes_scene_da3": (
        "rfdetr_lanes_scene_da3.yaml",
        (),
        ({"path": TWINLITENET}, {"hf": SEGFORMER}),
    ),
    "rfdetr_da3_track": ("rfdetr_da3_track.yaml", (), ()),
    "rfdetr_da3_motion": ("rfdetr_da3_motion.yaml", (), ()),
    "rfdetr_sam2_lanes_motion_da3": (
        "rfdetr_sam2_lanes_motion_da3.yaml",
        (),
        ({"refiner": "sam2"}, {"path": TWINLITENET}),
    ),
    # Light stack for a 4 GB GPU (task_0010): weights only on the laptop, skips elsewhere.
    "yolo_light_da3": (
        "yolo_light_da3.yaml",
        (),
        (
            {"path": YOLO / "yolo26s-seg.pt"},
            {"path": TWINLITENET_SMALL},
            {"hf": SEGFORMER_B0},
            {"hf": DA3_METRIC},
        ),
    ),
}


@pytest.mark.parametrize("case", list(GPU_SMOKE_CASES))
def test_real_adapters_end_to_end(
    case, video, hf_cache, require_assets, run_cli_subprocess, tmp_path
):
    """Exit 0 and a structurally complete run folder (hf_cache: skips without
    the DA3 checkpoints every case needs)."""
    config, overrides, assets = GPU_SMOKE_CASES[case]
    for asset in assets:
        require_assets(asset)
    argv = [str(video), "--config", f"configs/{config}"]
    for override in (
        *overrides,
        f"source.max_frames={N_FRAMES}",
        "camera.focal_px=1390",
        f"output.out_root={tmp_path}",
    ):
        argv += ["--set", override]
    run_cli_subprocess(argv, cwd=REPO)
    out = check_run_outputs(single_run_dir(tmp_path))
    assert out.metrics["frames_processed"] == N_FRAMES
