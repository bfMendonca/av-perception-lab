# det-depth-distance — Project Description

Comprehensive project description maintained for external LLMs. Keep this
file up to date in the same commit as any structural change (see
`AGENTS.md` § "Manter DESCRIPTION.md Atualizado").

## 1. What the Project Is

A research prototype that combines an object detector (Roboflow RF-DETR) and
a monocular metric depth estimator (ByteDance Depth Anything 3, "DA3") on
the same video frames. For every detected bounding box, the depth pixels
inside the box are aggregated with robust statistics (mean, median,
variance, percentiles, outlier rejection) to estimate the object's distance
to the camera. The intended output is an annotated video (boxes plus
distance) and a per-detection CSV. Accuracy matters more than speed: both
models run in PyTorch, frame by frame. The user is a computer-vision
researcher working on agricultural robotics; first test data is iPhone
driving footage (1920×1080 HEVC).

Current state: the first pipeline (task_0001) is implemented: a CLI that
processes a video frame by frame with both models and writes an annotated
video, a per-box CSV and a metrics JSON.

## 2. Technology Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 (`requires-python >=3.11,<3.12`) |
| Env / deps | Astral uv, `pyproject.toml` + versioned `uv.lock` |
| Base deps | numpy<2 (required by depth-anything-3), opencv-python, pillow |
| `infer` extra (GPU host only) | torch 2.8.0 + torchvision 0.23.0 + xformers 0.0.32.post2 from the PyTorch cu128 index; `rfdetr` @ git 1f2af2ea; `depth-anything-3` @ git 3d835ec; `addict` (imported by depth-anything-3 but not declared by it) |
| Dev group | pytest, ruff |
| Video I/O | system ffmpeg/ffprobe |
| Inference host | `gpu-host`: NVIDIA RTX PRO 6000 Blackwell 96 GB (sm_120), no sudo |
| CI / Docker | none |

## 3. Project Structure

```
det-depth-distance/
├── pyproject.toml           # uv manifest; "infer" extra holds the GPU stack
├── uv.lock
├── src/det_depth_distance/
│   ├── cli.py               # argparse, per-frame loop, outputs (torch imported inside run())
│   ├── detector.py          # RF-DETR wrapper -> Detections (numpy), class filter
│   ├── depth.py             # DA3 wrapper -> DepthFrame in metres; geometry/focal helpers
│   ├── distance.py          # pure numpy: box -> map pixels -> robust stats -> distance
│   ├── render.py            # PIL boxes/labels/HUD, colourised depth panel + legend
│   ├── video_io.py          # ffprobe / ffmpeg pipes
│   └── __main__.py          # python -m det_depth_distance
├── tests/                   # pytest, one file per module, no GPU needed
├── utils/sync_gpu_host.sh    # rsync push code / pull outputs with the GPU host
├── docs/                    # task files (docs/task_DDDD.md) + template
└── _testbed/                # git-ignored: test videos, outputs, logs
```

## 4. Data Model

No database. Runtime value objects (frozen dataclasses):

- `Detections` (detector.py): xyxy (N,4) in frame pixels, confidence, class_id, class_name.
- `DepthFrame` (depth.py): depth_m (map_h, map_w) float32 metres, `CameraIntrinsics` at map resolution, optional sky mask.
- `MapGeometry` (distance.py): per-axis affine frame -> depth map (sx, sy, ox, oy).
- `DistanceConfig` / `BoxDepthStats` (distance.py): parameters and per-box statistics (n_valid, n_inliers, mean, median, std, var, p10..p90, min, max, distance or None).
- `VideoInfo` / `VideoFrame` (video_io.py), `BoxAnnotation` (render.py), `FrameResult` (cli.py).

Per run: `video.mp4`, `detections.csv` (one row per box), `metrics.json`, optional `depth.npz`.

## 5. Architecture Patterns

- Two environments from one manifest: a light local env (numpy/OpenCV) for
  developing and unit-testing pure-numpy logic, and the heavy `infer` extra
  on the GPU host. Heavy imports (`torch`, `rfdetr`, `depth_anything_3`)
  must stay inside the modules that need them.
- No git remote: code reaches the GPU host via rsync (`utils/sync_gpu_host.sh`).
- DA3METRIC-LARGE output is not in metres: `metres = raw * (fx + fy) / 2 / 300`,
  with fx/fy at the depth-map resolution. The focal comes from `--focal-px`
  (frame pixels) or is estimated once per video by an any-view DA3 model
  (median over N frames) before the main models load. Model kind is decided
  from the checkpoint's `model_name`, not `Prediction.is_metric`.
- `distance.py` is pure numpy; outlier filters and statistics are registries
  (`_OUTLIER_FILTERS`, `_STATS`). `cli.process_frame` depends on
  `DetectorLike` / `DepthLike` Protocols so it is tested with fakes.
- `depth.da3_map_geometry` replicates DA3's private InputProcessor sizing;
  `DepthEstimator` checks the real depth shape against it on the first frame.
- DA3 is called with one frame per `inference()` call (a list switches
  any-view models to multi-view mode).

## 6. Authentication & Authorization

None. Local CLI tool; no users, sessions or network services.

## 7. URL / API Design

No network API. The interface is the CLI `det-depth-distance` /
`python -m det_depth_distance VIDEO [options]`; options are grouped as
input/output, detector, depth, distance, rendering and execution (see the
README table or `--help`).

## 8. Key Business Concepts

- **Metric depth**: per-pixel distance in metres along the optical axis (z).
- **Euclidean range**: straight-line camera-to-point distance; larger than z
  for off-axis points.
- **Focal length (px)**: camera intrinsic needed to convert DA3METRIC output
  to metres; either calibrated or estimated by an any-view DA3 model.
- **Box shrink / outlier rejection**: techniques to suppress background
  pixels inside a non-tight bounding box.

## 9. Security Model

No secrets are read by the code. Hugging Face model downloads use the
user's existing token cache on the GPU host, outside this repository.

## 10. Development & Deployment

- Local: `uv sync`, `uv run pytest`, `uv run ruff check . && uv run ruff format --check .`
- GPU host: `utils/sync_gpu_host.sh push`, then on `gpu-host`
  `~/.local/bin/uv sync --extra infer`; outputs fetched with
  `utils/sync_gpu_host.sh pull`.
- No deployment target beyond running the CLI on the GPU host.

## 11. Conventions Summary

- Code, docstrings, comments in English; task files, AGENTS.md, CLAUDE.md,
  README in pt-BR; CHANGELOG in pt-BR ASCII.
- Ruff defaults; snake_case functions/modules, PascalCase classes.
- Conventional Commits with pt-BR subject; scopes: core, tests, utils, deps, docs.
- Never commit to `main`; work on `feat/`/`fix/` branches.

## 12. Task Documentation & Lifecycle (docs/)

This project uses task-driven development: every implementation has a
`docs/task_DDDD.md` file (Descrição, Escopo, Planejamento, Análise de
Impacto, Verificação, Dependências) created before coding. On completion the
title is marked `(CONCLUÍDA)` and the file gains `## Detalhes de
Implementação` and `## Caderno de Testes` sections in the same commit as the
implementation. See `AGENTS.md` for the mandatory rules.

## 13. What This Project Does NOT Have

- No TensorRT/ONNX path, no batching, no real-time optimisation.
- No segmentation masks (boxes only, for now), no object tracking or temporal smoothing.
- No ground-truth distance evaluation.
- No CI, no Docker, no git remote, no web UI or API.
