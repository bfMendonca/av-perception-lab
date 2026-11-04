# av-perception-lab

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![uv](https://img.shields.io/badge/packaging-uv-de5fe9.svg)](https://docs.astral.sh/uv/)

A perception test bed for autonomous driving: pluggable detectors, metric depth, segmentation, tracking and motion prediction, with per-object distance, camera calibration and evaluation against field references.

![Six seconds of a highway recording processed by the full pipeline: the annotated frame with per-object distance and relative speed, the metric depth map with its legend in meters, and a bird's-eye view inset.](docs/assets/demo-highway.gif)

*Six seconds of a highway recording, processed with RF-DETR, SAM 2.1, TwinLiteNet+ and Depth Anything 3. Left: the annotated frame with the distance and relative speed of each tracked vehicle, and a bird's-eye view in the corner. Right: the metric depth map.*

A research prototype for studying how detection, depth, segmentation and tracking models chain into a perception stack, using only open-source models. Every stage is a plug-in component that you pick in a YAML file or in the web interface. For each detected object it reports the distance with an uncertainty; for tracked objects it also reports the relative speed and a predicted trajectory.

![The pipeline editor with configs/rfdetr_sam2_lanes_motion_da3.yaml loaded: a video with a three-second segment selected, the pipeline stages from source to semantic segmentation, and the Kalman smoothing settings open in the side panel.](docs/assets/ui-editor.png)

## Requirements

| Tool | Check | Needed for |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | `uv --version` | everything; it installs Python 3.11 and the packages |
| git | `git --version` | cloning; Prepare also clones a pinned copy of TwinLiteNet+ |
| ffmpeg and ffprobe | `ffmpeg -version` | decoding and writing video |
| Node.js 22.12 or newer, with npm | `node --version` | building the web interface once (`utils/run_web.sh`) |
| NVIDIA GPU and a driver for CUDA 12.8 | `nvidia-smi` | running the models (steps 3 and 4 below) |

The repository has no sample video. Use any MP4, MOV or MKV that ffmpeg can decode.

## Quick start

Clone the repository and run every command from its root.

### 1. Check the install (no GPU, about a minute)

```bash
uv sync
uv run pytest -q
```

The run ends with a summary such as `898 passed, 9 skipped, 48 deselected` (the counts change as tests are added). To list the detectors, depth models and other components with their options:

```bash
uv run av-perception-lab --list-models
```

### 2. Open the web interface (no GPU)

```bash
utils/run_web.sh
```

The first run builds the interface (needs Node.js 22.12 or newer, about a minute) and installs the `web` extra, then starts the server. Open <http://127.0.0.1:8000>. Stop it with Ctrl-C; add `--port 8010` to change the port. Click **Open YAML** and choose `configs/rfdetr_da3.yaml`. Every model shows **Not ready**, because PyTorch is not installed in this environment. Step 3 fixes that. The **Presets** button lists ready-made pipelines (the files in `configs/`) and your own saved ones: loading a preset keeps the selected video and output folder, and **Save as preset...** stores the current pipeline on the server (default folder `_testbed/presets/`, option `--presets-root`).

### 3. Process a video (NVIDIA GPU)

With a GPU, skip step 2 and start the interface like this:

```bash
utils/run_web.sh --gpu
```

The first run also installs the GPU stack: about 9 GB of Python packages. Later runs find it and keep it, so `--gpu` is only needed once. If you run `uv` yourself afterwards, always pass `--extra infer --extra web`: a plain `uv sync` removes the GPU stack.

In the page at <http://127.0.0.1:8000>:

1. **Open YAML**, then `configs/rfdetr_da3.yaml`.
2. Click **Prepare all** and confirm. This downloads about 3.1 GB into `weights/`: RF-DETR (136 MB), Depth Anything 3 metric (1.3 GB) and the Depth Anything 3 model that estimates the focal length (1.6 GB). The cards turn solid when each model is ready.
3. Drop your video on the page. The default segment is the first 3 seconds.
4. Click **Process**. A 3-second clip takes about 12 seconds on an RTX 3070 (1296x972, 25 fps), including model loading.

The annotated video, `detections.csv` and `metrics.json` appear under **Runs** and in `_testbed/outputs/<timestamp>_<video>_<detector>_<depth>/`.

The script runs `(cd web && npm ci && npm run build)` when `web/dist` is missing, then `uv run --extra infer --extra web av-perception-lab-web` (only `--extra web` without the GPU stack). `utils/run_web.sh --dry-run` prints what it would run.

### 4. The same from the command line

```bash
uv run --extra infer av-perception-lab --config configs/rfdetr_da3.yaml --prepare
```

It prints a table ending in `ready` for each model and exits with code 0. Then process the first 3 seconds of a video:

```bash
uv run --extra infer av-perception-lab --config configs/rfdetr_da3.yaml my-video.mp4 --set source.duration=3
```

The last line looks like `[done] 76 frames, 319 boxes (0 n/a) in 9.8 s -> _testbed/outputs/<folder>`.

### If it fails

- `error: 3 model(s) not ready under the weights root` (exit code 2): the weights are not downloaded. Run the `--prepare` command above, or use **Prepare all** in the interface.
- `package torch is not installed (extra 'infer')` in a model's status: you started the interface without the GPU stack. Run `utils/run_web.sh --gpu`.
- `error: building the web interface needs Node.js 22.12 or newer`: install Node.js from <https://nodejs.org/> and run the script again.
- `400 Bad Request` when opening the page from another computer: the server only accepts requests addressed to localhost. Start it with `--host 0.0.0.0 --allowed-host <the address you type in the browser>` and read the security notes in [the web interface reference](docs/reference.md#interface-web-segurança-rede-local-e-desligamento) first: the interface has no authentication.

## Documentation

- [Model weights and Prepare](docs/model_environments.md): where weights come from, how a model declares them, how to add a model, and how to resolve dependency conflicts.
- [Extending the web interface](docs/web_extending.md): a new model or pipeline stage shows up in the interface without editing it.
- [Detailed reference](docs/reference.md): YAML configuration, motion estimation, camera calibration, evaluation against field references, outputs and how the distance is computed. It is the original long README, still in Portuguese.
- [`configs/`](configs): ready-to-run pipelines. [CHANGELOG](CHANGELOG.md), [DESCRIPTION](DESCRIPTION.md) and [AGENTS](AGENTS.md) cover history, a project summary for other tools, and contribution rules.

## Status and limits

This is a research prototype, not a validated perception system. On an RTX 3070 (8 GB) with a 1296x972, 25 fps video it runs at about 7 frames per second with `configs/rfdetr_da3.yaml` (RF-DETR large and Depth Anything 3) and at about 2 to 3 frames per second with `configs/rfdetr_sam2_lanes_motion_da3.yaml` (adding SAM 2.1 masks, lane segmentation, tracking and motion). The web interface has no authentication and listens on `127.0.0.1` unless you change it.

## License

The code is licensed under the [Apache License 2.0](LICENSE) (see also [NOTICE](NOTICE)). To cite it, use [CITATION.cff](CITATION.cff).

Model weights are downloaded by Prepare and are not part of this repository. They keep their own licences: RF-DETR, Depth Anything 3 (`DA3METRIC-LARGE`, `DA3-LARGE-1.1`) and SAM 2.1 are Apache-2.0; the YOLO26 weights are AGPL-3.0; the SegFormer Cityscapes checkpoints carry an NVIDIA licence; `DA3-LARGE`, `DA3-GIANT` and the nested DA3 models are CC-BY-NC-4.0; TwinLiteNet+ is MIT and its weight files have no licence of their own, so MIT is assumed. Sources and licences per model are in [`docs/task_0021.md`](docs/task_0021.md).

The `infer` extra installs `ultralytics` (AGPL-3.0), which the YOLO adapters import. Making it an optional extra is planned.
