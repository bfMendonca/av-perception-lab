# MCAP input — design

Date: 2026-11-04. Status: draft, awaiting approval. Branch for the work: `feat/mcap_input` (task file `docs/task_0033.md` to be written first, per AGENTS.md).

## Intent

Today a run starts from a video file. The user wants to test the pipeline on files recorded from a simulator or a robot, which are MCAP files (ROS 2). The user should be able to drop an `.mcap` on the web interface, see its image topics, choose one, and have the run take its frames **and** its camera parameters straight from the file. The existing video flow stays as it is; the MCAP input is optional.

## Goals

- Run the whole pipeline (detect, depth, distance, tracking, motion, render, sinks) on frames read directly from an MCAP image topic.
- Support `sensor_msgs/Image` and `sensor_msgs/CompressedImage`, ROS 2 CDR encoding (decided by the user as the first target).
- Web: drag and drop of an `.mcap` (and import by path under the allowed roots); topic discovery; topic picker; the run records which MCAP and topic it used.
- Camera: use the topic's `CameraInfo` as a calibration source, next to the existing options (calibration file, fixed focal, DA3 estimate).
- Use the message timestamps as `Frame.t`; irregular timing must work (all temporal parameters are already in seconds).
- Keep the CLI/YAML path working: an MCAP run is reproducible from the YAML alone.

## Non-goals (now)

- Stereo (left/right pairs), ZED SVO, Fast Foundation Stereo. That is a separate, later feature. This design only keeps the seam open (see Constraints).
- Foxglove protobuf/JSON schemas (later, through the same reader interface).
- Reading tf/odom/IMU (ego-motion), LiDAR/depth ground truth, 3D boxes from a simulator, and writing results back to MCAP. Listed as follow-ups.
- Several topics processed in one run. One topic per run.
- Bayer encodings, 16-bit and float image encodings, fisheye camera models (refused with a clear error).

## Anti-goals

- Converting the MCAP to a video behind the scenes (rejected, see Approaches): that would lose timestamps and double-compress compressed images.
- Inventing a constant frame rate, for processing or for the rendered video. Timing comes from the file.
- Changing the behavior or outputs of existing video runs.

## Constraints

- The light environment must keep working without torch. The MCAP reader is pure Python (`mcap`, `mcap-ros2-support`); no ROS installation. Decoding uses numpy and OpenCV, already dependencies.
- `Frame` stays `(index, t, rgb)`. When stereo arrives, an optional second image can be added to `Frame` and the depth stage can choose DA3, a stereo model, or both; nothing in this feature may assume one image per timestamp in a way that blocks that (keep image decoding in the source, not in the pipeline).
- Pixel convention: `CameraInfo` K is in OpenCV convention, so `cx/cy` go through `Calibration.from_opencv` (+0.5).
- Gotcha "Lens check": `camera_metadata()` of an MCAP source returns `{}`, so the QuickTime lens check does not apply.
- Adapter/config rules in CLAUDE.md apply (frozen/slots/kw_only options, `ConfigError` for user errors, framework imports lazy — the `mcap` libraries are light but should still be imported inside the reader module, not in `core/`).
- Web security: uploads and imports go through `library` and `security` the same way videos do; the server never opens a path outside the allowed roots.
- The task lifecycle in AGENTS.md applies (task file first, `DESCRIPTION.md`, `CHANGELOG.md`, tests, no commit on `main`).

## Approaches considered

**A. `McapSource`, reading the MCAP directly (chosen).** A new `FrameSource` streams the chosen topic and decodes each message to RGB. `SourceConfig` gains `mcap` and `topic`. `build_camera` accepts any `FrameSource`, plus a new calibration source from the MCAP's `CameraInfo`. The library holds videos and MCAPs side by side. More code touched, but exact timestamps, one decode, and later stereo/SVO/tf are natural additions.

**B. Convert on import (rejected).** Extract the topic to mp4 + calibration file at import time and leave the pipeline unchanged. Much less code, but it contradicts "process directly from the MCAP", loses timestamps (constant fps), re-encodes compressed images, and has no path to tf, ground truth or stereo.

The user chose A.

## Design

### Reader (`io/mcap_reader.py`, new)

- `probe_mcap(path) -> McapInfo`: topics with schema name, message count, first/last time, estimated rate, and (for image topics) the resolution from the first message. Uses the MCAP summary/statistics section when present (fast, no full scan); falls back to a bounded scan when absent. Lists only topics whose schema is `sensor_msgs/msg/Image` or `sensor_msgs/msg/CompressedImage`, plus `sensor_msgs/msg/CameraInfo` topics for pairing.
- Pairing of an image topic with its `CameraInfo`: same namespace prefix (`/cam/image_raw` and `/cam/camera_info`; `/cam/image_raw/compressed` pairs with `/cam/camera_info`). Ambiguous or missing pairing is reported to the UI, which lets the user pick or choose "none".
- `iter_images(path, topic, start, duration, stride, max_frames)`: yields `(t, rgb)` in order.
- Decoding: `Image` with encodings `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8` to RGB uint8 (respecting `step` and `is_bigendian` irrelevant for 8-bit); `CompressedImage` jpeg/png through `cv2.imdecode`. Other encodings raise `ConfigError` naming the encoding.

### `McapSource` (`io/source.py`)

- Implements `FrameSource`: `frame_size` (from the first message; every message must match, else `ContractError`), `name` (file stem + topic slug, used in the run folder name), `__iter__`, `sample(n)` (spread by message index over the selection), `describe()` (path, topic, schema, count, rate, `CameraInfo` topic).
- `Frame.t` is seconds from the first message of the selection. Time base: `header.stamp` when it is set and monotonic, otherwise the MCAP `log_time`; the choice is recorded in `describe()`.
- `start/duration/max_frames/stride` of `SourceConfig` apply to the selection, in the same way as for videos (`stride` counts messages).
- Supports the same `FrameTransform` (undistortion) as `VideoSource`.
- Rendered video keeps the **real input timestamps** (variable frame rate): each output frame is shown at its message time, so the video stays in sync with the recording and with other topics, and playback speed is true. `VideoWriter` today takes one constant fps through an ffmpeg pipe, so the MCAP path needs per-frame presentation timestamps (for example a timestamp remux pass or a writer that sets PTS); the builder picks the mechanism after a short spike. Video-input runs keep the current constant-fps writer unchanged. `metrics.json` records the nominal (median) fps for information only.

### Camera from `CameraInfo`

- New intrinsics source in `camera.*` ("from the source's `CameraInfo`", `camera.calibration.from_source`): converts the first `CameraInfo` of the paired topic to a `Calibration` (via `from_opencv`), then the existing flow applies (`scaled_to` the frames, optional cross-check). Lens-distortion removal is **not** configured here any more: it is the separate Rectification stage below.
- Raw `CameraInfo` (non-empty D): use K and D with `plumb_bob` / `rational_polynomial`. Rectified `CameraInfo` (D empty or all zeros): use the left 3x3 of P, no distortion. `equidistant` and other models, non-zero `roi`, or `binning > 1`: `ConfigError` for now.
- `width/height` of `CameraInfo` must match the image size (else error); a different size is handled only by the existing `scaled_to`, which already refuses another aspect ratio.
- If the MCAP has no `CameraInfo`, the other camera options remain available (calibration file, fixed focal, DA3 estimate).

### Rectification stage (new, optional, first stage after the source)

- A new UI-facing stage `rectify` (`Stage(yaml_key="rectify", order=15, optional=True)`), right after Source, so everything downstream (detectors, depth, distances, CSV boxes, render) sees rectified pixels. It wraps the existing `Undistorter` (`FrameTransform`) built from the source's `CameraInfo`; options: `enabled`, `alpha` (as `camera.calibration.alpha`).
- It uses the `CameraInfo` of the selected topic. When enabled, the intrinsics are the rectified K (`Undistorter.output`) automatically; no separate `from_source` is needed.
- **Availability rules:** the stage can only be checked when the source is an MCAP **and** the selected topic has a `CameraInfo` **with distortion** (D not empty/all zeros). Without `CameraInfo` the stage is shown disabled with the reason ("no CameraInfo for this topic"); with an already-rectified `CameraInfo` it is disabled with "already rectified". With a standard video source the stage is not offered (hidden in the UI) and `rectify.enabled: true` in a YAML with a video source is a `ConfigError`.
- Default is off. Existing video runs and the existing `camera.calibration.undistort` for videos keep working untouched (that option stays for calibration files; the new stage is only for the MCAP source).
- Field references for evaluation are annotated on the original images; as with `camera.calibration.undistort`, `metrics.json` records the K and distortion used so the evaluation can transform the references.

### Config / YAML

- `rectify: {enabled: false, alpha: 0.0}` (see above).
- `source.mcap: path` and `source.topic: /cam/image_raw`; `source.video` and `source.mcap` are mutually exclusive; `topic` is required with `mcap`, and the load fails with the list of image topics when the topic does not exist.
- `camera.calibration.from_source: true` selects the `CameraInfo` as the intrinsics source without rectifying (exclusive with `path`; `cross_check` keeps its meaning; `undistort` is rejected with it — use the `rectify` stage). `check_lens` is ignored with an MCAP source.
- `cli.py` composes `VideoSource` or `McapSource` from the config; `build_camera` takes a `FrameSource`.

### Web

- Library: entries get a `kind` (`video` | `mcap`); existing entries are videos. MCAP upload reuses the streaming multipart receiver and `library.add_staged`; import by path reuses `add_from_path` and the allowed-roots check. Validation of an MCAP is `probe_mcap` (instead of ffprobe) and requires at least one image topic.
- New route `GET /api/inputs/{id}/topics` (or the equivalent under the existing videos routes, builder's choice) returning the probe result, including `CameraInfo` pairing.
- Drop zone accepts `.mcap` as well as videos. After a drop, the UI shows the topic list (name, schema, resolution, rate, count, `CameraInfo` found or not); selecting a topic fills `source.mcap` and `source.topic`, and, if the user ticks "use the CameraInfo", `camera.calibration.from_source`. The Rectification stage lane appears only for an MCAP source, enabled or disabled as described above.
- Run submission puts the MCAP and topic in the config; the run record and `metrics.json` store them, so "re-run" and "reopen config" keep working.
- Size limits: the existing upload limit (8 GiB) applies; large recordings can be imported by path instead of uploaded.

### Tests

- A small fixture MCAP written in the tests with `mcap-ros2-support` (no ROS, no binary file in git): raw and compressed images, `CameraInfo` raw and rectified, irregular timestamps.
- Unit: probe, pairing, decoding of each encoding, errors (unknown encoding, size change, missing topic), time base choice, `CameraInfo` to `Calibration`.
- Smoke: end-to-end CLI run on the fixture with the existing fakes (`tests/smoke/smoke_fakes.py`), checking CSV `t` equals the message times.
- Web: upload and import of an MCAP, topics route, run submission with an MCAP; existing video tests unchanged.

## Left to the builder

- How the variable-frame-rate video is produced (see the reader section).
- Exact route naming and the library storage layout for MCAPs.
- The CSV/metrics fields that record the source (follow the existing `describe()` pattern; if a CSV column is added, update `TEXT_COLUMNS` and `tests/run_outputs.py` as CLAUDE.md requires).
- Dependency versions for `mcap` and `mcap-ros2-support`, and whether they belong in the base dependencies (my call: yes, they are small and pure Python).

## Calls I made (to check while reading)

1. One topic per run.
2. Time base is `header.stamp`, falling back to `log_time`; recorded in `describe()`.
3. Rendered video uses the real message timestamps (variable frame rate), as the user asked; the nominal median fps is only informational.
4. Rectified `CameraInfo` uses P; unsupported models, ROI and binning are errors for now.
5. Supported encodings: `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8`, and jpeg/png compressed.
6. `stride` counts messages, not seconds.
7. Large files: import by path as the recommended route, upload still allowed within the existing limit.

## Follow-ups (not in this feature)

tf/odom/IMU to ego-motion; LiDAR/depth topics and simulator 3D boxes as evaluation references; write-back of annotated images and detections to MCAP (Foxglove-compatible); Foxglove schemas; stereo and SVO with a stereo depth model.
