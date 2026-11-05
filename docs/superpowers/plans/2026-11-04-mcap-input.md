# MCAP Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the pipeline on an image topic read directly from a ROS 2 MCAP file (dropped on the web UI or given in YAML), with the camera taken from `CameraInfo`, an optional rectification stage, and a rendered video that keeps the real message timestamps.

**Architecture:** A new `io/mcap_reader.py` (pure Python probe, decode, iterate) feeds a new `McapSource(FrameSource)`. `FrameSource` is widened so `cli`, `build_camera` and the sinks no longer touch `VideoSource.info`. `CameraInfo` becomes an intrinsics source (`camera.calibration.from_source`) and the input of a new `rectify` stage built on the existing `Undistorter`. The web library stores MCAPs next to videos (`kind`), exposes their topics, and the React UI gains a topic picker and the rectification toggle.

**Tech Stack:** Python 3.11, `mcap` + `mcap-ros2-support` (pure Python), numpy, OpenCV, PyAV (variable-frame-rate writer), FastAPI, React 18 + vitest.

**Spec:** `docs/superpowers/specs/2026-11-04-mcap-input-design.md`

## Global Constraints

- Light environment keeps working without torch; MCAP libs are imported only inside `io/mcap_reader.py`, never in `core/`.
- ROS 2 CDR only; encodings `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8`, and jpeg/png `CompressedImage`; anything else is a `ConfigError` naming the encoding.
- `Frame` stays `(index, t, rgb)`; `Frame.t` is seconds from the first selected message.
- Time base: `header.stamp` when set (> 0) and non-decreasing, else MCAP `log_time`; recorded in `describe()`.
- `stride` counts messages. One topic per run. `source.video` and `source.mcap` are mutually exclusive.
- `CameraInfo` K is OpenCV convention: go through `Calibration.from_opencv` (+0.5 on cx/cy). Raw `CameraInfo` (D not all zero): K and D, models `plumb_bob`/`rational_polynomial`. Rectified (D empty or all zero): left 3x3 of P. `equidistant`, non-zero `roi`, `binning > 1`: `ConfigError`.
- Rectification: only MCAP source with a `CameraInfo` that has distortion; default off; `rectify.enabled: true` with a video source is a `ConfigError`. Existing `camera.calibration.undistort` for videos is untouched.
- Rendered video of an MCAP run uses per-frame real timestamps (variable frame rate); video-input runs keep the constant-fps writer.
- New docs and task files in English. Task file `docs/task_0033.md` before code. Branch `feat/mcap_input`; never commit on `main`. Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Tests: `uv run pytest`, `uv run ruff check --fix . && uv run ruff format .`; frontend `cd web && npm test && npm run typecheck`.

## Review Focus

1. A topic with no `CameraInfo` while `rectify.enabled` or `from_source` is set: clear `ConfigError` (and HTTP 422), never a crash after models load. (Task 6, 9)
2. `header.stamp` zero or decreasing (simulators without stamps): falls back to `log_time`, `t` still increasing. (Task 3)
3. Two messages with the same timestamp: the tracker divides by `dt`, so duplicates are dropped and counted in `describe()`. (Task 3)
4. An MCAP without a summary section (recording interrupted) or truncated/corrupt: probe falls back to a scan or reports a 422 `InvalidVideo`, never a 500. (Task 2, 9)
5. `Image.step` larger than `width * channels` (row padding): decoded without shear. (Task 2)

---

### Task 1: Branch, task file, dependencies, fixture writer

**Files:**
- Create: `docs/task_0033.md` (from `docs/task_TEMPLATE.md`), `tests/mcap_fixture.py`
- Modify: `pyproject.toml`, `uv.lock`

**Interfaces:**
- Produces: `tests/mcap_fixture.py`:
  `write_mcap(path: Path, *, images: Sequence[ImageSpec] = (), camera_infos: Sequence[CameraInfoSpec] = (), indexed: bool = True) -> Path`
  `ImageSpec(topic: str, frames: Sequence[np.ndarray], times_ns: Sequence[int], encoding: str = "rgb8", compressed: str | None = None, header_ns: Sequence[int] | None = None, step_pad: int = 0)` (frames are RGB uint8; `compressed` is `"jpeg"`/`"png"`; `header_ns=None` means stamp = time; `step_pad` adds bytes per row)
  `CameraInfoSpec(topic: str, width: int, height: int, k: Sequence[float], d: Sequence[float] = (), model: str = "plumb_bob", p: Sequence[float] | None = None, roi: tuple[int, int, int, int] = (0, 0, 0, 0), binning: int = 0)`

- [ ] **Step 1:** `git switch -c feat/mcap_input`; write `docs/task_0033.md` from the template (Description, Scope, Plan, Impact analysis, Verification, Dependencies) summarising the spec and linking it.
- [ ] **Step 2:** Add `mcap` and `mcap-ros2-support` and `av` to base `dependencies` in `pyproject.toml` (remove `av` from the `web` extra); `uv sync`.
- [ ] **Step 3:** Write `tests/mcap_fixture.py` with `mcap_ros2.writer.Writer`, registering the `sensor_msgs/msg/Image`, `CompressedImage`, `CameraInfo` message definitions (with `std_msgs/Header`) by `register_msgdef`; `indexed=False` writes without a summary section.
- [ ] **Step 4:** Smoke-check the fixture with a throwaway test `tests/test_mcap_fixture.py::test_roundtrip` that writes two frames and reads them back with `mcap.reader.make_reader` (assert 2 messages on the topic, schema name `sensor_msgs/msg/Image`); run `uv run pytest tests/test_mcap_fixture.py -v` → PASS.
- [ ] **Step 5:** Commit `chore: mcap dependencies and test fixture writer (task_0033)`.

### Task 2: MCAP reader (probe, pairing, decode)

**Files:**
- Create: `src/av_perception_lab/io/mcap_reader.py`, `tests/test_mcap_reader.py`

**Interfaces:**
- Consumes: `tests/mcap_fixture.py` (Task 1).
- Produces (all in `io/mcap_reader.py`):
  `class McapError(ValueError)`
  `@dataclass(frozen=True, slots=True) TopicInfo(name: str, schema: str, count: int, first_ns: int, last_ns: int, rate_hz: float)`
  `ImageTopic(topic: TopicInfo, width: int, height: int, encoding: str, camera_info_topic: str | None)`
  `McapInfo(path: Path, duration_s: float, images: tuple[ImageTopic, ...], camera_info_topics: tuple[str, ...])`
  `CameraInfoMsg(width: int, height: int, k: tuple[float, ...], d: tuple[float, ...], model: str, p: tuple[float, ...], roi: tuple[int, int, int, int], binning: int)`
  `ImageMessage(log_ns: int, header_ns: int, rgb: np.ndarray)`
  `probe_mcap(path: Path) -> McapInfo`; `pair_camera_info(image_topic: str, camera_info_topics: Sequence[str]) -> str | None`; `read_camera_info(path: Path, topic: str) -> CameraInfoMsg`; `iter_images(path: Path, topic: str) -> Iterator[ImageMessage]`; `decode_image(msg: Any, schema: str) -> np.ndarray`

- [ ] **Step 1: Failing tests** in `tests/test_mcap_reader.py`:
  `test_probe_lists_image_topics_with_resolution_and_count` (two image topics + one non-image topic; assert only the image ones are listed, `width/height/count/rate_hz`),
  `test_pairing` (`/cam/image_raw`→`/cam/camera_info`; `/cam/image_raw/compressed`→`/cam/camera_info`; no match → `None`; two candidates in different namespaces pick the nearest common prefix),
  `test_decode_encodings` (parametrised over rgb8, bgr8, rgba8, bgra8, mono8, jpeg, png: decoded RGB equals the source within 0 (raw) / ≤ 6 mean abs (jpeg)),
  `test_unknown_encoding_names_it` (`16UC1` → `McapError` containing `16UC1`),
  `test_step_padding_decodes_without_shear` (**Review Focus 5**),
  `test_probe_without_summary_scans` (`indexed=False`; **Review Focus 4**) and `test_truncated_file_raises_mcap_error` (cut the bytes of a fixture in half → `McapError`, not another exception),
  `test_read_camera_info_first_message`.
- [ ] **Step 2:** Run `uv run pytest tests/test_mcap_reader.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement the module. Use the MCAP summary/statistics for counts and times when present, otherwise one scan; resolution and encoding come from the first message of each image topic; `rate_hz = (count - 1) / (last - first)`. `decode_image` honours `step`; compressed images go through `cv2.imdecode` then BGR→RGB. Wrap every library exception in `McapError`.
- [ ] **Step 4:** Run the tests → PASS. `ruff`. Commit `feat(io): MCAP reader (task_0033)`.

### Task 3: `FrameSource` widening and `McapSource`

**Files:**
- Modify: `src/av_perception_lab/io/source.py`, `src/av_perception_lab/config.py` (`SourceConfig`), `src/av_perception_lab/io/sinks.py:154-160,454-460,537`, `src/av_perception_lab/cli.py:173,212,240`
- Create: `tests/test_mcap_source.py`
- Test: `tests/test_source.py`, `tests/test_sinks.py` (must still pass)

**Interfaces:**
- Consumes: Task 2 reader.
- Produces: on `FrameSource` (ABC): `with_transform(transform: FrameTransform) -> FrameSource` (abstract), `nominal_fps: float` (property), `n_frames_total: int | None` (property), `input_name: str` (property; the file name for messages), `audio_from() -> Path | None` (default `None`), `camera_info() -> CameraInfoMsg | None` (default `None`), `frame_times_known: bool` (default `False`). `VideoSource` implements them from `self.info` (`audio_from()` returns the path only when `is_whole_video and info.has_audio`). `McapSource(cfg: SourceConfig, *, transform: FrameTransform | None = None)` with `frame_times_known = True`.
- `SourceConfig` gains `mcap: Path | None = None`, `topic: str | None = None`; `__post_init__` rejects `video` + `mcap`, and `mcap` without `topic`.

- [ ] **Step 1: Failing tests** in `tests/test_mcap_source.py`: `test_frames_use_message_times` (irregular times 0, 0.10, 0.35 s → `Frame.t` equals them relative to the first; indices 0..2), `test_stride_start_duration_max_frames_count_messages`, `test_header_stamp_zero_falls_back_to_log_time` (**Review Focus 2**; `describe()["time_base"] == "log"`), `test_decreasing_header_falls_back_to_log_time`, `test_duplicate_timestamps_are_dropped_and_counted` (**Review Focus 3**; `describe()["duplicates_dropped"] == 1`, `t` strictly increasing), `test_size_change_raises_contract_error`, `test_missing_topic_lists_image_topics` (`ConfigError` containing the available topic names), `test_sample_spreads_over_selection`, `test_transform_applied_and_shape_checked`, `test_source_config_exclusivity`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `McapSource` in `io/source.py`: probe at construction; `name` = file stem + topic slug (`/` → `_`); decide the time base from the first 50 messages; drop non-increasing timestamps; `nominal_fps` = median message rate; `describe()` = path, topic, schema, count, rate, time_base, duplicates_dropped, camera_info_topic, transform.
- [ ] **Step 4:** Replace the `source.info.*` uses in `cli.py` and `sinks.py` with the new members (`[run]` line prints `nominal_fps`; error message uses `input_name`; `RunContext.source: FrameSource`; `VideoSink` uses `source.audio_from()`; the `n_total` of the metrics uses `source.n_frames_total`, `None` handled).
- [ ] **Step 5:** `uv run pytest -q` → all PASS (existing video tests unchanged). `ruff`. Commit `feat(io): McapSource and widened FrameSource (task_0033)`.

### Task 4: `CameraInfo` → `Calibration`

**Files:**
- Modify: `src/av_perception_lab/camera.py` (+ `tests/test_camera.py`)
- Create: `tests/test_camera_info.py`

**Interfaces:**
- Consumes: `CameraInfoMsg` (Task 2), `Calibration.from_opencv` (existing).
- Produces: `calibration_from_camera_info(msg: CameraInfoMsg, image_size: tuple[int, int]) -> Calibration` and `has_distortion(msg: CameraInfoMsg) -> bool` in `camera.py`.

- [ ] **Step 1: Failing tests** in `tests/test_camera_info.py`: `test_raw_uses_k_and_d` (cx/cy gain +0.5 via `from_opencv`), `test_rectified_uses_p` (D empty and D all zeros → fx/cx from P, `has_distortion` False), `test_size_mismatch_raises`, `test_equidistant_roi_and_binning_raise_config_error` (parametrised, message names the field).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. **Step 5:** Commit `feat(camera): CameraInfo to Calibration (task_0033)`.

### Task 5: Config — `from_source`, `rectify` stage, validation

**Files:**
- Modify: `src/av_perception_lab/config.py` (`CalibrationConfig` ~l.73, `RunConfig` ~l.203, stage list ~l.407), `src/av_perception_lab/web/stages.py` if it lists lanes
- Test: `tests/test_config.py`, `tests/web/test_api_config.py`

**Interfaces:**
- Produces: `CalibrationConfig.path: Path | None = None`, `CalibrationConfig.from_source: bool = False`; `RectifyConfig(enabled: bool = False, alpha: float = 0.0)`; `RunConfig.rectify: RectifyConfig`; `Stage(yaml_key="rectify", title="Rectification", order=15, config_cls=RectifyConfig)`.

- [ ] **Step 1: Failing tests** in `tests/test_config.py`: `from_source` XOR `path` (both/neither → error); `from_source` + `undistort` → error mentioning `rectify`; `from_source` or `rectify.enabled` with `source.video` → `ConfigError` with dotted path (`camera.calibration.from_source` / `rectify.enabled`); with `source.mcap` + `topic` both load; `rectify.alpha` outside [0,1] → error; the catalog (`tests/web/test_api_config.py`) lists the `rectify` stage between `source` and `detectors`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (cross-section checks in `RunConfig.__post_init__`, following how existing cross-field checks are reported with dotted paths). **Step 4:** Run → PASS; existing config tests still PASS. **Step 5:** Commit `feat(config): MCAP source fields, from_source, rectify stage (task_0033)`.

### Task 6: Camera composition and rectification

**Files:**
- Modify: `src/av_perception_lab/camera.py:545-592` (`CameraSetup`, `build_camera`), `src/av_perception_lab/cli.py:173`
- Test: `tests/test_camera.py`

**Interfaces:**
- Consumes: Tasks 3–5.
- Produces: `build_camera(cfg: CameraConfig, source: FrameSource, rectify: RectifyConfig | None = None) -> CameraSetup`; `CameraSetup.source: FrameSource`. `cli.run` builds `McapSource` or `VideoSource` from `cfg.source` and passes `cfg.rectify`.

- [ ] **Step 1: Failing tests**: `test_from_source_uses_camera_info_intrinsics` (no rectification: frames unchanged, K from the message, `details["calibration"]` present), `test_rectify_undistorts_and_returns_rectified_k` (frames differ from input, intrinsics are `Undistorter.output`, `details["undistort"]` records K and distortion), `test_no_camera_info_with_rectify_or_from_source_raises_config_error` (**Review Focus 1**; raised before any model load), `test_rectified_camera_info_with_rectify_raises_config_error` ("already rectified"), `test_rectify_off_leaves_video_path_untouched` (video source + defaults == old behaviour).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement: `rectify.enabled` implies `CameraInfo` intrinsics; `check_lens` skipped when `source.camera_metadata()` is `{}`; `cross_check` works with `from_source`. **Step 4:** Run `uv run pytest -q` → PASS. **Step 5:** Commit `feat(camera): CameraInfo intrinsics and rectification (task_0033)`.

### Task 7: Variable-frame-rate video

**Files:**
- Modify: `src/av_perception_lab/io/video.py`, `src/av_perception_lab/io/sinks.py` (`VideoSink.open/write`)
- Test: `tests/test_video_io.py`, `tests/test_sinks.py`

**Interfaces:**
- Produces: `TimestampedVideoWriter(path: Path, width: int, height: int, *, crf: int = 20)` with `write(rgb: np.ndarray, t: float) -> None`, `close() -> None`, context manager; H.264, yuv420p, `+faststart`. `VideoSink` uses it when `source.frame_times_known`, else the existing `VideoWriter` unchanged.

- [ ] **Step 1: Spike (time-boxed):** write the writer with PyAV (`libx264`, `stream.time_base = Fraction(1, 90000)`, `frame.pts = round(t * 90000)`). If the bundled encoder lacks `libx264`, fall back to encoding with `VideoWriter` and remuxing the packets with the target PTS via PyAV (as `web/library._remux_h264_av` does); keep the same public interface.
- [ ] **Step 2: Failing tests**: `test_vfr_video_keeps_timestamps` (write frames at t = 0, 0.10, 0.35 s; read packet PTS back with PyAV; assert equal within 1 ms), `test_video_sink_uses_real_times_for_mcap_source` (fake source with `frame_times_known=True`), `test_video_sink_unchanged_for_video_source`.
- [ ] **Step 3:** Run → FAIL; implement; run → PASS. **Step 4:** Commit `feat(io): variable-frame-rate video writer (task_0033)`.

### Task 8: End-to-end CLI smoke on an MCAP

**Files:**
- Create: `tests/smoke/test_smoke_mcap.py`
- Modify: `tests/smoke/smoke_scene.py` or `tests/smoke/conftest.py` only if a helper is needed to write the synthetic scene frames into the fixture MCAP.

**Interfaces:** Consumes everything above; produces nothing.

- [ ] **Step 1: Failing tests**, using the scene frames and `install_scene_fakes` (as `test_smoke_pipeline.py` does), one `output.out_root` per run: `test_mcap_run_produces_outputs_with_message_times` (CSV `t` column equals the message times relative to the first; `metrics.json["source"]` has topic and time base; `video.mp4` PTS match), `test_mcap_camera_info_intrinsics_used` (`from_source` gives the K of the message), `test_mcap_rectify_end_to_end` (distorted fixture + `rectify.enabled`: boxes are in rectified pixels, `metrics.json` records K and distortion), `test_video_run_unchanged` (an existing video config yields the same CSV columns count as `tests/run_outputs.py` expects).
- [ ] **Step 2:** Run `uv run pytest -m smoke -v` → FAIL, fix wiring until PASS. **Step 3:** Commit `test(smoke): MCAP end to end (task_0033)`.

### Task 9: Web backend — MCAP in the library and topics route

**Files:**
- Modify: `src/av_perception_lab/web/library.py` (`VideoRecord`, `add_staged`, `add_from_path`, `_finish`), `src/av_perception_lab/web/app.py` (`submit_run`, new route), `src/av_perception_lab/web/jobs.py` / `history.py` only where `video_id` is used for the thumbnail or record
- Create: `tests/web/test_api_mcap.py`

**Interfaces:**
- Consumes: `probe_mcap` (Task 2).
- Produces: `VideoRecord.kind: str = "video"` (`"video" | "mcap"`) and `VideoRecord.mcap: dict | None` (JSON of `McapInfo`: `images` with `camera_info_topic`, camera-info availability and `has_distortion` per topic); for an MCAP, `width/height/fps/duration_s` come from the first image topic, `preview=None`, `preview_status="ready"`, `thumb` = JPEG of its first frame. Route `GET /api/videos/{video_id}/topics` → the `mcap` JSON (404 for a video). It is registered **before** `/api/videos/{video_id}/{name}`. `GET /api/videos/{id}/file` serves the original for either kind.

- [ ] **Step 1: Failing tests** in `tests/web/test_api_mcap.py`: `test_upload_mcap_lists_topics` (two image topics, one with `CameraInfo`; response `kind == "mcap"`, topics with resolution/rate/count/pairing/`has_distortion`), `test_import_mcap_by_path_under_roots_and_403_outside`, `test_mcap_without_image_topics_is_422`, `test_corrupt_or_truncated_mcap_is_422_not_500` (**Review Focus 4**), `test_unindexed_mcap_is_accepted`, `test_run_with_mcap_binds_source_mcap` (config `source.topic` given, `source.mcap` set to the library original, `source.video` absent), `test_run_with_rectify_but_topic_without_camera_info_is_422` (**Review Focus 1**; error path `rectify.enabled`), `test_video_routes_unchanged` (existing `tests/web/test_api_videos.py` passes).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement: branch on the `.mcap` suffix in `add_staged`/`add_from_path`; validation via `probe_mcap` raising `InvalidVideo` on `McapError` or no image topic; `submit_run` binds `source.mcap` for an `mcap` record and rejects a missing `source.topic` or a topic not in the file (422, path `source.topic`). **Step 4:** `uv run pytest tests/web -q` → PASS. **Step 5:** Commit `feat(web): MCAP inputs in the library (task_0033)`.

### Task 10: Web frontend — drop, topic picker, rectification toggle

**Files:**
- Create: `web/src/video/TopicPicker.tsx`, `web/src/video/mcap.test.tsx`
- Modify: `web/src/video/DropZone.tsx`, `web/src/video/useVideoSession.ts`, `web/src/video/useProcess.tsx`, `web/src/video/VideoDeck.tsx`, `web/src/api/video.ts`, `web/src/api/types.ts`, `web/src/editor/*` (Rectification lane control), `web/src/test/fixtures.ts`

**Interfaces:**
- Consumes: Task 9 JSON.
- Produces: `fetchTopics(videoId: string): Promise<McapTopics>` in `api/video.ts`; types `McapTopic { name; schema; width; height; rateHz; count; cameraInfoTopic: string | null; hasDistortion: boolean }`; `TopicPicker({ topics, value, onChange, useCameraInfo, onUseCameraInfo })`.

- [ ] **Step 1: Failing tests** in `web/src/video/mcap.test.tsx`: `drop accepts .mcap and videos`; `after an MCAP drop the topic list shows name, resolution, rate and CameraInfo badge`; `selecting a topic writes source.mcap/source.topic into the config`; `ticking "use CameraInfo" sets camera.calibration.from_source`; `rectification toggle is hidden for a video source`; `rectification toggle is disabled with "no CameraInfo for this topic" when the topic has none`; `rectification toggle is disabled with "already rectified" when the CameraInfo has no distortion`; `rectification enabled sets rectify.enabled`.
- [ ] **Step 2:** `cd web && npm test -- mcap` → FAIL. **Step 3:** Implement; reuse the existing upload and import calls (extend the accepted extensions and the record type). **Step 4:** `npm test && npm run typecheck` → PASS. **Step 5:** Commit `feat(web): MCAP drop, topic picker, rectification toggle (task_0033)`.

### Task 11: Docs, task closure and full verification

**Files:**
- Modify: `DESCRIPTION.md`, `README.md` (usage line), `docs/reference.md` (YAML keys `source.mcap`, `source.topic`, `camera.calibration.from_source`, `rectify`; MCAP section), `docs/web_extending.md` if the stage list is described, `CLAUDE.md` (gotchas: MCAP time base, duplicate drop, VFR video), `CHANGELOG.md`, `docs/task_0033.md` (title `(COMPLETED)`, Implementation details, Test notebook)

- [ ] **Step 1:** Write the docs above (English); `DESCRIPTION.md` in the same commit as the code per CLAUDE.md.
- [ ] **Step 2:** Run `uv run ruff check --fix . && uv run ruff format . && uv run pytest -q && uv run pytest -m smoke -q` and `cd web && npm test && npm run typecheck && npm run build`; all PASS.
- [ ] **Step 3:** Manual check with the real server (skill `run`): upload a ROS 2 MCAP, pick a topic, run, confirm the video plays with the recording's timing.
- [ ] **Step 4:** Commit `docs: MCAP input (task_0033)`; open a PR from `feat/mcap_input` (no commits on `main`).
