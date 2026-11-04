# Config presets in the web interface — design

Status: draft for approval (2026-11-03). Not yet a `docs/task_DDDD.md`; that
file is created from this design before any code (see AGENTS.md).

## Intent

Today the web editor can only open a YAML from the user's computer and
download it back (`Open YAML` / `Save YAML`, `web/src/App.tsx`). The server
remembers nothing. The user wants a **list of ready-made pipeline setups** to
load in one click, and a **save with a name of their choosing** that stores a
setup on the server for later. The 16 files in `configs/` must appear in that
list with short, clear names instead of file names.

## Goals

- A "Presets" list in the editor: the built-in setups from `configs/` plus the
  user's own, each with a short name and a one-line description.
- Loading a preset fills the editor with the pipeline part of that setup.
- "Save as preset…" asks for a name and stores the current pipeline on the
  server.
- A user preset can be deleted. Built-in presets cannot be changed or deleted.
- Presets are plain YAML files, usable from the command line too.

## Non-goals

- No sharing or syncing between servers or users; presets live on one server.
- No versioning or history of a preset.
- No renaming (save under the new name, delete the old one).
- Not replacing `Open YAML` / `Save YAML`; those stay for files on the user's
  own computer.

## Anti-goals

- A preset must never carry the video, the output folder or machine paths.
  Loading one must not change the video or the output folder chosen in the
  interface.
- Saving must never overwrite or alter a file in `configs/`, and never
  silently overwrite another user preset.

## What a preset contains (decided)

A preset holds the **pipeline** only. Everything below is removed when saving
and left untouched when loading:

- `source` (the whole section: `video`, `start`, `duration`, `max_frames`,
  `stride`)
- `output` (the whole section: `out_root`, `tag`, `save_depth`)

`device` **is** part of the preset (it depends on the model; the light preset
exists for a 4 GB GPU). Everything else in `RunConfig` (detectors, depth,
camera, aggregation, refiners, semantic, agreement, tracking, smoothing,
motion, distance, render) is part of the preset.

Loading replaces the pipeline keys in the editor and keeps the editor's
current `source` and `output` as they are. If the editor has unsaved changes,
the existing "Discard the unsaved changes?" confirmation applies, as for
`Open YAML`.

Built-in files in `configs/` still contain `source` and `output`. They are
served to the interface with those two sections stripped; the files on disk do
not change.

## Constraints

- The server already sets `source.video`, `output.out_root` and `output.tag`
  at run time (`web/app.py`, `web/compare.py: SERVER_SET_PATHS`); the preset
  design follows that split.
- Client-supplied names are never used as paths directly. A user preset name
  becomes a file name through a strict slug (letters, digits, `-`, `_`), and
  every path goes through `web.paths.safe_join`.
- Saved content is checked with the existing `/api/config/validate` rules
  before writing, and size-limited like `/api/config/parse` (`MAX_YAML_CHARS`).
- The web server runs on loopback by default; the new endpoints use the same
  host and origin protections as the others (`web/security.py`).
- Documentation and the task file are written in English (CLAUDE.md).
- The frontend follows the existing structure (`web/src/api/`, `editor/`,
  one hook per concern) and has tests next to the code.

## Approaches considered

- **A. YAML files on disk plus a names manifest (chosen).** Built-ins stay in
  `configs/*.yaml`; a new `configs/presets.yaml` maps each file to a name and a
  description. User presets are YAML files in a server folder. Plain files, no
  change to the config loader. Cost: the file name limits the name a user may
  type (it is slugged).
- **B. One JSON file holding all user presets.** Any name is accepted, one
  write. Cost: presets are locked inside the JSON, cannot be used from the CLI
  without an export, and concurrent writes need care.
- **C. Name and description inside each YAML** (`preset:` key). Name travels
  with the file. Cost: the config loader rejects unknown keys, so the core
  would change; the most invasive option.

The user picked A.

## Design (A)

### Storage

- Built-ins: `configs/*.yaml` (read-only) and the manifest
  `configs/presets.yaml`:

  ```yaml
  presets:
    - file: rfdetr_da3.yaml
      name: "RF-DETR + DA3"
      description: "Detector RF-DETR with metric depth, no tracking."
  ```

  The manifest lists the files in display order. A `.yaml` in `configs/` that
  is not in the manifest is not shown (it is not a runnable config: for
  example the manifest itself and `labels/`).
- User presets: `<presets-root>/<slug>.yaml`, one per preset. The default
  `<presets-root>` is `_testbed/presets/` (not tracked by git), set with a new
  server option `--presets-root`, next to `--out-root` and `--library-root`.
- The slug loses spaces, accents and case, so the name the user typed is also
  stored in a first-line comment `# name: <typed name>` and shown in the list;
  if the comment is missing, the slug is shown.

### Built-in names (proposal, to be confirmed)

| File | Name | Description |
|---|---|---|
| `rfdetr_da3` | RF-DETR + DA3 | Boxes from RF-DETR, metric depth from DA3. The base pipeline. |
| `rfdetr_da3_track` | RF-DETR + DA3 + tracking | Adds IoU tracking and Kalman smoothing of the distance. |
| `rfdetr_da3_motion` | RF-DETR + DA3 + motion | Tracking plus 2D motion and trajectory prediction. |
| `rfdetr_da3_calibrated` | RF-DETR + DA3, calibrated camera | Uses a camera calibration file you provide. |
| `rfdetr_seg_da3` | RF-DETR Seg + DA3 | Distance over instance masks. |
| `rfdetr_sam2_da3` | RF-DETR + SAM 2 + DA3 | Boxes refined into masks by SAM 2.1. |
| `rfdetr_semantic_da3` | RF-DETR + DA3 + scene map | Excludes sky and road surface using SegFormer. |
| `rfdetr_lanes_scene_da3` | RF-DETR + DA3 + lanes and scene | Adds lane lines and the scene map. |
| `rfdetr_sam2_lanes_motion_da3` | Full stack | RF-DETR, SAM 2, lanes, tracking and motion. Needs a large GPU. |
| `yolo_o365_da3` | YOLO Objects365 + DA3 | YOLO26-L with 365 classes. |
| `yolo_o365_sam2_da3` | YOLO Objects365 + SAM 2 + DA3 | YOLO boxes refined into masks. |
| `yolo_o365_seg_da3` | YOLO Objects365 Seg + DA3 | Instance masks, Objects365 vocabulary. |
| `yolo_seg_da3` | YOLO Seg (COCO) + DA3 | YOLO26-L instance masks. |
| `yolo_light_da3` | Light (4 GB GPU) | YOLO26s-seg, tracking and smoothing, for small GPUs. |
| `multi_detector` | RF-DETR + YOLO, all boxes | Both detectors, every box kept and tagged. |
| `multi_detector_fused` | RF-DETR + YOLO, fused | Both detectors, boxes of one object merged. |

### API (new, under `web/app.py`; logic in a new `web/presets.py`)

- `GET /api/presets` → list of `{id, name, description, kind}` where `kind` is
  `builtin` or `user`; ids are `builtin:<file stem>` / `user:<slug>`.
- `GET /api/presets/{id}` → `{config, extras}` in the shape of
  `/api/config/parse`, with `source` and `output` removed.
- `PUT /api/presets/user:<slug>` with `{name, config, extras, overwrite}` →
  validates, strips `source`/`output`, writes atomically. Answers 409 if the
  slug exists and `overwrite` is false, 422 for an invalid config or name,
  and 409 if the slug collides with a built-in id.
- `DELETE /api/presets/user:<slug>` → 204; `builtin:` ids answer 403.

### Frontend

- A "Presets" button beside `Open YAML` in `App.tsx` opens a list with two
  groups (Built-in, Mine), each row showing name and description. Choosing a
  row loads it through the existing editor load path, keeping the editor's
  `source` and `output`.
- "Save as preset…" asks for a name, shows a clear message if the name already
  exists (offer overwrite or another name), and refreshes the list.
- A trash action on user rows, with a confirmation.
- A new `api/presets.ts` client and a `useProcess`-independent hook, with tests
  next to them.

## Edge cases

- Name with spaces, accents, or only symbols: slugged; an empty slug is
  rejected with a message.
- A preset that references weights or label maps missing on this machine:
  loads normally; the existing model-readiness banner already reports what is
  missing.
- A user file edited by hand and now invalid: listed, and loading shows the
  validation errors instead of failing silently.
- Missing or malformed manifest: the server logs it, lists the built-ins by
  file stem, and keeps working.
- Two saves of the same name at once: the write is atomic and the second gets
  409 unless `overwrite` is set.

## Verification

- Backend tests: listing (built-in + user + manifest missing), strip of
  `source`/`output`, slug and traversal attempts (`../x`, absolute paths,
  symlinks), 409/403/422 cases, atomic write, size limit.
- Frontend tests: list rendering, load keeps the editor's video and output,
  overwrite prompt, delete confirmation.
- A manifest test: every file in the manifest exists and parses, and every
  runnable `configs/*.yaml` is listed (so a new config is not forgotten).
- Manual check in the browser: load a preset, change the video, run.

## Left to the builder

- Exact layout and styling of the list (follow the existing editor).
- Default and exact wording of messages.
- Whether `presets.yaml` is read once at startup or on each request (prefer
  on each request; the file is tiny).

## Documentation to update in the same commit

`DESCRIPTION.md` (new endpoints), `README.md` / `docs/reference.md` (one
short section), `docs/web_extending.md` if it lists endpoints, `CHANGELOG.md`,
`AGENTS.md` only if the file layout list changes, and the new
`docs/task_0030.md` (English).
