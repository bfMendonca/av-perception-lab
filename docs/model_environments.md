# Model environments: requirements, Prepare and dependency conflicts

This document explains how a model tells the project what it needs to run,
how the CLI and the web interface obtain it, and what to do when a model's
Python packages cannot live in the same environment as the others. It is the
reference for two readers:

- **someone who cloned the repository** and wants a sample YAML to run
  (sections 1–6);
- **someone adding a model** (section 7, a worked example) or facing a
  dependency conflict (section 8, the five resolution paths).

Everything below was checked against the code on 2026-10-31 (tasks
0019–0024). The implementation lives in `src/av_perception_lab/core/requirements.py`
(declarations and the torch-free check), `core/provision/` (engine, sources,
packages layer), `adapters/_weights.py`, `adapters/_hf_hub.py`,
`adapters/_ultralytics.py` (shared adapter helpers), `cli.py` (`--prepare`,
the not-ready gate) and `web/models.py` (status rows, the prepare queue).

---

## 1. The model "recipe"

Every component class (detector, depth model, refiner, semantic segmenter,
tracker) answers one question, from its **already-validated options** and
never from `load()`:

```python
@classmethod
def requirements(cls, options) -> Requirements: ...
```

The default (`Component.requirements` in `core/components.py`) declares
nothing. A `Requirements` (`core/requirements.py`) has three layers:

```python
@dataclass(frozen=True, kw_only=True)
class Requirements:
    assets: tuple[Asset, ...] = ()       # weights and other files
    code: tuple[CodeRepo, ...] = ()      # pinned git clones
    packages: tuple[str, ...] = ()       # import names (checked with find_spec)
    extra: str | None = None             # the pyproject extra that provides them
```

| Layer | What it is | Where it lives | How "ready" is decided |
|---|---|---|---|
| **assets** | files a model loads (checkpoints, configs) | `<weights root>/<target>` | each asset kind's `present(root)` (file exists, size, sha256, HF cache layout) |
| **code** | an upstream repository used as source code, not as a package | `<weights root>/_code/<name>` | the folder is a git clone whose `HEAD` is the pinned commit (`git rev-parse HEAD`) |
| **packages** | Python import names | the running environment | `importlib.util.find_spec` of the top-level name (nothing is imported) |

The check (`core.requirements.check(req, root)`) uses only the file system,
`git rev-parse` and `find_spec`, so the CLI, the web server and the UI
compute readiness on a fresh clone without torch. A model is `ready` only
when the three layers are ready.

Every current adapter declares `packages` from the existing `infer` extra
(`extra="infer"`, the constant `INFER_EXTRA` of `adapters/_weights.py`);
only *new* models get their own small extras (section 6).

### 1.1 Asset kinds

Asset kinds are registered classes (`@register_asset_kind(name)`), each with
its own `present(root) -> AssetState` (`ready`, `missing`, `partial`,
`invalid`). `partial` is the tool's own unfinished work (a `.part`, an
`.incomplete` blob, a clone being made); `invalid` is an unsafe path or an
existing file that is not the declared one (see "exists with a different
size" below), which nothing ever replaces. Fields common to every asset
(keyword-only):

| Field | Meaning |
|---|---|
| `key` | name shown in messages (`"weights"`) |
| `target` | path relative to the weights root; `..`, absolute paths and symlinks that leave the root are refused (`UnsafePathError`) |
| `size_bytes` | expected size (shown before Prepare; integrity check for URLs; a lower bound for Hugging Face) |
| `license` | shown in the UI next to the size |
| `gated` | the source requires accepting a licence (see 5.3) |
| `known` | **fails closed: `False` by default.** The adapter's code sets `known=True` only when both the source (a table in its code) and the target (the canonical location under the weights root) are its own: `ultralytics/<release name>`, the `hf` cache for repos in `HF_FACTS`, the RF-DETR table, the TwinLiteNet+ Drive files at `_code/TwinLiteNetPlus/pretrained/<variant>.pth` of the pinned clone. A release file name at a `weights` path of the YAML's choosing, a known repo in another `hf_cache` folder or a repo typed in the YAML stays `False`. The web prepares only known sources (`engine.prepare(..., allow_unknown=False)`, re-checked when the job runs); the CLI prepares both (`allow_unknown=True`). `CodeRepo.known` follows the same rule (the pinned TwinLiteNet+ clone is known). `known` does not apply to kinds that are never downloaded (`UserProvided`) |

Built-in kinds:

| Kind (registered name) | Class | Use | Rules |
|---|---|---|---|
| Hugging Face (`huggingface`) | `HuggingFaceAsset(repo, files, patterns, revision)` | a model repo on the Hub | `target` is a **cache folder** (e.g. `hf`), read in the Hub's own layout `models--org--name/snapshots/<rev>/…`, so an existing `weights/hf` cache counts as ready unchanged. `files` are exact names, `patterns` fnmatch globs (both empty = the whole snapshot). `revision` is a branch, tag or commit name **without `/`**: a pull-request ref such as `refs/pr/1` is refused; use the commit hash of that PR instead. A `.incomplete` blob is never ready. |
| URL (`url`) | `UrlAsset(url, sha256)` | one file over http(s) | **declare `size_bytes` or `sha256`** (project rule). The download goes to `<target>.part` and is renamed only when the declared size or sha256 matches, or the server's full length was received; otherwise it fails with "cannot verify download integrity: declare size_bytes or sha256 for this asset". A `.part` is resumed (`Range` + `If-Range`) only when size or sha256 is declared and the server answers 206 at the right offset; anything else restarts. Redirects go only to http(s), never from https down to http. The body is bounded by the expected size (the smaller of the server's length and the declared size): more bytes fail and remove the `.part`; free disk space is checked before the first byte is written. **An existing file at the target whose size or sha256 differs** (for example a fine-tuned checkpoint saved under a release name) is never downloaded over, in the CLI either: the model is `failed` with "exists with a different size at <target>; move or delete it, or point the config at another file". |
| Google Drive (`gdrive`) | `GoogleDriveAsset(file_id, sha256, manual)` | one public file on Google Drive | the same rules as URL (`present`, `.part`, resume, size cap, free disk, never over an existing file; **declare `size_bytes` and `sha256`**), downloaded from the fixed endpoint `https://drive.usercontent.google.com/download?id=<file_id>&export=download&confirm=t`. `file_id` must match `[A-Za-z0-9_-]{10,}` (never a URL). Redirects only over https and only to `drive.usercontent.google.com`, `drive.google.com` or `*.googleusercontent.com`; any other host is refused. An HTML answer (`Content-Type: text/html`, or a body that starts like HTML) is the virus-scan confirmation or the quota page, never the file: it fails with "Google Drive returned a confirmation or quota page instead of the file; try again later or download it manually from <manual>" and nothing is kept. `describe_source()` is "Google Drive file <id>". |
| User-provided (`user`) | `UserProvided(option, hint)` | a file only the user can supply (a custom `.pt`, a manual download) | never downloaded: Prepare reports "provide this file: <hint> (expected at …, or set the option '<option>')" and the model stays `missing` (not `failed`; `--prepare` exits 1 only because the model is not ready). When such files are all that is missing (the code layer ready), the web row has `reason: "user_provided"`, no Prepare button, and Prepare all skips it. |
| Code repo (layer `code`) | `CodeRepo(name, url, commit)` | an upstream repository used as code | `name` is one plain folder name; `commit` must be **7–64 hexadecimal characters** (a branch or tag is refused: they move). Prepare runs `git clone --no-checkout` into `_code/<name>.part`, `git switch --detach` to the pin, then renames. An existing clone at another commit is fetched and switched, never deleted; a folder that is not a clone is left alone (error: move it away). Each git command times out after 900 s. |

The TwinLiteNet+ pretrained weights (`pretrained/<variant>.pth`) are
`GoogleDriveAsset`s (see 5.4): Prepare downloads them into the pinned clone.
A file whose target lies inside a declared clone's folder is fetched only
after that clone is ready (code first; otherwise it fails "not downloaded:
it goes inside code:<name>, which is not ready"), so a download never creates
the folder that `git clone` would then refuse.

**Limitation of the Drive source.** `confirm=t` skips the virus-scan page
that Drive shows for large files, and public files under a few hundred MB
come straight back. Drive can still answer with an HTML page (a file shared
too often hits a daily download quota; the endpoint's behaviour is not a
documented API): the source refuses it as above, a later Prepare retries.
Only public files work (no Google sign-in, no token).

### 1.2 Adding a source kind

A new source (an S3 bucket, a mirror) is two classes and two decorators,
and never an edit of `check`, `prepare` or the web layer. Google Drive is
the worked example (`GoogleDriveAsset` in `core/requirements.py`,
`GoogleDriveSource` in `core/provision/sources.py`); declaring a Drive file
in an adapter is one value:

```python
GoogleDriveAsset(
    key="weights",
    target="_code/Tool/pretrained/model.pth",  # under the weights root
    file_id="1N7jNsa8P1dM-UevqPRsx6heUpN4L00FJ",  # from the share link
    size_bytes=594_777,  # Content-Length of a verified download
    sha256="0dd3bd73...",  # sha256sum of the same file (full 64 hex digits)
    license="MIT",
    manual="https://drive.google.com/drive/folders/<folder id>",  # shown on failure
    known=True,  # only for a fixed id at its canonical target
)
```

A new kind:

```python
# a stdlib-only module (no framework imports)
from dataclasses import dataclass
from av_perception_lab.core.requirements import Asset, AssetState, register_asset_kind

@register_asset_kind("s3")
@dataclass(frozen=True, kw_only=True)
class S3Asset(Asset):
    bucket: str
    key_name: str
    sha256: str | None = None

    def describe_source(self) -> str:      # shown in the UI "source" column
        return f"S3 · {self.bucket}"

    def present(self, root) -> AssetState:  # ready / missing / partial
        ...
```

```python
# next to the built-in sources (core/provision/sources.py, or a module that
# core/provision/__init__.py imports, so it is registered before any Prepare)
from av_perception_lab.core.provision.engine import ProvisionError, register_source

@register_source(S3Asset)
class S3Source:
    def prepare(self, asset, root, progress, cancel) -> None:
        # write through core.requirements.contained(root, ...) to "<target>.part",
        # call progress(done, total), check cancel between chunks (raise
        # ProvisionCancelled), raise ProvisionError("readable reason") on failure,
        # then os.replace() the .part into place
        ...
```

An asset whose kind has no registered source fails with "no source
registered for asset kind '<name>' (add one with register_source)". A kind
that must never be fetched needs **both**: the class attribute
`downloadable = False` (as `UserProvided` does: the engine then reports its
source's reason as a note, not a failure, and the web marks the row
`user_provided`) **and** a registered source whose `prepare` raises
`ProvisionError("provide this file: <hint>")`. With `downloadable = False`
alone the user sees "no source registered for asset kind … (add one with
register_source)" instead of the hint.

---

## 2. Where things go

### 2.1 The weights root

Everything a model downloads lives under one folder, the **weights root**:

1. `--weights-root DIR` of the web server (`av-perception-lab-web`), else
2. the environment variable `AVPL_WEIGHTS_ROOT`, else
3. `weights/` of the checkout (git-ignored).

The CLI (`av-perception-lab`) has no flag: it reads `AVPL_WEIGHTS_ROOT`.
The web server passes its root (from `--weights-root`, else
`AVPL_WEIGHTS_ROOT`) to every run worker subprocess as an absolute, resolved
`AVPL_WEIGHTS_ROOT`, so `--weights-root` alone is enough for the web; export
`AVPL_WEIGHTS_ROOT` only for CLI runs outside the server. The weights root
must be a dedicated folder: `/`, the home folder, and folders that contain
the project, the active Python environment or the run outputs are refused
(CLI: exit 2; web: start-up error), and the web server also refuses one that
overlaps `--library-root` or `--out-root`. Messages sent to the browser show
`<weights root>` instead of its absolute path (and are scrubbed of
credentials).

### 2.2 Default locations

The adapter decides the default location; a folder per family under the
root (`<root>/<family>/…`, with a sub-folder per model when a family has
several files per model, as in the example of section 7) and one per clone
under `<root>/_code/<name>/`.

| Model | Default location | Source |
|---|---|---|
| DA3 (`depth`, `camera.estimate`), SAM family, SegFormer | `weights/hf` (Hugging Face cache layout) | Hugging Face: the weights file + `*.json` |
| RF-DETR (`variant`) | `weights/rfdetr/<library file name>` | the URL table of the `rfdetr` library |
| YOLO, Ultralytics SAM (`weights`) | `weights/ultralytics/<file>` for a bare file name | `ultralytics/assets` v8.4.0 for published names; any other file is user-provided |
| TwinLiteNet+ (`repo`) | `weights/_code/TwinLiteNetPlus` (pinned clone) | git; `pretrained/<variant>.pth` from Google Drive (size and sha256 pinned, `docs/task_0028.md`) |

Sources, sizes and licences of every declared file are recorded in
`docs/task_0021.md`. Nothing is downloaded at run time (DA3 included: a
missing checkpoint is an error naming `--prepare`).

### 2.3 Overrides in the YAML

An explicit path option overrides the default:

- `hf_cache: some/dir` (DA3, SAM family, SegFormer), relative to the working
  directory;
- `weights: …` (YOLO, Ultralytics SAM, RF-DETR, TwinLiteNet+);
- `repo: ~/dev/TwinLiteNetPlus` (TwinLiteNet+).

An explicit path **inside** the weights root is declared like a default
(status and Prepare see it). One **outside** the root is the user's
responsibility: it is not declared (targets never leave the root), the
status does not track it, and `load()` fails with a `ConfigError` if it is
missing. Prepare never moves or deletes existing files (`~/.roboflow`,
`~/dev/...`, `weights/hf`).

**YOLO-style `weights` (YOLO, Ultralytics SAM):** a *bare file name* (one
path part, e.g. `yolo26l.pt`) resolves under `<weights root>/ultralytics/`;
a path with a directory part (`_testbed/weights/yolo26s.pt`, `/abs/x.pt`,
`~/x.pt`) keeps its usual meaning, relative to the working directory. Edge:
`./yolo26l.pt` is normalised by `pathlib` to the bare name `yolo26l.pt`, so
it also resolves under `<weights root>/ultralytics/`, not in the working
directory; write an absolute path (or one with a real directory part) for a
file in the working directory. With `--asset-root`, the web server checks a
bare name at `<weights root>/ultralytics/<name>`, where the adapter opens it.

### 2.4 Defaults that moved (existing machines)

Task 0021 moved the defaults: RF-DETR from `~/.roboflow/models` to
`weights/rfdetr`, DA3 from the runtime download to `weights/hf`, YOLO from
machine paths in the sample YAMLs (`~/dev/ultralytics/...`) to
`weights/ultralytics`. On a machine that already had the weights (the
author's laptop) run `--prepare` once per sample
config, or point the YAML options at the existing files (outside the root
they are simply not tracked). The GPU tests skip an example config until its
models are ready under the weights root.

---

## 3. States

`check` produces `ready`, `missing` or `failed` (an unsafe declaration); the
packages layer adds `conflict` and turns a dry-run that could not run into
`failed`; the web server shows `preparing` for the model a Prepare job is
working on.

| State | Meaning | How the editor draws the card |
|---|---|---|
| `ready` | all three layers present | solid outline |
| `missing` ("Not ready") | something is absent or partial (a `.part`, a wrong size, a clone at another commit) | **yellow dashed outline and a small alert icon at the top-right corner** (hover or focus it for the reason) |
| `preparing` (and the UI's `queued`) | a Prepare job is working on it (or waiting) | yellow dashed outline with a progress bar on the card |
| `failed` | a Prepare failed (offline, HTTP 404, gated, sha256 mismatch), the declaration is unusable, or the packages dry-run could not run (uv not found, timeout) | **red dashed outline**, red alert icon, the reason, **Retry** |
| `conflict` | the model's own extra cannot be installed next to the current environment (resolver error, or the plan would remove or replace installed packages) | yellow dashed + alert icon, the resolver's reason and a pointer to this document |

Any state the interface does not know uses the yellow warning style with the
server's message. Each status row also has `layers` (each of `assets`,
`code`, `packages`: `ready`, `missing`, `partial` or `invalid`), `missing`
(labels such as `weights`, `code:TwinLiteNetPlus`, `package:mmengine`),
`size_bytes` still to download and `message`.

---

## 4. Preparing from the CLI

```bash
# base stack once (see 6.1), then:
uv run --extra infer av-perception-lab --config configs/rfdetr_da3.yaml --prepare
```

`--prepare` builds the components (nothing is loaded, the video is not
probed), obtains every missing code repo and asset of every model (and of the
focal model of `camera.estimate` when it runs), prints progress, a status
table (`component  type  state  assets  code  packages`) with the weights root
and notes, and exits **0** only when every model is ready (**1** otherwise,
**2** for a configuration error, **130** on Ctrl-C; partial downloads are kept
and resumed by the next `--prepare`). It is idempotent: a ready model is a
no-op. The packages layer is reported, not installed: the base stack's
command, or for a per-model extra the clean dry-run with
"run again with --install-packages"; `--prepare --install-packages` installs
such an extra after a fresh clean dry-run (section 6). `--install-packages`
without `--prepare` is an argument error.

**The not-ready gate.** A normal run (without `--prepare`) checks every
declared requirement after every configuration error and before any model is
loaded. If a model is not ready it stops with exit code **2**:

```text
error: 3 model(s) not ready under the weights root:
  detectors[0] rfdetr-large (rfdetr): weights: rfdetr/rf-detr-large-2026.pth not found
  depth da3metric-large (da3): weights: depth-anything/DA3METRIC-LARGE not in hf
  camera.estimate da3-large-1.1 (da3): weights: depth-anything/DA3-LARGE-1.1 not in hf
run `av-perception-lab --config <yaml> --prepare` first
```

When only Python packages are missing, the hint names the base stack command
(`uv sync --extra infer`, `--prepare` does not install it) or, for a
per-model extra, `--prepare --install-packages`.

---

## 5. Preparing from the web interface

```bash
uv run --extra infer --extra web av-perception-lab-web --weights-root weights
```

### 5.1 Status

The editor asks `POST /api/models/status` with `{config}` (debounced) whenever
the pipeline changes. The body has `models` (one row per model the config would
load: `stage`, `index`, `type`, `name`, `state`, `layers`, `missing`,
`size_bytes`, `message`, `source`, `license`, `target` relative to the
weights root, `packages` (the packages plan or null), `reason`
(`unknown_source`, `user_provided`, `packages_only` or null) and
`preparable`),
`total_size_bytes` and `restart_required`. Nothing is loaded and torch is
not imported.

### 5.2 Prepare model, Prepare all

- **Prepare model** (inspector, section "Weights & requirements": layers,
  what is missing, source, size, licence, target folder) queues one model;
  **Retry** after a failure.
- **Prepare all** (the banner above the lanes: "N models need preparing · X
  to download", with their licences) opens a confirmation listing only the
  downloads (name, source, size, licence) and the total, plus, apart, the
  files you must provide yourself; **Download** queues every preparable
  model, **Cancel** (or Escape) does nothing.

Both are `POST /api/models/prepare` with `{config, models}` (`"all"` or a
list of `{stage, index}`); progress is SSE at
`GET /api/models/prepare/{id}/events`, cancel is
`DELETE /api/models/prepare/{id}`, recent jobs are
`GET /api/models/prepare` (a reloaded page re-attaches). One Prepare runs at
a time and never blocks the GPU run queue. The client only names models:
URLs and targets come from the adapters' code.

**Process** stays disabled until every model is ready; the server refuses a
run with 409 `{models_not_ready: [...]}` (the editor then offers Prepare all).

**Unknown sources.** The web prepares only sources the adapters vouch for
(`known=True`). A Hugging Face repo typed in `depth.model`,
`refiners[].model` or `semantic[].model` that the project does not ship
(`adapters/_hf_hub.py: HF_FACTS`) reads "unknown source: prepare it from the
CLI (--prepare)": naming it in a Prepare is a 422, `models: "all"` prepares
the rest and lists it under `skipped`, and the inspector shows the command
`av-perception-lab --config <file> --prepare` with a copy button. The same
holds for a known file or repo at a location of the YAML's choosing (a
release file name in a `weights` path outside `<root>/ultralytics/`, an
`hf_cache` other than `<root>/hf`): `known` is tied to the canonical target.
The queue runs the engine with `allow_unknown=False`, so this is re-checked
when the job runs (a file deleted after the job was queued is not fetched).
The CLI prepares any repo the user typed (local trust).

**User-provided files** (`reason: "user_provided"`). When everything a
Prepare can fetch is there and only files the user must provide are missing
(a custom YOLO `.pt`, a TwinLiteNet+ `weights` file with another name than `<variant>.pth`), the row is not
preparable: its message is the manual-download hint, Prepare all lists it
under `skipped`, naming it is a 422, and Process stays locked until the file
exists. A Prepare that fetched the rest ends `done`, not `failed`, and the
row then shows the hint.

**Existing files of another size.** A file at a target whose size or sha256
differs from the declaration is the user's (a fine-tuned checkpoint saved
under a release name): the row is `failed`, not preparable, with "exists with
a different size at <target>; move or delete it, or point the config at
another file", and no Prepare (web or CLI) replaces or deletes it.

### 5.3 Gated models

A gated asset (`gated=True`; today `facebook/sam3` for `sam3_tracker`)
needs the licence accepted on the source site and credentials. There is no
token entry in the UI: the server (or the CLI) uses what `huggingface_hub`
already reads, `HF_TOKEN` in its environment or a token stored by
`hf auth login`. Without them Prepare fails with "gated - accept the
licence at https://huggingface.co/<repo> and set HF_TOKEN (or run
`hf auth login`)". Start the server with `HF_TOKEN` exported (or log in
first) to prepare a gated model from the web.

### 5.4 TwinLiteNet+ weights (Google Drive, automatic)

TwinLiteNet+ (`semantic` type `twinlitenet`) is a pinned clone plus
`pretrained/<variant>.pth`, which upstream publishes only in a public Google
Drive folder
(`https://drive.google.com/drive/folders/1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz`).
Prepare (web or `--prepare`) clones the repository first, then downloads the
file into `<weights root>/_code/TwinLiteNetPlus/pretrained/` with the Google
Drive source; the file id, size and sha256 of each variant are fixed in the
adapter (`TWINLITENET_WEIGHTS`, verified in `docs/task_0028.md`):

| Variant | Drive file id | Bytes |
|---|---|---|
| `nano` | `1SvD03WZOq8eN4X0bMFNZyzj4upZM1QVl` | 217 395 |
| `small` | `1N7jNsa8P1dM-UevqPRsx6heUpN4L00FJ` | 594 777 |
| `medium` | `121z9XUh7_lgze8i6nS6Ne2HQ7ZG5_uG9` | 2 054 463 |
| `large` | `1H8P-GrOUBOaVs5LEqXBfz0dC9gguUUio` | 7 959 943 |

The web prepares it only at that canonical place (`known`); a `weights`
path elsewhere in the root named `<variant>.pth`, or a `repo` the YAML names,
is prepared by the CLI only, and a file with another name is user-provided.
Licence: the repository is MIT; the Drive files carry no licence of their
own, so the repository's is assumed. If Drive answers with a confirmation or
quota page, Prepare fails with that message: retry later, or save
`<variant>.pth` from the folder above at the path the message names.

---

## 6. The packages layer

### 6.1 The base stack (one-time, never run by the server)

The base Python stack is the `infer` extra (torch, `rfdetr`,
`depth-anything-3`, `transformers`, `timm`) plus `web` for
the server. `ultralytics` (AGPL-3.0) is not in it: it is the opt-in `yolo` extra
(`YOLO_EXTRA`), a per-model extra (6.2) that the YOLO detector and
`ultralytics_sam` declare, installed with `--prepare --install-packages` or
`uv sync --extra infer --extra yolo --extra web`. It is installed once, by hand, in the repository:

```bash
uv sync --extra infer --extra web
```

(On the notebook, with `UV_PROJECT_ENVIRONMENT=_testbed/venv-infer` exported.)
Neither the server nor `--prepare` ever runs it: when it is missing, the
inspector shows this command with a copy button instead of a button, and the
CLI names `uv sync --extra infer`. `uv sync` is exact by default: it removes
every extra it does not name, so add `--extra eval` and any per-model extra
you already installed to that command, and never run a bare `uv sync` on the
GPU host.

### 6.2 A per-model extra: detect, dry-run, install

A model whose packages are not in `infer` declares them as import names in
`packages` and names its extra in `extra` (e.g. `model-metric3d`). When they
are missing (`core/provision/packages.py`):

1. **Detect** the installed extras (`detect_extras`): the extras of
   `[project.optional-dependencies]` whose distributions all have metadata
   (`importlib.metadata`) and whose declared import names are found by
   `find_spec`. Nothing is imported.
2. **Dry-run**: `uv sync --dry-run --locked --inexact` with `--extra` for
   *every* detected extra plus the new one (run as
   `uv --color never sync --locked --inexact --extra … --dry-run`). It
   writes nothing. `--inexact` means nothing extraneous is removed; `--locked`
   means the lock is not rewritten. A dry-run that fails, or whose plan would
   remove or replace an installed package, is a **conflict**.
3. **Install** only after a clean dry-run and an explicit confirmation: the
   same command without `--dry-run`, in its own process group, output
   streamed, cancellable. From the CLI: `--prepare --install-packages`. From
   the web: **Install packages** in the inspector opens a dialog stating the
   exact command; the job (`POST /api/models/prepare` with `packages: true`
   and named models, never `"all"`) streams the `uv` output as `output`
   events. The job runs a fresh dry-run first and refuses to run if it no
   longer gives the command the user confirmed.

The base extra is never installed this way (422 from the API; the CLI prints
the command), and a per-model extra is installed only when the base stack is
detected.

**Rules and knobs:**

- **The extra must already be in `uv.lock`.** Because of `--locked`, a new
  extra added to `pyproject.toml` without updating the lock makes the dry-run
  fail ("the lockfile needs to be updated"), shown as `conflict`. Add it with
  `uv add --no-sync --optional model-metric3d '<package>'` (or edit
  `pyproject.toml` and run `uv lock`) and commit both files. An extra the
  `pyproject.toml` does not declare is `failed` with "extra '<name>' is not
  declared in pyproject.toml".
- **`AVPL_UV`** names the `uv` executable (default `uv` on the `PATH`; set
  `AVPL_UV` when uv is not on the `PATH`). The
  command runs in the repository; when `UV_PROJECT_ENVIRONMENT` is not set
  it targets the environment the server or CLI runs in.
- **Transient failure vs conflict.** uv not found, a timeout or a cancel is
  *not* a conflict: the row is `failed` with the reason (naming `AVPL_UV` when
  uv is missing), the result is not cached and the next status check asks
  again. A resolver error or a plan that would remove/replace packages is
  `conflict` (cached until `pyproject.toml` or `uv.lock` changes).
- **Installs and GPU runs exclude each other.** While a run is queued or
  running, Install packages answers 409 (a running worker must not see a
  half-installed package); while an install is queued or running,
  `POST /api/runs` answers 409 (`packages_installing`).
- **After an install.** The status body then has `restart_required: true`
  and the editor notes that new runs use the packages at once: each run
  starts a fresh worker process, and the status checks re-scan the
  environment. A server restart (no web worker restart is ever needed) is
  only for an upgraded package the server process had already imported.
- uv output shown to clients is scrubbed of credentials and shows the
  project folder as `.`.

---

## 7. Worked example: adding a Metric3D depth model

A hypothetical adapter for Metric3D v2 (metric depth from one image and the
focal). It is *not* in the repository; URLs, sizes, licence and commit below
are **placeholders** to replace with real values (an HTTP HEAD gives the
size; never invent them). The code was checked by loading it next to the
real registry: its requirements, status rows and catalog entry come out as
described below.

### 7.1 The adapter module

`src/av_perception_lab/adapters/depth/metric3d.py`:

```python
"""Metric3D v2 metric depth.

Weights: a URL asset under `<weights root>/metric3d/<variant>/`; code: the
upstream repository as a pinned clone under `<weights root>/_code/Metric3D`;
packages: the `model-metric3d` extra. Frameworks are imported only in `_load`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from av_perception_lab.adapters._weights import not_prepared, under_root
from av_perception_lab.core.components import DepthEstimator, LoadContext
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.registry import register
from av_perception_lab.core.requirements import (
    CodeRepo,
    Requirements,
    UrlAsset,
    UserProvided,
    weights_root,
)
from av_perception_lab.core.types import DepthResult

METRIC3D_DIR = "metric3d"  # <weights root>/metric3d/<variant>/<file>
METRIC3D_EXTRA = "model-metric3d"  # [project.optional-dependencies]
# PLACEHOLDERS: file name, URL, bytes, licence from the upstream release.
METRIC3D_CHECKPOINTS: dict[str, tuple[str, str, int, str]] = {
    "vit-small": ("metric_depth_vit_small.pth",
                  "https://example.org/metric3d/metric_depth_vit_small.pth",
                  150_000_000, "BSD-2-Clause"),
    "vit-large": ("metric_depth_vit_large.pth",
                  "https://example.org/metric3d/metric_depth_vit_large.pth",
                  1_300_000_000, "BSD-2-Clause"),
}
METRIC3D_REPO = CodeRepo(
    name="Metric3D",
    url="https://github.com/YvanYin/Metric3D",
    commit="0123456789abcdef0123456789abcdef01234567",  # PLACEHOLDER: 7-64 hex
    known=True,  # fixed here, cloned to its fixed folder: the web may prepare it
)


def default_weights(variant: str) -> Path:
    return weights_root() / METRIC3D_DIR / variant / METRIC3D_CHECKPOINTS[variant][0]


@register("depth", "metric3d")
class Metric3DDepth(DepthEstimator):
    """Metric3D v2: metric depth from one image and the camera focal.

    (The first paragraph is the module's description in the web catalog.)
    """

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options:
        variant: Literal["vit-small", "vit-large"] = field(
            default="vit-small",
            metadata={"ui": {"label": "Variant", "help": "vit-large: slower, sharper."}},
        )
        # None = <weights root>/metric3d/<variant>/<file>, downloaded by Prepare
        weights: Path | None = field(
            default=None, metadata={"ui": {"label": "Checkpoint", "advanced": True}}
        )
        process_res: int = field(
            default=616,
            metadata={"ui": {"label": "Input size", "unit": "px",
                             "min": 224, "max": 1064, "step": 14}},
        )

        def __post_init__(self) -> None:
            if self.process_res < 14:
                raise ValueError("process_res must be >= 14")

    @classmethod
    def requirements(cls, options: Options) -> Requirements:
        name, url, size, licence = METRIC3D_CHECKPOINTS[options.variant]
        # The default: the table's file at its canonical target, so known=True
        # (known fails closed: anything the YAML chose stays False).
        if options.weights is None:
            assets = (UrlAsset(key="weights",
                               target=f"{METRIC3D_DIR}/{options.variant}/{name}",
                               url=url, size_bytes=size,  # or sha256=...
                               license=licence, known=True),)
        else:  # the user's own file: checked inside the root, never downloaded
            target = under_root(options.weights)
            assets = () if target is None else (
                UserProvided(key="weights", target=target, option="weights",
                             hint=f"provide the Metric3D checkpoint at {options.weights}"),
            )
        return Requirements(
            assets=assets,
            code=(METRIC3D_REPO,),
            packages=("torch", "mmengine"),  # import names (illustrative)
            extra=METRIC3D_EXTRA,
        )

    @property
    def needs_intrinsics(self) -> bool:
        return True  # metric scale comes from the focal

    def _load(self, ctx: LoadContext) -> None:
        weights = (default_weights(self.options.variant)
                   if self.options.weights is None
                   else self.options.weights.expanduser())
        if not weights.is_file():  # checked before any framework import
            if under_root(weights) is not None:
                raise not_prepared("weights", "checkpoint", weights)
            raise ConfigError(f"weights: file not found: {weights}")
        # Framework imports (torch, the clone's modules) go only here and in the
        # methods called after `_load`, never at module level.
        ...  # put the clone on sys.path, build the network, load `weights`,
        # set self._geometry (frame -> map geometry)

    def _predict(self, rgb: np.ndarray) -> DepthResult:
        ...  # infer, convert to metres, return DepthResult(depth_m, K, sky, geometry)
```

What each part buys:

- `requirements(options)` reads only options and fixed tables: the UI and
  `--prepare` know the file, its size, licence and target before anything is
  loaded. A path the user sets inside the root becomes `UserProvided`
  ("provide this file"); outside the root it is not declared and `_load`
  reports it.
- `_load` checks the file **before** importing torch and names `--prepare`
  (`adapters._weights.not_prepared`), so a missing file is a clean
  `ConfigError` (exit 2), never a download at run time.
- Options are `frozen, slots, kw_only`; a subclass of a base options class
  with its own `__post_init__` calls the base explicitly (see the layered
  options rule in `core/components.py`). `ui` metadata keys (`label`, `unit`,
  `min`, `max`, `step`, `advanced`, `help`) are all optional and reach the
  editor unchanged as the field's `hint`; a path option typed `Path`
  (or `Path | None`) joins the server's path policy by itself.

### 7.2 The extra in `pyproject.toml`

```toml
[project.optional-dependencies]
# Metric3D (adapters/depth/metric3d.py): only what infer does not bring
model-metric3d = [
    "mmengine>=0.10",  # illustrative: whatever the upstream code imports
]
```

Add it so the lock is updated without touching the environment, and commit
both files:

```bash
uv add --no-sync --optional model-metric3d 'mmengine>=0.10'
```

If the new requirements cannot be resolved together with `infer` (for
example, a package that needs `numpy>=2` while `depth-anything-3` needs
`numpy<2`), the lock itself fails; see section 8 before forcing anything.

### 7.3 Register the module

Add one line to `ADAPTER_MODULES` in `core/registry.py`:

```python
    "av_perception_lab.adapters.depth.metric3d",
```

Tests: a unit test of `requirements()` (default, own file inside and outside
the root) in `tests/test_adapter_requirements.py`, and a GPU contract test
under `tests/gpu/`. Update `DESCRIPTION.md` (a new model and a dependency)
and record the real URL, size and licence in the task file.

### 7.4 What appears with no web edits

Nothing under `web/` or `src/av_perception_lab/web/` changes:

- the catalog (`GET /api/catalog`) lists `depth.metric3d` with its options,
  the hints above and the docstring's first paragraph; `--list-models` shows
  `metric3d: variant='vit-small', weights=None, process_res=616`;
- a config with `depth: {type: metric3d}` gets a status row with
  `source` "example.org; git · https://github.com/YvanYin/Metric3D",
  `license`, `target` "metric3d/vit-small/metric_depth_vit_small.pth;
  _code/Metric3D" and `size_bytes` 150000000;
- before the extra is declared in `pyproject.toml` the row is `failed` (red
  dashed): "extra 'model-metric3d' is not declared in pyproject.toml";
  once it is declared (and locked) the row is `missing` (yellow dashed with
  the alert icon) until every layer is ready;
- **Prepare model** clones the repo and downloads the checkpoint; the
  packages layer offers **Install packages** after a clean dry-run, or shows
  `conflict` with uv's reason.

---

## 8. Dependency conflicts: the five resolution paths

A *conflict* means the packages a new model needs cannot be installed in the
environment the other models use: the resolver finds no common versions
(numpy 1 vs 2, two torch builds, a package pinned by both at incompatible
versions), or the only plan replaces a package the others need. Only path 1
is implemented; paths 2–5 are documented options for when a real model needs
them.

### Path 1 — Detect before installing *(IMPLEMENTED)*

**What it does.** Section 6.2: the dry-run with every installed extra plus
the new one decides; a failure or a plan that would remove or replace a
package becomes the state `conflict` with the resolver's reason, and the
environment is not touched (no lock written, no package installed).

**When to use.** Always: it is the guard in front of every other path, and it
is enough as long as every model's extra is compatible with `infer`.

**Cost.** One `uv sync --dry-run` per status check of a model with a missing
per-model extra (cached per pyproject/lock modification time, one run shared
by concurrent checks, killed after 180 s).

**Limits.** It never *resolves* anything: a conflicting model stays not ready
until someone applies one of the paths below. It only sees conflicts the lock
expresses (a runtime incompatibility that the package metadata does not
declare passes the dry-run). Its output format is uv's; the reason is
extracted from the first `error`/`×` line onward.

**How it plugs in.** `core/provision/packages.py` (`detect_extras`,
`dry_run`, `plan_packages`), `engine.with_package_plan` (the state),
`engine.install_packages` (the install), the CLI's `--install-packages`, the
web's packages jobs.

### Path 2 — Mutually exclusive extras via `[tool.uv] conflicts` *(NOT implemented)*

**What it is.** uv can lock extras that cannot be installed together, as long
as they are declared mutually exclusive; it then resolves each side
separately in the same `uv.lock`:

```toml
[tool.uv]
conflicts = [
    [ { extra = "model-a" }, { extra = "model-b" } ],
]
```

**When to use.** Two *per-model* extras disagree with each other (model A
needs `x<2`, model B `x>=2`) but each is compatible with `infer`. One of them
is installed at a time.

**Cost.** Low on disk (one environment); a switch reinstalls the differing
packages (seconds to minutes). A pipeline using A and B together is
impossible.

**Limits.** It does not help when the conflict is with `infer` itself (every
model needs `infer`; an extra conflicting with it cannot be installed in the
base environment at all, see paths 3–4). The lock grows with every fork.

**How it would plug in.** Today path 1 already *reports* it: the dry-run
names every installed extra plus the new one, uv refuses to sync two extras
declared as conflicting, and the row becomes `conflict` with uv's message.
What would change:

- `plan_packages` would read `[tool.uv] conflicts` from `pyproject.toml` and,
  when the new extra conflicts with a detected one, return a new action
  (e.g. `switch`) listing the extra that would be removed instead of
  `conflict`;
- the install would run an **exact** sync naming the target set (every
  detected extra minus the conflicting one, plus the new one, without
  `--inexact`), after a dry-run whose removals are shown and confirmed —
  today any planned removal is refused by design;
- the UI would show **Switch to model B** with the list of packages that go
  away; the base extra could never be the one removed;
- runs of configs that need the removed extra would turn not ready (the
  status already detects it).

### Path 3 — One environment per profile, used by the worker subprocess *(NOT implemented)*

**What it is.** A *profile* is a set of extras installed into its own
environment, e.g. `_testbed/envs/<profile>/` through `UV_PROJECT_ENVIRONMENT`
(the notebook's `_testbed/venv-infer` already works this way). The web server
stays torch-free and keeps running in the base environment; only the
**worker subprocess**, which is the one process that loads models, runs with
the profile's interpreter.

**When to use.** A model conflicts with `infer` (or with another model) and
cannot be fixed by pinning, but a whole pipeline can run in one environment
built for it (e.g. a model that needs a different torch).

**Cost.** Disk and time per profile: a full GPU stack is about 9 GB (the
`infer` + `web` environment `_testbed/venv-infer` measured 9.0 GB with
`du -sh` on 2026-11-01; uv hardlinks
from its cache when the cache and the environment are on the same file
system, which reduces it); each profile is one `uv sync` to build and to keep
in step with the lock. A config can only mix models of one profile.

**Limits.** The packages check runs in the server's environment, so it would
no longer tell the truth about another environment; one run cannot combine a
model from profile A with one from profile B (that needs path 4); GPU memory
is unaffected (still one process per run).

**How it would plug in.**

- The worker command is already injected: `web/app.py:default_worker_argv`
  returns `[sys.executable, "-m", "av_perception_lab.web.worker", CONFIG]`
  and `JobQueue` starts it as a subprocess. It would pick the profile's
  interpreter instead (`<env>/bin/python`), chosen from the extras the
  config's components declare (`Requirements.extra`, already known per row).
- `plan_packages` / `detect_extras` would run *inside* the target
  environment (a small `python -c` probe of `importlib.metadata` and
  `find_spec` with that interpreter) instead of in the server's process.
- The install command would gain `UV_PROJECT_ENVIRONMENT=<env>` (the
  subprocess environment is already built in `packages._subprocess_env`), and
  creating a profile would be an explicit, confirmed action like Install
  packages; the base stack of a profile, like today's, would be the one-time
  documented command.
- The CLI needs nothing new: `UV_PROJECT_ENVIRONMENT=<env> uv run --extra
  infer --extra <x> av-perception-lab --config …`.
- Status rows would carry the profile so the UI can say which environment a
  config runs in.

### Path 4 — The conflicting model served as its own process *(NOT implemented)*

**What it is.** The model runs in its own environment as a long-lived
subprocess (or a local server) that receives frames and returns results; the
pipeline talks to it through a thin adapter that is a normal `Component`
subclass (`DepthEstimator`, `Detector`…), so the rest of the pipeline does
not change.

**When to use.** A pipeline must combine models whose environments cannot
coexist (different torch/CUDA builds, a different Python version), and paths
2–3 do not apply.

**Cost.** Highest: a second CUDA context (hundreds of MB of GPU memory) and a
copy of the model in another process; per-frame serialisation (a 1920×1080
RGB frame is ~6 MB; a 504×280 float32 depth map ~0.56 MB), best done through
shared memory; a protocol, start-up, health checks, timeouts and crash
handling to maintain; a second environment to build (as in path 3).

**Limits.** Latency per frame; errors cross a process boundary (they must be
turned into `ConfigError`/`ContractError` with the server's message); the
model's own options must be serialisable.

**How it would plug in.**

- A generic "remote" adapter base in `adapters/` whose `_load` starts the
  server with the other environment's interpreter (process group, killed in
  `_close`, like the web worker and the uv subprocesses), and whose
  `_predict` / `_refine` / `_segment` send the frame and rebuild the shared
  contract (`DepthResult`, `Instances`, …), validated by the base classes as
  today.
- `requirements(options)` would declare the main-environment side only (no
  heavy packages); the other environment would need a new requirement layer
  or asset kind ("environment": its extras and interpreter path, checked by a
  probe, created by an explicit confirmed action), registered like the
  existing kinds so the engine and the web need no model-specific edits.
- `DeviceMonitor` and `metrics.json` would have to account for the second
  process's GPU memory.

### Path 5 — Manual pinning or forking *(NOT implemented as a feature; always available by hand)*

**What it is.** Make the requirements compatible by editing them:

- pin a version of the shared dependency that satisfies both sides in the
  model's extra, or add a constraint or override in `pyproject.toml`
  (`[tool.uv] constraint-dependencies`, `override-dependencies`) when the
  upstream metadata is stricter than the code needs;
- pin the upstream package by git commit in `[tool.uv.sources]` (as done
  for `rfdetr` and `depth-anything-3`);
- fork the upstream repository to relax its pins, or use its code as a
  pinned clone on `sys.path` (a `CodeRepo`, as TwinLiteNet+ does) instead of
  installing it as a package with conflicting dependencies.

**When to use.** The conflict comes from over-strict metadata (an upper bound
the code does not need) or from one package that both sides can agree on;
usually the cheapest first attempt.

**Cost.** Maintenance: every override is a claim the project now owns
(re-check it when updating the lock); a fork must be rebased on upstream
fixes. Changing a pin of the base stack is a dependency change
(`DESCRIPTION.md`, and on the GPU host the one-time base command again).

**Limits.** An override can hide a real incompatibility that only shows at
run time (the dry-run trusts the overridden metadata). It cannot reconcile
two genuinely different binary builds (two torch/CUDA versions): that is
path 3 or 4.

**How it plugs in.** No code change: after editing `pyproject.toml` and
running `uv lock` (commit both), the next status check runs path 1's
dry-run; a clean one turns `conflict` into **Install packages**. A future
feature could only add a note to the conflict message (e.g. naming the
packages in the resolver's reason) pointing here.

### Choosing a path

| Situation | Path |
|---|---|
| A new model's extra is compatible with `infer` and with the other extras | 1 (nothing to do: Install packages) |
| Over-strict metadata, or one shared package both sides accept | 5 |
| Two per-model extras disagree with each other, both fine with `infer` | 2 |
| A model needs a stack incompatible with `infer`, used in its own pipelines | 3 |
| Models whose stacks cannot coexist must run in the same pipeline | 4 |
