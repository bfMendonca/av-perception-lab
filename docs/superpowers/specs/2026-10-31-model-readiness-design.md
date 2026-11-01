# Model readiness ("Prepare model") — design

Date: 2026-10-31. Status: draft for approval (nothing built).
Language: this document and every document it asks for are in **English**
(user decision, 2026-10-31). Existing project docs are in pt-BR; see "Open
decisions".

## Intent and why

The project is a study bench for perception stacks built only from open-source
algorithms and models. New models (detectors, depth, refiners, segmenters)
will keep arriving, each as a subclass of the base classes in `core/`. Today
every adapter finds its weights in its own way, so a friend who clones the
repository from GitHub and loads a sample YAML cannot run it without reading
code:

- Hugging Face models (SAM 2.1, SegFormer): `hf_cache` + `local_files_only`;
  the error prints a download command.
- DA3: downloads at run time if the checkpoint is missing.
- YOLO: a `.pt` path written in the YAML.
- RF-DETR: `~/.roboflow/models`.
- TwinLiteNet+: a manual `git clone` under `~/dev`.
- The sample YAMLs carry paths of the author's machine (`~/dev/...`).

Goal: **clone, load a sample YAML, click "Prepare", run.** The cost of telling
the system how to obtain a model's requirements moves to the implementation of
that model, so adding a model never requires touching the interface.

## Goals

1. **One declaration per component class** of everything it needs to run, in
   three layers: *weights* (Hugging Face, plain URL, Google Drive, extensible
   source types), *external code* (a pinned `git clone`), *Python packages*
   (a `pyproject.toml` extra).
2. **One interface action for every model:** "Prepare model" (and "Prepare
   all" for a whole pipeline). The UI never knows what a model needs.
3. **Default locations are decided by the adapter**, under `weights/<model>/`
   (already git-ignored) and a git-ignored code folder. A path in the YAML
   becomes an optional override. Sample YAMLs stop carrying machine-specific
   paths.
4. **Readiness is checkable without importing torch** (file/folder existence,
   `importlib.util.find_spec`), so the server and the UI work on a fresh clone.
5. **Automatic as far as safe:** weights and code downloads run without
   further questions; Python packages are installed only after a conflict
   check passes and the user confirms (see decisions).
6. **Conflicts are detected, never created:** if installing a model's
   packages cannot be resolved against the current environment, the model is
   shown as *not ready: conflict* with the reason, and the environment is not
   touched.
7. **Visible states** on the pipeline flow (user's spec): a model that is not
   ready is drawn with a **yellow dashed outline and a small alert icon at the
   top-right corner**; states are *ready*, *not ready*, *preparing* (progress),
   *failed* (red dashed) and *conflict*.
8. **Documentation:** a new `docs/model_environments.md` (English) describes
   every way to resolve a dependency conflict (see "Documentation deliverables")
   and the README gets a TODO entry pointing to it.

## Non-goals

- No automatic resolution of dependency conflicts in this version (detect and
  report only). The richer paths (mutually exclusive extras, one environment per
  profile for the worker, a model served as its own process, manual pinning)
  are documented, not built.
- No entering of access tokens in the UI. Gated models use the credentials
  `huggingface_hub` already reads (`HF_TOKEN`, `hf auth login`); the UI explains
  what to do.
- No model catalogue or search; only models that exist as adapters.
- No training, conversion or quantisation of weights.
- No downloading from addresses the client typed: sources are fixed in code.

## Anti-goals (failure even if it "works")

- A "Prepare" that can break the working environment (the incident to avoid:
  `uv sync` without the right extras removes the GPU stack).
- The interface growing model-specific code or lists of model names.
- A friend's first run failing on a path from the author's machine.
- Silent multi-GB downloads or silent acceptance of a licence.

## Constraints

- `core/` and `adapters/` must not import the web layer; the CLI must keep
  working without the `web` extra. The requirements declaration lives in
  `core/` (framework-free) and is read by the CLI too (`--prepare`
  command planned as a thin wrapper, so the web is optional).
- Framework imports stay inside adapter `_load` methods (existing rule).
- Same future-proof rule as the web interface: adding a model or a source type
  needs zero edits under `web/` or `src/.../web/`.
- Downloads must be atomic (`.part` then rename), resumable when the source
  supports it, cancellable, and idempotent (a second Prepare is a no-op).
- One Prepare at a time (disk and network), independent from the GPU job queue
  (downloads must not block processing).
- Repository process (`AGENTS.md`): a `docs/task_DDDD.md` per task before code;
  `DESCRIPTION.md`, `README.md`, `CHANGELOG.md` updated in the same commit;
  never commit to `main`.

## Approaches considered

- **A. Declared requirements per class, three layers (chosen).** Each component
  class declares assets, code and packages; one generic engine checks and
  prepares. Cost of a new model stays inside the model.
- **B. Each model as a separate plugin package** with its own requirements.
  Best isolation, too heavy for a study repository.
- **C. Keep per-adapter ad hoc handling** and only surface errors in the UI.
  Does not deliver "just click".

Within A, the conflict question was discussed and decided: **v1 = detect
conflicts before installing (path 1)**; the other resolution paths are
documented for later (see below).

## Decisions made by the user

- Option A; maximum automation; model authors own the "where and how".
- Source types must be agnostic (Hugging Face, Google Drive, URL, …).
- Not-ready models drawn as yellow dashed outline + alert icon top-right.
- Sharing path: clone from GitHub + a YAML + click Prepare.
- v1 handles conflicts by detection only; document the other paths in the
  README TODO and in a detailed `docs/` document, in English.

## Decisions made by the designer (the user may revise)

- **Declaration shape.** A frozen dataclass hierarchy in `core/`:
  `Requirements(assets: tuple[Asset, ...], code: tuple[CodeRepo, ...],
  packages: tuple[str, ...])`, returned by a classmethod/hook on `Component`,
  `requirements(options) -> Requirements` (default: empty). Asset kinds are
  registered (like adapters), so a new source type is one new class plus a
  registration: `HuggingFaceAsset(repo, files|patterns, revision)`,
  `UrlAsset(url, sha256?)`, `GoogleDriveAsset(file_id, sha256?)`; `CodeRepo(url,
  commit)`. Each asset has `size_bytes?`, `license`, `gated: bool`, and a
  `target` relative to the model's directory.
- **Default locations.** `weights/<family>/<model>/...` for assets and
  `weights/_code/<name>/` for clones, configurable by one server/CLI setting
  (`--weights-root`, env `DDD_WEIGHTS_ROOT`). The adapter computes the
  default path; an explicit option in the YAML overrides it. Sample YAMLs are
  edited to drop machine paths.
- **Status model.** `ModelStatus(state: ready|missing|conflict|failed|
  preparing, layers: {assets, code, packages}, missing: [...], size_bytes,
  message)`. Computed without torch. `ready` requires all three layers.
- **Packages layer in v1.** Packages are declared as a `pyproject.toml` extra
  name per model family (e.g. `model-metric3d`). The check is `find_spec` of the
  declared import names. If missing, the engine runs a dry-run resolution
  (`uv`'s dry-run against the lock with the union of already-installed extras
  plus the new one). Clean dry-run → the UI offers **Install packages**
  (explicit confirmation, runs `uv sync` with the *union* of extras as a job,
  then shows "restart the server"); conflict → state `conflict` with the
  resolver's message and a link to `docs/model_environments.md`. Never removes
  the existing stack.
- **Engine and API.** A framework-free engine in `core/` (`check(components)`,
  `prepare(component, progress, cancel)`); the web layer exposes
  `GET /api/models/status` (for the loaded pipeline), `POST
  /api/models/prepare` (model key or "all"), progress via the same SSE style as
  jobs, separate queue from GPU jobs. The CLI gets the `--prepare`
  flag (see "Calls made after the builder check").
- **Security.** Prepare requests carry only a component *type name + options
  already validated*; the sources come from code. Downloads go only under the
  weights root (path containment, same helpers as the web hardening). Licence
  and size are shown before **Prepare all**; gated assets require the user to
  have accepted the licence on the source site (message with link).
- **UI.** Card states as in the mockup
  (`docs/superpowers/specs/mockups/model-readiness-mockup.html`): global banner
  with **Prepare all** and total size; inspector section "Weights & requirements"
  listing layers, source, size, licence, target folder, a **Prepare model**
  button, an *Advanced* override of the location; progress bar on the card;
  failure shows the reason and a retry.
- **Migration.** All current adapters (DA3, RF-DETR, YOLO, HF SAM, Ultralytics
  SAM, SegFormer, TwinLiteNet+) declare their requirements; existing behaviour
  with explicit paths in YAML keeps working. DA3 stops downloading at run time
  (it becomes a `missing` state with a Prepare action).

## Documentation deliverables (all English)

1. `docs/model_environments.md`: the model "recipe" format and how to add a
   model with weights/code/packages; and, for dependency conflicts, **each
   resolution path in detail** — (1) detect before installing *(implemented)*,
   (2) mutually exclusive extras via `[tool.uv] conflicts`, (3) one environment
   per profile used by the worker subprocess, (4) the conflicting model served as
   its own process, (5) manual pinning/forking — with when to use each, cost,
   limits, and how it would plug into this design.
2. README: a "TODO" entry listing paths 2–5 and linking to the document, plus a
   "Model weights" usage section (English).
3. `DESCRIPTION.md` and `CHANGELOG.md` updated per `AGENTS.md`;
   `docs/web_extending.md` gains a section on requirements for new models.

## Calls made after the builder check (the user may revise)

- **`infer` stays as it is.** All current adapters declare the existing
  `infer` extra as their packages layer; only *new* models get their own small
  extras (e.g. `model-metric3d`). No split of `infer`, so the lock and the
  notebook workflows in `CLAUDE.md` do not change and the "uv sync
  removes the GPU stack" risk is not reopened.
- **The base stack is not installed by the server.** On a fresh clone the
  prerequisite is one documented command (`uv sync --extra infer --extra web`,
  README). If the base stack is missing, the UI shows that exact command (with
  a copy button) instead of a Prepare button; per-model extras (small) are
  installable by the "Install packages" action after the dry-run passes. How the
  engine knows which extras are installed: from `importlib.metadata` /
  `find_spec` of each extra's declared import names (no environment variable
  guessing); the `uv sync` command is built from the extras detected as
  satisfied plus the new one.
- **Requirements may depend on validated options, never on `load()`.**
  `requirements(options)` is called with the already-validated options (YOLO
  weights name, RF-DETR variant, HF repo id). Add an asset kind
  `UserProvided(option)`: a file the user supplies (custom `.pt`/checkpoint);
  it has no source, is checked for existence only, and Prepare reports "provide
  this file" instead of downloading.
- **Adapters load from explicit paths where the library allows.** RF-DETR:
  Prepare downloads the checkpoint to `weights/rfdetr/` and the adapter passes
  `from_checkpoint`; YOLO: the downloaded path; Hugging Face adapters and DA3
  keep the HF cache layout, with `cache_dir` = `weights/hf` (so the author's
  existing `weights/hf` keeps working untouched). Prepare never moves or
  deletes existing weights (`~/.roboflow`, `weights/hf`).
- **TwinLiteNet+:** its pretrained weights ship inside the pinned clone
  (`pretrained/small.pth`, commit `90f1b86` per `CLAUDE.md`), so a `CodeRepo`
  covers code and weights; the clone moves to `weights/_code/TwinLiteNetPlus/`
  and the `~/dev` location keeps working as an override.
- **Google Drive:** v1 ships the extension point with Hugging Face, URL and git
  sources only; a Drive source is added when a model needs it (documented in
  `docs/model_environments.md`).
- **A pipeline with a not-ready model.** Loading and validating the YAML still
  works. The UI blocks Process with "N models are not ready — Prepare"; the
  server refuses a run at submit time with 409 listing the models; the CLI
  fails before loading anything with a message naming `--prepare`. DA3 stops
  downloading at run time (hard error with that hint).
- **CLI shape:** a flag, `det-depth-distance --config X.yaml --prepare`
  (prepares and exits; no subcommands, the existing invocations are
  unchanged). The earlier "subcommand" wording in this document is replaced.
- Gated models (e.g. SAM 3) keep their current behaviour: Prepare reports "gated
  — accept the licence at <link> and set `HF_TOKEN`".

## Left to the builder

- Exact class names, module layout (suggest `core/requirements.py` and
  `core/provision/` for the engine and source types), and the sources'
  libraries (`huggingface_hub` is present; `gdown` or plain HTTP for Drive).
- How to run `uv`'s dry-run robustly and parse its error for the message.
- Resumable-download details, checksum policy (optional `sha256`).
- Splitting into tasks (suggested order: requirements model + status check
  → source types + engine + CLI → adapter migration + sample YAMLs → web API
  and UI states → packages layer with dry-run → documentation).

## Resolved decisions

- **Language (user, 2026-10-31): everything in English, for consistency.**
  This covers the documentation this feature produces *and* the
  `docs/task_DDDD.md` files, whose template headings are translated
  (Description, Scope, Plan, Impact analysis, Verification, Dependencies;
  closing sections "Implementation details" and "Test notebook"). The
  implementation includes a one-line amendment to `AGENTS.md` stating that
  rule. Migrating the *existing* pt-BR documents (README, DESCRIPTION,
  CHANGELOG, older task files) is not part of this feature; it is proposed as a
  separate follow-up so this change stays reviewable.
- Google Drive is not in v1 (extension point only), see the calls above.
