# Changelog

Todas as mudancas relevantes do av-perception-lab serao documentadas neste arquivo.

---

## [Unreleased] MCAP input

ROS 2 MCAP files as a pipeline input (task_0033).

### Added

- `source.mcap` + `source.topic`: read a `sensor_msgs/Image` or
  `CompressedImage` topic (rgb8, bgr8, rgba8, bgra8, mono8, jpeg, png) from an
  indexed or unindexed `.mcap` (`io/mcap_reader.py`, `McapSource`). `Frame.t`
  uses `header.stamp` when it is positive and non-decreasing, else the log
  time; repeated timestamps are dropped (`duplicates_dropped`).
- `camera.calibration.from_source`: K and distortion from the topic's
  `CameraInfo` (raw: K/D; rectified: P).
- `rectify` section and stage: removes the lens distortion with the
  `CameraInfo` before every other stage.
- The annotated `video.mp4` of an MCAP run has variable frame rate
  (`TimestampedVideoWriter`, PyAV); `metrics.json["video"]` describes the
  source.
- Web: `.mcap` drop, import and library entries, `GET /api/videos/{id}/topics`,
  a topic picker, a "Use the CameraInfo" tick and a Rectification lane.
- Evaluation loads MCAP runs and uses their recorded frame times; runs with
  `source.start > 0` are refused.

### Changed

- `mcap`, `mcap-ros2-support` and `av` are base dependencies (`av` left the
  `web` extra); `FrameSource` gained `input_name`, `nominal_fps`,
  `n_frames_total` and optional members; `camera.calibration.path` is optional.

---

## [Unreleased] Delete a run

Delete a run from the web history (task_0032).

### Added

- `GET /api/runs/{id}/size` (`{bytes}`) and `DELETE /api/runs/{id}` (204; 404
  for an unknown or unsafe id, 409 while the run's job is queued or running).
  A `job-<id>` record (failed job without a folder) is deleted by removing its
  job record. Symlinks are never followed.
- Web runs view: trash button in the history rows and in the run detail, with
  a confirmation dialog that shows the space the run uses. Deleting closes the
  run if it is open and drops it from the compare selection; a 409 shows an
  inline message.
- `JobQueue.holds_run(folder)`, `history.run_size` and `history.delete_run`.

---

## [Unreleased] Rename to av-perception-lab

Project renamed (task_0031): it is now a perception test bed for autonomous
driving, not just a distance script.

### Changed

- Python package `det_depth_distance` -> `av_perception_lab`; distribution and
  console scripts `det-depth-distance[-calibrate|-evaluate|-web]` ->
  `av-perception-lab[-calibrate|-evaluate|-web]`; front-end package
  `av-perception-lab-web`.
- Environment variables `DDD_*` -> `AVPL_*` (`AVPL_WEIGHTS_ROOT`, `AVPL_UV`, ...).
  There is no compatibility alias: **migration** = rename the variables in your
  shell/profile and run `uv sync` again (the editable install changes name).
- Project summary (README, DESCRIPTION, CITATION, `pyproject.toml`) rewritten
  as a perception test bed. Task files `docs/task_0001`-`task_0030` and
  `docs/superpowers/` keep the old name (historical records).

## [Unreleased] Config presets

Web config presets (task_0030): backend and front-end.

### Added

- `GET /api/presets`, `GET /api/presets/{id}`, `PUT /api/presets` and
  `DELETE /api/presets/{id}`: built-in presets from `configs/` (manifest
  `configs/presets.yaml`) plus the user's own, stored without `source`/`output`.
- `av-perception-lab-web --presets-root` (default `_testbed/presets`) and
  `WebPaths.presets_root`; `create_app(builtin_presets_dir=...)`.
- Web editor: **Presets** menu (built-in and own presets, delete for own),
  **Save as preset...** dialog with overwrite confirmation, and the
  `loadPreset` store action, which keeps the current video and output folder.
- Preset names with U+2028/U+2029 are refused (they would break the `# name:`
  comment); a dump failure of the saved data is a 422, not a 500. A built-in
  name clash is a 422 (only an existing user preset is a 409 and can be
  overwritten).
- Docs: presets in README, `docs/reference.md`, `DESCRIPTION.md`.

---

## [2026-11-03] One command to start the web interface

`utils/run_web.sh` checks the prerequisites, builds the front-end when
`web/dist` is missing, picks the extras and starts `det-depth-distance-web`
(task_0029).

### Added

- `utils/run_web.sh [--gpu] [--rebuild] [--dry-run] [SERVER OPTIONS...]`: needs
  `uv` and, only to build, Node.js 22.12 or newer. It never selects fewer extras
  than the environment already has (an installed PyTorch keeps `infer`, because
  `uv run` syncs exactly); `--gpu` installs the GPU stack on the first run.
  Other arguments go to the server (`--port 8010`).
- `tests/test_run_web_script.py`: dry runs with fake `uv`, `node` and `npm`.

---

## [2026-11-01] Google Drive source: TwinLiteNet+ weights are prepared automatically

Prepare (CLI `--prepare` and the web "Prepare model") now downloads the
TwinLiteNet+ pretrained weights from the upstream Google Drive folder
(task_0028).

### Added

- Asset kind `GoogleDriveAsset(file_id, sha256, manual)` (`gdrive`) and its
  source `GoogleDriveSource`: the fixed endpoint
  `drive.usercontent.google.com/download?id=<id>&export=download&confirm=t`,
  the URL source's rules (`.part` + atomic rename, Range/If-Range resume,
  size and sha256 checked before the rename, size cap, free-disk check,
  cancel, never over an existing file), redirects only over https to
  `drive.usercontent.google.com`, `drive.google.com` or
  `*.googleusercontent.com`. An HTML answer (confirmation or quota page) is
  refused with "Google Drive returned a confirmation or quota page instead of
  the file; try again later or download it manually from <folder>" and never
  saved.

### Changed

- TwinLiteNet+ `pretrained/<variant>.pth` (nano/small/medium/large) is a
  `GoogleDriveAsset` with pinned file id, size and sha256 (verified against
  the author's copies), licence MIT; `known` (web-preparable) only at
  `<weights root>/_code/TwinLiteNetPlus/pretrained/<variant>.pth` of the
  pinned clone. A `weights` file with another name stays user-provided; the
  `repo`/`weights` overrides keep working.
- The provisioning engine fetches a file whose target lies inside a declared
  clone's folder only once that clone is ready, so a download never blocks
  `git clone`.
- `UrlSource` and `GoogleDriveSource` share one download implementation
  (`_HttpFileSource`); the URL source behaves as before.

---

## [2026-11-01] Remove remote GPU host references and the sync script

The project is now documented as running only on the local machine
(task_0027).

### Removed

- The rsync push/pull sync script (`utils/`) for the former GPU server and
  every command, section, table row and gotcha about syncing to, running on
  or pulling outputs from that host, including its address.

### Changed

- `utils/gpu_pending.sh` header and final message are local ("run on the GPU
  machine"); behaviour is unchanged.
- Comments in `pyproject.toml` and `tests/gpu/conftest.py`, and incidental
  mentions in the docs and task records, name no host; technical results are
  kept.

---

## [2026-11-01] Model readiness: final fixes

The whole model-readiness branch was reviewed; this wave fixes what the
reviews found before merge (task_0026).

### Fixed

- A model whose only missing file is a manual download (TwinLiteNet+
  `pretrained/<variant>.pth`, a custom YOLO `.pt`) is no longer offered as
  Prepare: the row has `reason: "user_provided"`, its message carries the
  hint, Prepare all lists it under `skipped`, and a Prepare that fetched the
  rest ends `done` instead of `failed`. `--prepare` reports "provide this
  file: <hint>" with state `missing` (exit 1 only because it is not ready).
- An existing file whose size or sha256 differs from the declaration (a
  fine-tuned checkpoint saved under a release name) is never downloaded over,
  in the CLI either: "exists with a different size at <target>; move or
  delete it, or point the config at another file".
- `Asset.known` / `CodeRepo.known` fail closed (default False); the adapters
  vouch only for the canonical target (`ultralytics/<release name>`, the `hf`
  cache for the shipped repos, the RF-DETR table, the pinned clone). The web
  queue runs the engine with `allow_unknown=False`, re-checked when the job
  runs; the CLI passes `allow_unknown=True`.
- The run worker gets the server's weights root as an absolute
  `DDD_WEIGHTS_ROOT`, so a run that passes the 409 check passes the CLI gate
  with any `--weights-root`.
- URL downloads stop when the body exceeds the expected size and check free
  disk space first.
- Redaction and scrubbing: path boundaries include backtick, `<`, `>`, `{`,
  `}`; 422 replies and job exception texts are redacted and scrubbed; plan
  messages go through `relative()`; the scrub no longer rewrites prose such as
  "basic requirements" and knows `Authorization:` headers and `github_pat_`
  tokens; the code source shows host and path only.
- The weights root may not be `/`, the home folder, or contain the project,
  the active environment or the run outputs (CLI exit 2, web start-up error).
- `is_repo_id` checks the `org/name` shape before any file-system probe.
- A transient dry-run failure during an install names its reason.
- The dry-run timeout test no longer flakes (3 s instead of 1 s).

### Changed

- Prepare all asks for confirmation: a dialog lists the downloads (name,
  source, size, licence) and the total, and separately the files you must
  provide; only Download starts it.
- After a packages install the note says new runs use the packages (each run
  starts a fresh worker); a server restart is only for an upgraded package the
  server had already imported.
- README "From a fresh clone": prerequisites, start with
  `rfdetr_da3.yaml`, and the manual TwinLiteNet+ download.

---

## [2026-10-31] Model readiness documentation

How a model declares what it needs, how a fresh clone prepares it and how a
dependency conflict can be resolved, in English (task_0025).

### Added

- `docs/model_environments.md`: the model recipe (`requirements(options)`,
  the assets/code/packages layers, the Hugging Face, URL, user-provided and
  code-repo kinds and how to register a new source kind), the weights root
  (`--weights-root`, `DDD_WEIGHTS_ROOT`), default locations and YAML
  overrides (the bare-name rule for YOLO-style `weights` and the `./name.pt`
  edge), the states and how the editor draws them, the CLI (`--prepare`,
  `--install-packages`, the not-ready gate with exit code 2) and web flows
  (status, Prepare model, Prepare all, unknown sources, gated models and
  `HF_TOKEN`, the manual TwinLiteNet+ download), the packages layer
  (one-time base stack, detect, dry-run, install, `uv.lock`, `DDD_UV`,
  transient failure vs conflict, installs vs runs, restart), a worked example
  of adding a Metric3D depth model, and the five dependency-conflict
  resolution paths (only detection is implemented; mutually exclusive
  extras, one environment per profile, a model in its own process and
  manual pinning/forking are documented options).
- README: "From a fresh clone" steps in "Model weights" (base stack, build,
  open a sample, Prepare all; the CLI equivalent; weights root; gated
  models; TwinLiteNet+), TODO entries for task_0024, task_0025 and the open
  resolution paths 2-5.
- `docs/web_extending.md`: requirements of a new model need no web edits.

---

## [2026-10-31] Packages layer: detect, dry-run, install

Missing Python packages are no longer just reported: the installed extras
are detected, a `uv sync --dry-run` checks a model's own extra against them,
and only a clean dry-run can be installed, after an explicit confirmation
(task_0024).

### Added

- `core/provision/packages.py`: `detect_extras` (`pyproject.toml` extras whose
  distributions are installed and whose declared import names are found,
  without importing them), `dry_run` (`uv sync --dry-run --locked --inexact`
  with `--extra` for every detected extra plus the new one; the resolver's
  reason, project-relative), `install` (same command without `--dry-run`, own
  process group, streamed output, cancellable) and `plan_packages`
  (`ready` / `base` / `install` / `conflict` / `invalid`).
- `engine.install_packages` and `with_package_plan`: a failed dry-run is the
  model state `conflict`, pointing to `docs/model_environments.md`.
- CLI `--install-packages` (with `--prepare`): runs the clean dry-run, then the
  install.
- Web: status rows carry `packages`; `POST /api/models/prepare` with
  `packages: true` runs an Install packages job (`output` SSE events); the
  status body has `restart_required`. The inspector offers **Install
  packages** with a confirmation dialog stating the command, shows the `uv`
  output, and asks for a server restart afterwards.

### Changed

- The CLI gate names `uv sync --extra infer` when only packages are missing
  (`--prepare` does not install packages); `--prepare` reports the packages
  plan.
- A missing base stack (`infer`) shows the one-time command
  `uv sync --extra infer --extra web` with a copy button; the server never
  installs it. A dry-run that would remove or replace a package is a
  conflict.

### Fixed (review round 1)

- uv not found, a timeout or a cancel is reported as `failed` (naming
  `DDD_UV`), never as a cached `conflict`.
- Concurrent status checks share one dry-run; a dry-run timeout kills uv's
  whole process group.
- Install packages and runs exclude each other (409 both ways).
- The install runs only the plan the user confirmed (same import names,
  same command); a base-stack plan is refused even by a direct API call.
- Credentials are scrubbed from uv output.

---

## [2026-10-31] Model readiness in the web editor

The pipeline editor shows which models are ready, prepares the missing ones
with one click and keeps Process locked until every model is ready
(task_0023).

### Added

- Card states on the pipeline flow: a model that is not ready has a yellow
  dashed outline and an alert icon at the top-right corner; preparing shows a
  progress bar fed by the Prepare SSE events; failed is red dashed with its
  reason; conflict and unknown states use the warning style with the server's
  message; ready cards stay solid.
- `ReadinessBanner`: "N models need preparing · X GB" with **Prepare all**, the
  list of models, overall progress and Cancel; "Check again" when the status
  cannot be fetched.
- Inspector section "Weights & requirements": layers, missing items, size,
  location, **Prepare model** / **Retry**, the copyable
  `det-depth-distance --config <file> --prepare` for sources the web may not
  download, and the module's path option as the location override.
- `web/src/api/models.ts` (client of `/api/models/*`) and
  `web/src/models/useModelStatus.ts` (debounced status, prepare jobs, per-model
  progress and failures, shared through a context).

### Changed

- Process is disabled with "N models are not ready — Prepare them first."
  while a model is not ready; a 409 from `POST /api/runs` opens a dialog that
  lists the models (rerun shows them in the notice).
- Status rows carry `source` (`Asset.describe_source()`), `license`,
  `target` (weights-root-relative), `reason` and `preparable`; the UI prefers
  them to parsing messages. `GET /api/models/prepare` lists Prepare jobs.
- The banner counts only what Prepare all can fetch: "N models need preparing
  · X to download".

### Fixed

- A model that finished in a multi-model Prepare no longer flashes back to
  "Not ready"; a reload re-attaches to a running Prepare (progress, Cancel);
  an SSE reconnect never rewinds progress; removing or reordering modules
  during a Prepare never shows its state on another card.

---

## [2026-10-31] Web API for model status and Prepare

The web server reports which models of a config are ready, downloads the
missing ones on request and refuses runs that would fail on missing weights
(task_0022).

### Added

- `POST /api/models/status`: per-model state, layers, missing items, size and
  message for a config, plus the total size to download; computed like
  `--prepare`, without importing torch or loading a model.
- Prepare queue (`web/models.py`, `PrepareQueue`): `POST /api/models/prepare`
  (`models: "all"` or `[{stage, index}]`), `GET /api/models/prepare/{id}`,
  SSE `GET /api/models/prepare/{id}/events` with per-model progress and
  `DELETE /api/models/prepare/{id}`; one Prepare at a time, independent from
  the GPU job queue, engine in a worker thread.
- `--weights-root` server flag (`DDD_WEIGHTS_ROOT` still honoured); the root
  may not overlap the library or `--out-root`.
- `core.options.PATH_RESOLVER` field metadata: path checks outside an adapter
  see the file the adapter opens (YOLO / Ultralytics SAM `weights`).

### Changed

- `POST /api/runs` and rerun answer 409 `{models_not_ready: [...]}` when a
  model of the config is not ready.
- `--asset-root` checks accept paths inside the weights root, and bare YOLO /
  Ultralytics SAM weight names resolve under `<weights root>/ultralytics/`, so
  every shipped config passes them.

- `det-depth-distance --config X.yaml` (without `--prepare`) stops with exit
  code 2 before loading any model when a declared model is not ready, listing
  what is missing and naming `--prepare`.

### Fixed

- Messages sent to web clients no longer garble when the weights root is
  relative (`--weights-root weights`) and no longer rewrite sibling folders
  such as `weights-old`: only the absolute root, at a path boundary.

### Security

- From the web, Prepare only downloads sources the adapters vouch for
  (`Asset.known`; Hugging Face repos of `HF_FACTS`). Any other repo typed in a
  config reads "unknown source: prepare it from the CLI (--prepare)": 422 when
  named, skipped (and reported) by `models: "all"`. The CLI is unchanged.
- Model request bodies refuse unknown fields (no client URL or path);
  sources and targets come from the adapters' code and stay inside the weights
  root; messages sent to clients show `<weights root>` instead of its absolute
  path.

---

## [2026-10-31] Adapter requirements and portable sample YAMLs

Every adapter declares what it needs, so a fresh clone can run `--prepare`
on a sample YAML (task_0021).

### Added

- `requirements(options)` on every adapter (`extra="infer"`, import names as
  packages): DA3 and the Hugging Face refiners/SegFormer declare a
  `HuggingFaceAsset` (weights file + `*.json`) in `<weights root>/hf`; RF-DETR a
  `UrlAsset` per variant (`weights/rfdetr/`, library URL table, sizes and
  Apache-2.0); YOLO and Ultralytics SAM a `UrlAsset` for the
  `ultralytics/assets` v8.4.0 names (YOLO26 COCO and Objects365, SAM, SAM 2.x,
  MobileSAM) and `UserProvided` for any other file; TwinLiteNet+ the pinned
  clone (`CodeRepo` at `weights/_code/TwinLiteNetPlus`, commit `90f1b86...`) and
  `pretrained/<variant>.pth` as `UserProvided` (Google Drive upstream).
- `cli.config_requirements()`; `--prepare` also prepares the focal model of
  `camera.estimate` when it is used.

### Changed

- Defaults resolve under the weights root: `hf_cache: null` = `<weights
  root>/hf` (was the user-wide HF cache); RF-DETR loads
  `weights/rfdetr/<file>` through `pretrain_weights=` (was `~/.roboflow/models`,
  which is left untouched); TwinLiteNet+ `repo` = `weights/_code/TwinLiteNetPlus`
  (was `~/dev/TwinLiteNetPlus`, still valid as an explicit `repo`); a bare file
  name in YOLO / Ultralytics SAM `weights` resolves under
  `weights/ultralytics/` (a path with a directory part keeps its old meaning).
  Explicit paths behave as before.
- DA3 no longer downloads at run time (`local_files_only=True`); a missing
  checkpoint is a `ConfigError` naming `--prepare`, raised before
  `depth_anything_3` is imported. Missing default weights of every adapter
  name `--prepare` too.
- Every `configs/*.yaml` drops machine paths (`~/dev/...`): no `hf_cache`,
  bare YOLO file names, no TwinLiteNet+ `repo`.

---

## [2026-10-31] Provisioning engine and `--prepare`

Obtain what a model declares (task_0020). Adapters declare nothing yet
(`task_0021`), so every model is reported ready for now.

### Added

- `core/provision/`: framework-free engine `prepare(req, root, progress,
  cancel) -> ModelStatus` (no-op when ready; failures end as `failed` with a
  readable reason; cancel through a `threading.Event`) and a source registry
  (`register_source`, duplicate kinds raise `ValueError`).
- Built-in sources: Hugging Face (`snapshot_download` into the `weights/hf`
  cache layout with `allow_patterns`, lazy import; gated, offline and 404
  messages), URL (`urllib`, `<target>.part`, atomic rename; accepted only when
  the declared size or sha256 matches or the full `Content-Length` arrived;
  resume with `Range` + `If-Range` only when size or sha256 is declared;
  redirects only to http(s), no https -> http), git (clone + pinned checkout
  under `weights/_code/<name>`, per-command timeout, killed with its helpers
  on cancel; an existing clone is moved to the pin, never deleted) and
  user-provided ("provide this file", never downloads). Every write path is
  confined to the weights root.

### Changed

- `CodeRepo.commit` must be a hex commit id (7-64 characters); branches and
  tags are refused.
- CLI flag `--prepare`: prepares every component of the YAML, prints a status
  table, exit code 0 only when all are ready (130 on Ctrl-C, partial
  downloads kept for resume).

---

## [2026-10-31] Model requirements declaration and torch-free status check

Foundation of the "model readiness" work (task_0019); no behaviour change yet.

### Added

- `core/requirements.py`: `Requirements` (assets, pinned code clones, package
  import names + extra), registered asset kinds (`HuggingFaceAsset`,
  `UrlAsset`, `UserProvided`, extensible with `register_asset_kind`),
  `weights_root()` (`set_weights_root`, `DDD_WEIGHTS_ROOT`, repo `weights/`)
  and `check()` returning a `ModelStatus` without importing torch.
- `Component.requirements(options)` hook (default: nothing required).
- Truncated, hash-mismatched and `.part` downloads are never ready; targets
  cannot leave the weights root (`..`, absolute paths, symlinks).

### Changed

- New documentation and `docs/task_DDDD.md` files are written in English
  (`AGENTS.md`); `docs/task_TEMPLATE.md` was translated.

---

## [2026-10-31] Endurecimento da interface web, desligamento e jobs

Onda final de correcoes da revisao do branch da interface web (task_0018).

### Seguranca

- **Host/Origin**: so nomes de loopback, `--host` e `--allowed-host` sao
  atendidos; POST/PUT/PATCH/DELETE de outra origem dao 403 (`--dev-origin`
  libera um front-end de desenvolvimento).
- **`source.video` sem `video_id`** precisa estar numa `--data-root` ou ser
  original da biblioteca (422 sem revelar se o caminho existe).
- **Campos `Path` do YAML** (pesos, rotulos, repo, cache HF, calibracao,
  fonte; descobertos pelos esquemas) nao podem apontar para a biblioteca nem
  para as saidas; `--asset-root` restringe a pastas do servidor, com aviso ao
  ouvir fora do loopback sem ela.
- **Envio**: 413 (tamanho, `Content-Length`, `meta` de 64 MiB), limites de
  cabecalho e partes (400), 429 para mais de 2 envios, 507 sem disco; sufixo
  guardado so de video (`.bin` para o resto).
- **ffprobe/ffmpeg/PyAV**: so arquivos locais e demuxers de video aceitos;
  timeout na previa.
- **Arquivos servidos** com tipo explicito, `nosniff`, `CSP: sandbox` e
  download fora dos tipos de midia; YAML com expansao de aliases limitada;
  corpos JSON ate 1 MiB; ids de execucao com `.` inicial nao resolvem.

### Corrigido

- Ctrl-C com SSE aberto nao trava mais (3 s); jobs pendentes terminam
  `cancelled` e registros orfaos viram interrompidos na partida.
- Recarregar a pagina mantem os jobs (`GET /api/jobs`, reassinatura,
  `Last-Event-ID`; job desconhecido apos reinicio e encerrado).
- O historico mostra o nome do video da biblioteca, nao `original.mp4`.

### Alterado

- `uniqueTag` removido do front-end (o servidor define `output.tag`);
  `JOBS_DIR`/`VIDEO_ID_ENV` em `web/paths.py`; teste de nomes fixos tambem
  cobre `web/src`.

---

## [2026-10-31] Historico e comparacao de execucoes na interface web

Quinta e ultima entrega da interface web (task_0017): resultados de cada
execucao e comparacao de duas.

### Novo

- **`web/compare.py`**: resumo por execucao (mediana da distancia, jitter
  bruto e suavizado por trilha, vazao, quadros, caixas, trilhas), serie de
  distancia por trilha reduzida a <= 1500 pontos com o tempo no video
  anotado, diff plano do YAML efetivo, comparacao com deltas e lista dos
  artefatos da pasta; pastas antigas sem metrics.json/config.yaml/colunas
  novas sao toleradas.
- **Rotas** `GET /api/compare`, `GET /api/runs/{id}/{summary,series,artifacts}`,
  `GET /api/runs/{id}/files/{caminho}`, `GET /api/runs/{id}/metrics`,
  `POST /api/runs/{id}/reopen` e `POST /api/runs/{id}/rerun`.
- **Tela Runs & comparison**: historico filtravel, video anotado, cartoes de
  indicadores com delta assinado (melhora/piora), diff do YAML, grafico de
  distancia (B sobre A, bruta em cinza; clique salta o video), artefatos
  descobertos com registro de visoes (`registerArtifactView`), arvore
  generica do metrics.json, Reopen in editor e Rerun; rota
  `#/runs/<id>/vs/<id>`.

### Alterado

- `evaluation.run_data.read_columns` extraido de `load_run` (mesmo leitor do
  CSV para a avaliacao e a web).
- `RunRecord` ganha `video_id` e `log_tail`; `has_config` tambem conta o
  `config` do metrics.json.
- `POST /api/runs` e o rerun compartilham a mesma validacao.

---

## [2026-10-31] Biblioteca de videos, trecho e processamento na interface web

Quarta entrega da interface web (task_0016): o botao Process passa a rodar
o trecho escolhido de um video enviado ou importado.

### Novo

- **Biblioteca de videos** (`web/library.py`): envio multipart gravado em
  disco por blocos, importacao de caminho do servidor so dentro de
  `--data-root` (padrao `data`), deduplicacao por sha256, metadados por
  ffprobe, miniatura, recusa de arquivo que nao e video ou truncado sem deixar
  nada, conversao de `.h264` + `.meta.json` para `.mkv` VFR com os
  `SensorTimestamp` e previa MP4 em segundo plano.
- **Rotas** `/api/videos`, `/api/videos/roots`, `/api/videos/import`,
  `/api/videos/{id}` e `/api/videos/{id}/{file|preview|thumb}`.
- **Painel de video** acima do editor: soltar em qualquer ponto da pagina,
  player imediato, barra de trecho com duas alcas, stride e estimativa ao
  vivo, Process e Process entire video, confirmacao com pouca memoria de GPU,
  painel de jobs (fila, progresso, quadro ao vivo, Cancel, cauda do log) e
  vista minima da execucao em `#/runs/<id>`.

### Alterado

- `POST /api/runs` recusa `video_id` fora da biblioteca (ou com o arquivo
  sumido) e, com `video_id`, sempre usa o original do video como
  `source.video`.
- Importacao aceita caminho relativo a raiz do repo (ainda so dentro de
  `--data-root`); original importado que some ou muda vira "File missing" e
  reimportar o mesmo conteudo conserta a entrada.
- `det-depth-distance-web --data-root` passa a ter padrao `data`.
- `av` (PyAV) declarado no extra `web`.

---

## [2026-10-31] Editor visual de pipeline da interface web

Terceira entrega da interface web (task_0015): o front-end em `web/` (React +
TypeScript + Vite, em ingles, so tema escuro), servido pelo FastAPI.

### Novo

- **Editor por pistas**: estagios na ordem do catalogo, um cartao por modulo,
  "+ Add" com os tipos registrados, inspetor com campos gerados do esquema
  (slider, select, switch, texto/caminho, lista, grupo, "Advanced"
  recolhido) e "General settings" para as secoes gerais.
- **YAML**: abrir (botao, Ctrl+O, arrastar), salvar com aviso unico de perda
  de comentarios, `extras` preservados, diff contra o ultimo YAML aberto.
- **Validacao** no servidor com debounce de 400 ms; erros no campo pelo
  caminho pontilhado, no cartao e na barra superior.
- **A prova de futuro**: estagio, tipo de campo e versao de catalogo
  desconhecidos caem em alternativas genericas (pista generica, editor YAML
  bruto, aviso); registro de extensoes `registerStageView` e
  `registerFieldControl`.

### Alterado

- `create_app(..., static_dir=...)` serve `web/dist` em `/` com fallback de
  SPA (nunca para `/api` nem para arquivo ausente em `assets/` ou com
  extensao); sem build, so a API; `det-depth-distance-web --static-dir`.
  `.gitignore` ganha `web/node_modules/` e `web/dist/`.
- `Stage.group` (opcional) no registro de estagios e no catalogo: estagios
  consecutivos do mesmo grupo dividem uma pista; `tracking` e `smoothing`
  formam "Tracking & filter". A ordem nativa das pistas passa a ser fonte,
  detectores, agregacao, refinadores, profundidade, rastreio e filtro,
  semantica, movimento.

---

## [2026-10-31] Execucoes, fila, progresso ao vivo e historico da interface web

Segunda entrega da interface web (task_0014): processar de fato, sem que o
servidor importe torch.

### Novo

- **Worker** (`python -m det_depth_distance.web.worker CONFIG.yaml`): roda
  `cli.run` num subprocesso e imprime um JSON por linha (`started`, `frame`,
  `done`, `error`; saida 2 em `ConfigError`, 1 nos demais). Grava
  `config.yaml`, `web_run.json` e `preview.jpg` na pasta da execucao.
- **`JobQueue`**: um executor, um job por vez, eventos para SSE com replay,
  cancelamento por `kill()` do grupo de processos, queda do worker vira
  `failed` com o fim do log.
- **Historico** (`scan_runs`, `load_run`): le as pastas de `out_root`,
  inclusive antigas e incompletas; ids validados contra `../` e symlinks.
- Rotas `/api/runs`, `/api/jobs`, `/api/gpu` (memoria livre via subprocesso).

### Alterado

- `cli.run(cfg, on_frame=None)` e `VideoSink(..., on_frame=None)`; sem
  `on_frame` o comportamento e identico.

---

## [2026-10-31] API de esquema e validacao da interface web

Primeiro passo da interface web: um backend FastAPI, sem torch, que descreve
o pipeline para um editor visual e valida o YAML. A task_0013 cria o pacote
`det_depth_distance.web` e o registro de estagios; a regra central e que um
modelo ou estagio novo aparece na interface sem editar nada em `web/`.

### Novo: Registro de estagios e dicas de interface

- **`Stage`, `register_stage`, `stages`** em `core/registry.py`: chave do YAML,
  titulo, familia de adaptador ou dataclass de configuracao, cardinalidade e
  ordem; os estagios nativos sao registrados no fim de `config.py`.
- **Metadados `ui`** nos campos das dataclasses (rotulo, unidade, faixa,
  passo, avancado, ajuda), semeados nos campos do Kalman e do rastreador IoU.

### Novo: Pacote `web` (extra `web`)

- `GET /api/catalog` e `POST /api/config/{validate,parse,dump}`; erros de
  validacao com o caminho pontilhado (`detectors[0].treshold`), nunca 500.
- `WebPaths` e `safe_join` (recusa `..`, caminho absoluto e symlink para fora).
- `det-depth-distance-web` (127.0.0.1 por padrao) e `docs/web_extending.md`.

### Alterado

- `config.config_from_mapping` extraido de `load_config` (a validacao da web
  usa o mesmo codigo); `uv.lock` ganha o extra `web` (`python-multipart`).

---

## [2026-10-23] Movimento por trilha e previsao de trajetoria

Ate aqui o pipeline media so a distancia de cada objeto e, com a suavizacao,
a taxa de variacao dessa distancia. Um veiculo cruzando a frente a distancia
constante aparecia como parado, todas as classes usavam o mesmo filtro e
nada previa onde o objeto estaria nos proximos segundos. A task_0009 cria o
estagio opcional `motion`: para cada trilha ele mede a posicao do objeto no
plano (lateral e a frente, em metros), estima velocidade, rumo e taxa de
giro com as incertezas, escolhe o modelo de movimento pela classe e preve a
trajetoria. A trajetoria aparece no video, num painel de vista superior e
num arquivo novo. Sem a secao `motion` as saidas sao as de antes.

### Novo: Estagio `motion` (pacote `motion/`, numpy puro)

- **Medida no plano:** o ponto de referencia da caixa (centro da base) e
  retroprojetado com as intrinsecas do mapa de profundidade e a distancia da
  instancia; a covariancia combina a incerteza da distancia e a do pixel.
- **Modelos por classe:** `cv` (velocidade constante) e `ctrv` (velocidade e
  taxa de giro constantes, EKF com aquecimento em `cv`), registrados e
  listados em `--list-models`; `classes` mapeia o rotulo canonico
  (`vehicle`, `person`, ...) para um modelo nomeado.
- **Previsao:** um ponto a cada `step_s` ate `horizon_s`, cada um com
  covariancia.
- **Reinicio:** troca de ID ou salto (NIS acima de `nis_gate`) reinicia o
  filtro; trilha sem medida por `reset_after_s` recomeca.
- **Ego-motion opcional:** provedor `csv` (`t,x,y,yaw` no relogio do
  video); com ele o filtro roda no mundo e a velocidade sobre o chao vai
  para `motion_abs_speed_mps`. Fora do intervalo do arquivo vale a pose da
  borda, com um aviso e a contagem `clamped_frames`.

### Novo: Saidas

- **`detections.csv`:** 16 colunas `motion_*` no fim (61 no total), vazias
  sem o estagio.
- **`trajectories.jsonl`:** uma linha por quadro e instancia com a
  trajetoria prevista e as covariancias.
- **`metrics.json`:** `pipeline.motion` e, por trilha, `motion` (modelo,
  percentis de velocidade, NIS medio, reinicios); tempo `motion`.
- **Video:** `v <m/s>` no rotulo, trajetoria a partir da base da caixa com
  opacidade decrescente e elipses de incerteza no chao, painel BEV opcional
  (`render.bev_inset`) e linha de HUD com os modelos e o horizonte.

### Novo: Exemplo `configs/rfdetr_da3_motion.yaml`

- Rastreamento e suavizacao de `rfdetr_da3_track.yaml`, mapa de rotulos
  COCO para obter `vehicle`, `cv` para todas as classes (sem ego-motion o
  giro do carro proprio aparece como movimento dos objetos), layout
  `overlay` com painel BEV.

### Novo: Exemplo `configs/rfdetr_sam2_lanes_motion_da3.yaml`

- Stack completa na máquina com GPU: RF-DETR + mascaras SAM 2.1 + DA3 +
  TwinLiteNet+ `lanes` (faixas pintadas fora da distancia) + rastreamento,
  Kalman, `motion` e painel BEV. Primeira execucao nos videos da Raspberry
  Pi (`RPI_0000`/`RPI_0001`, 1296x972 a 25 fps).

### Como usar

1. Ligar com os padroes: `--set motion={}` (exige `tracking`).
2. Ou rodar o exemplo: `uv run --extra infer python -m det_depth_distance --config configs/rfdetr_da3_motion.yaml`.
3. Com ego-motion: `--set 'motion.ego_motion={type: csv, path: ego.csv}'`.

### Detalhes tecnicos

- `rel_floor` 2 % e `nis_gate` 30 foram medidos nas trilhas da task_0001
  (Passo 0): NIS medio ~2 no video em linha reta e 3 a 6 vezes menos
  reinicios do que com o gate de qui-quadrado 99,9 % (13,8).
- `temporal.py` ganhou `distance_sigma` e `white_accel_cov`, extraidos sem
  mudanca de comportamento; o `MetricsSink` rele `Pipeline.describe()` no
  fechamento para registrar contadores finais.

### Observacoes importantes

- Sem ego-motion tudo e relativo a camera; a estimacao do movimento proprio
  pelo video fica para uma task futura (task_0011).
- Validado na máquina com GPU (`utils/gpu_pending.sh` passo `20-motion` e
  smoke `gpu`); a conferencia visual dos videos segue pendente.

### O que nao muda

- `smoothing`, o rastreador IoU e as colunas e o rotulo da suavizacao.
- Execucoes sem `motion`: colunas `motion_*` vazias, sem
  `trajectories.jsonl` e sem `motion` no resumo por trilha.

---

## [2026-10-15] Configuracao leve da stack completa para GPU de 4 GB

Os exemplos de `configs/` usam modelos grandes, pensados para o servidor
GPU (96 GB). A stack completa nessa forma (dois detectores, SAM 2.1 large,
SegFormer-B5) foi estimada em 5 a 6,5 GB e nao cabe na GPU de 4 GB do
notebook de desenvolvimento. A task_0010 cria um exemplo leve que mantem
todas as etapas (deteccao com mascara, profundidade metrica, rastreamento
com suavizacao e duas segmentacoes semanticas) com modelos menores. Nenhum
codigo do pipeline muda. O consumo de VRAM e uma estimativa; a medicao real
no notebook fica pendente.

### Novo: Exemplo `configs/yolo_light_da3.yaml`

- **Deteccao:** um unico YOLO26s-seg (COCO). A mascara vem do proprio
  detector, entao nao ha refinador SAM.
- **Profundidade:** DA3METRIC-LARGE, o unico DA3 metrico fora da familia
  giant.
- **Rastreamento e suavizacao:** os mesmos valores de
  `rfdetr_da3_track.yaml` (IoU + Kalman).
- **Semantica:** TwinLiteNet+ `small` para area trafegavel e faixas
  (`lanes`) e SegFormer-B0 Cityscapes para ceu e vegetacao (`scene`).
- **Video:** layout `overlay`, com mascaras, camadas semanticas, distancia
  suavizada e velocidade.

### Como usar

1. Liberar cerca de 13 GB no disco (extra `infer` ~9 GB + pesos ~3 GB + video).
2. Seguir a secao "Execucao no notebook (GPU de 4 GB)" do README: ambiente
   isolado, download dos pesos, smoke e medicao.
3. Rodar `uv run --extra infer python -m det_depth_distance --config configs/yolo_light_da3.yaml`.

### Atualizacao: Documentacao

- **README:**
  - nova secao "Execucao no notebook (GPU de 4 GB)";
  - quatro linhas novas na tabela de pesos;
  - nota sobre o extra `infer`, que nunca vai no `.venv`;
  - smoke de GPU agora com 10 exemplos.
- **AGENTS.md, CLAUDE.md e DESCRIPTION.md:** comandos e cuidados do notebook.

### Detalhes tecnicos

- **SegFormer:** `nvidia/segformer-b0-finetuned-cityscapes-640-1280`, com
  `size: [640, 1152]`. E a escala de teste do checkpoint, que foi treinado e
  testado em imagem inteira com lado curto de 640 px. O B2 foi descartado
  porque o decodificador dele tem cerca de 3 vezes mais ativacao.
- **Pesos:**
  - `yolo26s-seg.pt` (v8.4.0, 23 467 933 B) em `~/dev/ultralytics/_testbed/weights`;
  - DA3 e SegFormer-B0 em `weights/hf`;
  - TwinLiteNet+ `small.pth` (594 777 B) no clone `~/dev/TwinLiteNetPlus`.
- **Pico estimado:**
  - cerca de 2,4 GB alocados na fase principal (o DA3 domina);
  - cerca de 3 GB na fase da focal (`da3-large-1.1`, 16 quadros), que roda
    e e liberada antes dos modelos principais e nao aparece no `metrics.json`.
- **Alavancas:** `--set camera.estimate.frames=8` ou `--set camera.focal_px=<px>`.
- **Testes:**
  - `test_light_config_is_the_planned_light_stack` em `tests/test_config.py`;
  - caso `yolo_light_da3` no smoke de GPU, que pula onde faltam os pesos leves.

### Observacoes importantes

- O DA3 baixa um checkpoint ausente durante a execucao (os adaptadores
  `transformers` nao baixam): baixe os pesos antes.
- Com `UV_PROJECT_ENVIRONMENT=_testbed/venv-infer` exportado, todo `uv run`
  precisa de `--extra infer`; sem ele, o sync exato remove a stack.

### O que nao muda

- Os demais exemplos de `configs/` e o codigo do pipeline continuam iguais.

---

## [2026-10-14] Smoke test ponta a ponta do pipeline sem GPU

As tasks 0002 a 0007 foram verificadas so com testes unitarios no laptop,
com a máquina com GPU fora do ar, e as execucoes reais ficaram espalhadas em
seis arquivos de task. A task_0008 cria um smoke test que roda o sistema
inteiro pelo ponto de entrada real da CLI sem GPU, um smoke com os modelos
reais para o servidor e um unico script que executa, em ordem, todas as
pendencias de GPU. Nenhum comportamento da execucao muda; a avaliacao passa
a recusar duas execucoes com o mesmo nome de pasta.

### Novo: Smoke local (`tests/smoke/`, marcador `smoke`)

- **Cena sintetica:** video 320x180 de 12 quadros em tons de cinza com
  planos de profundidade conhecida (estrada, ceu, um carro que anda e dois
  aneis), gerado no `tmp_path`.
- **Dubles deterministicos:** detector (com e sem mascaras), refinador pela
  hierarquia `BoxPromptRefiner` e segmentador semantico que "veem" o quadro
  pela paleta; a profundidade e o adaptador `DA3Depth` real com so a rede
  trocada. Entram no registro so durante o teste.
- **Matriz em pares:** 9 execucoes cobrem todo par de niveis de agregacao,
  regiao, refinador, exclusao semantica, rastreamento e calibracao com
  correcao de distorcao; cada distancia do CSV e conferida contra o valor
  analitico da cena.
- **Invariante:** com tudo desligado as colunas da task_0002 ficam
  identicas, mesmo com etapas neutras ligadas.
- **Encadeamento:** calibrar tabuleiros renderizados, processar o video
  distorcido com a calibracao (com e sem correcao) e avaliar contra
  referencias derivadas da geometria.
- Roda por padrao no `uv run pytest` (~8 s; `-m smoke` seleciona so ele).

### Novo: Smoke com os modelos reais (`tests/gpu/test_smoke_gpu.py`)

- Uma execucao de 5 quadros em 9 dos exemplos de `configs/` (cada familia
  de adaptador real), em processo novo,
  conferida pela mesma verificacao estrutural do smoke local
  (`tests/run_outputs.py`); peso ausente pula o caso com a dica de download.

### Novo: Script de pendencias de GPU (`utils/gpu_pending.sh`)

- Roda os 19 passos pendentes das tasks 0002 a 0008 na ordem
  de dependencia, com um log por passo e um `summary.tsv`.
- **Como usar:** `utils/gpu_pending.sh run`, na maquina com GPU.
- Opcoes `list`, `--dry-run`, `--from`, `--only`, `--skip-long`; downloads
  so com `--allow-downloads` e confirmacao por item; os da task_0004 (fora
  do repositorio) so sao impressos.

### Corrigido: Avaliacao com execucoes de mesmo nome

- `det-depth-distance-evaluate` juntava em silencio os resultados de duas
  execucoes com o mesmo nome de pasta (pastas de `out_root` diferentes
  criadas no mesmo segundo); agora sai com erro pedindo para renomear uma
  delas (por exemplo com `output.tag`).

### Observacoes importantes

- A execucao na máquina com GPU continua pendente (servidor fora do ar).

---

## [2026-10-14] Calibracao da camera e avaliacao contra distancias de referencia

A distancia do DA3METRIC e proporcional a focal da camera, e a focal
estimada variou cerca de 8 % entre dois videos gravados com a mesma lente;
alem disso, nao havia como medir o erro das distancias. A task_0007
adiciona a calibracao da camera (ferramenta com tabuleiro ou ChArUco e uso
da calibracao na execucao, com correcao opcional da distorcao) e a
avaliacao das execucoes contra distancias medidas em campo. Sem
`camera.calibration`, a execucao e a mesma de antes.

### Novo: Ferramenta de calibracao (`det-depth-distance-calibrate`)

- **`calibrate`:** calibra a partir de um video (decodificado exatamente
  como nas execucoes) ou de imagens (`--images`), com tabuleiro (padrao
  9x6 cantos de 25 mm) ou ChArUco; usa um quadro a cada 0,5 s, ate 60
  vistas, descarta vistas atipicas e refaz o ajuste, informa RMS,
  desvios de K e a cobertura do quadro e confere os criterios de aceitacao
  do protocolo (RMS, cobertura, desvio de fx).
- **`board`:** gera o alvo em PNG para impressao em escala real.
- **`inspect`:** mostra os metadados de camera do video (modelo, lente,
  focal equivalente a 35 mm, que o `ffprobe` nao mostra) e confere uma
  calibracao contra o video.
- Arquivo de calibracao no formato padrao do OpenCV (YAML ou JSON), com os
  metadados da gravacao de calibracao.

### Novo: Calibracao na execucao (`camera.calibration`)

- **Precedencia:** calibracao > `focal_px` > estimativa.
- **K completa** (fx, fy, ponto principal), reescalada para o tamanho do
  video quando a proporcao e a mesma.
- **`check_lens`:** compara modelo, lente e focal 35 mm da calibracao com
  os do video antes de carregar qualquer modelo (`error`, `warn` ou `off`).
- **`undistort`:** corrige a distorcao de cada quadro antes de todos os
  componentes (`alpha` controla o recorte).
- **`cross_check`:** tambem estima a focal e registra as duas.
- Exemplo: `configs/rfdetr_da3_calibrated.yaml`.

### Novo: Avaliacao (`det-depth-distance-evaluate`)

- Arquivo de referencias por video (modelo ficticio em
  `configs/references/example_references.yaml`): objeto selecionado por
  caixa, ponto ou `track_id` num intervalo de quadros ou de tempo,
  distancia constante ou serie (RTK), instrumento e incerteza, alcance ou
  profundidade z.
- Erros absolutos e relativos (media, mediana, p90, RMSE, vies, fracao
  dentro de 2x a incerteza) por execucao, detector e estimador, e por faixa
  de distancia, classe, regiao (mascara/caixa), produtor da mascara e
  referencia; cobertura, trocas de ID e focal implicita.
- Varias execucoes lado a lado; saidas `eval.json`, `eval.md` e, com
  `--plot`, graficos.

### Novo: Protocolo de campo

- `docs/refs/protocolo_calibracao_e_referencias.md`: como gravar o alvo,
  criterios de aceitacao da calibracao, como medir e sincronizar as
  referencias, como anotar e como ler o relatorio.

### Como usar

1. Gerar e imprimir o alvo: `uv run det-depth-distance-calibrate board --out board.png`.
2. Gravar o alvo no mesmo modo das gravacoes de campo e calibrar:
   `uv run det-depth-distance-calibrate calibrate CALIB.MOV --square-mm 25 --out configs/calibration/<camera>.yaml`.
3. Executar com `configs/rfdetr_da3_calibrated.yaml` (apontando para o
   arquivo de calibracao).
4. Avaliar: `uv run det-depth-distance-evaluate --refs REFS.yaml RUN [RUN ...]`
   (graficos: `uv sync --extra eval` e `--plot`).

### Detalhes tecnicos

- O projeto usa coordenadas de pixel continuas (centro do pixel em
  `i + 0,5`); o OpenCV usa `i`. A conversao (+0,5 em cx/cy) fica so na
  leitura/escrita da calibracao.
- ChArUco usa o `cv2.aruco` do `opencv-python` (`CharucoDetector` +
  `matchImagePoints` + `calibrateCamera`); nao e preciso
  `opencv-contrib-python`.
- Metadados de camera do iPhone (lente, focal 35 mm) ficam no nivel da
  trilha do MOV; um leitor proprio, so com a biblioteca padrao, le apenas o
  atomo `moov`.
- `metrics.json` ganha `frame_intrinsics`, `camera` e `video.transform`.
- Novo extra opcional `eval` (matplotlib, so para os graficos); o ambiente
  leve continua sem ele. Duas CLIs novas em `[project.scripts]`.

### Observacoes importantes

- Tudo foi testado localmente com dados sinteticos (tabuleiros
  renderizados com distorcao conhecida, execucoes ficticias). Ficam
  pendentes: a calibracao real do iPhone 14 Pro Max, a coleta de
  referencias de campo e a validacao na máquina com GPU (fora do ar durante a
  implementacao).
- A focal equivalente a 35 mm dos metadados nao serve como focal do video
  (o recorte da estabilizacao nao e registrado): e mostrada so como
  diagnostico.
- Com `undistort`, caixas do CSV e o video ficam em pixels corrigidos; as
  referencias continuam anotadas no video original e a avaliacao as
  transforma com a K e a distorcao gravadas no `metrics.json` da execucao
  (recalibrar depois nao muda a avaliacao de execucoes antigas).
- Na máquina com GPU, os graficos usam o `--extra infer` (ja traz o
  matplotlib); `--extra eval` sozinho removeria a stack GPU.

### O que nao muda

- Com `camera.calibration: null` (padrao), deteccoes, distancias, CSV e
  video sao os da task_0006; o `metrics.json` so ganha `frame_intrinsics`,
  `camera: {}` e `video.transform: null`.

---

## [2026-10-13] Rastreamento de objetos e suavizacao temporal da distancia

Cada quadro era processado isoladamente: a distancia de um mesmo objeto
oscilava de um quadro para o outro e nao havia identidade do objeto ao
longo do video. A task_0006 adiciona um rastreador opcional, que da a cada
objeto um ID estavel, e a suavizacao da distancia por trilha, que tambem
estima a velocidade relativa. Tudo vem desligado por padrao; desligado, os
resultados sao exatamente os de antes.

### Novo: Rastreamento (`tracking`)

- **Rastreador IoU proprio** (`tracking: {type: iou}`), em numpy puro:
  casa cada trilha com a deteccao de maior sobreposicao, prevendo a caixa
  por velocidade constante, e mantem a trilha viva por `max_age_s`
  (0,5 s) sem casamento.
- **Agnostico de classe** por padrao (os detectores alternam `car`/`truck`
  no mesmo objeto); com varios detectores em `concat`, cada detector tem
  as proprias trilhas; com fusao, uma trilha por objeto.
- **Parametros em segundos**, calculados pelo tempo real de cada quadro:
  funciona com `source.stride` (ate 5 a 30 fps numa reproducao offline
  dos `detections.csv` da task_0001; validacao na GPU pendente).

### Novo: Suavizacao por trilha (`smoothing`)

- **Filtros:** `kalman` (padrao; estima distancia, velocidade relativa e
  as incertezas), `ema` (media movel exponencial) e `median` (mediana
  movel), todos com janelas em segundos.
- **Incerteza da medida:** usa o desvio da caixa ja calculado, escalado e
  com piso relativo de 1 % da distancia.
- A distancia bruta continua no CSV, ao lado da suavizada.

### Novo: Saidas

- `detections.csv` ganha no fim `track_id`, `distance_smooth_m`,
  `distance_smooth_std_m`, `rel_speed_mps` e `rel_speed_std_mps` (vazias
  sem rastreamento/suavizacao).
- Video: rotulo com `#<id>` antes da classe e a velocidade no fim
  (distancia suavizada e velocidade configuraveis em
  `render.show_smoothed`/`render.show_speed`) e uma linha de HUD com o
  rastreador e o filtro.
- `metrics.json` ganha `tracks` (resumo por trilha: rotulo, duracao,
  cobertura, lacunas, faixa de distancia, jitter bruto e suavizado,
  percentis da velocidade), `components.tracker`, `pipeline.smoothing` e
  os tempos `track` e `smooth`.

### Novo: Exemplo de configuracao

- `configs/rfdetr_da3_track.yaml`: `rfdetr_da3.yaml` com rastreador IoU e
  Kalman, lado a lado com a profundidade.

### Como usar

1. `uv run --extra infer python -m det_depth_distance --config configs/rfdetr_da3_track.yaml`.
2. Trocar o filtro: `--set smoothing.filter=ema`.
3. Ajustar o rastreador: `--set 'tracking={type: iou, iou_thr: 0.5}'`;
   desligar com `--set tracking=null --set smoothing=null`.

### Detalhes tecnicos

- Ordem dos estagios: deteccao -> concordancia -> agregacao -> refinadores
  -> rastreador -> profundidade -> semantica -> elegibilidade -> distancia
  -> suavizacao.
- Novo contrato `Tracker` (mesmo ciclo de vida dos demais componentes);
  outro rastreador e uma subclasse registrada, outro filtro e uma entrada
  em `FILTERS`.
- Os rastreadores do Ultralytics (ByteTrack/BoT-SORT) nao foram usados:
  exigem o pacote `lap`, importam `torch`, contam tempo em chamadas e sao
  AGPL-3.0. Nenhuma dependencia nova.

### Observacoes importantes

- O ruido de profundidade e correlacionado no tempo: a suavizacao reduz o
  jitter quadro a quadro em 3,5 a 4,5 vezes (reproducao offline), mas pouco
  o erro de nivel, e
  nao corrige vies do modelo ou da focal. Acima de ~30 m a velocidade
  relativa nao e confiavel.
- Reproducao offline com as caixas e distancias dos videos da task_0001
  confirmou os numeros do planejamento (uma unica trilha para o veiculo
  lider; jitter 0,213 -> 0,060 m e 0,568 -> 0,127 m por quadro). A
  validacao na máquina com GPU (testes `-m gpu` e execucoes nos videos) ficou
  pendente: o servidor estava fora do ar durante a implementacao.

### O que nao muda

- Sem `tracking` e `smoothing`, deteccoes, distancias e video sao os
  mesmos; o CSV so ganha as cinco colunas no fim, vazias, e o
  `metrics.json` so ganha `components.tracker` e `pipeline.smoothing` nulos.

---

## [2026-10-13] Refinadores de mascara (familia SAM) e segmentacao semantica

Detectores so de caixas (RF-DETR, YOLO26 Objects365) nao tinham como medir
a distancia pela mascara do objeto, e pixels de fundo como estrada, ceu,
vegetacao e faixas pintadas entravam no calculo. A task_0005 adiciona
refinadores de mascara, que transformam a caixa de qualquer detector numa
mascara de instancia (familia SAM), e segmentadores semanticos, que marcam
classes de fundo por pixel para retira-las da distancia. Sem refinador e
sem segmentador, os resultados sao exatamente os de antes.

### Novo: Refinadores de mascara (`refiners`)

- **Familia SAM via `transformers`:** `sam2` (SAM 2.1, padrao dos
  exemplos), `edgetam` (EdgeTAM, leve), `sam_hq` (SAM-HQ), `sam` (SAM 1) e
  `sam3_tracker` (rastreador do SAM 3; peso com acesso restrito).
- **`ultralytics_sam`:** segundo backend, para MobileSAM e pesos `.pt` de
  SAM/SAM 2 do Ultralytics.
- **Politica de alvos:** `targets: missing` (padrao) refina so instancias
  sem mascara, preservando as mascaras nativas de detectores de
  segmentacao; `targets: all` refina todas para comparar. Uma mascara
  refinada vazia nunca substitui a existente.
- **Opcoes comuns:** `max_boxes`, `multimask`, `mask_threshold`, `score`
  (`iou` ou `object`) e `min_object_score` (so nas familias com score de
  objeto).

### Novo: Segmentacao semantica (`semantic`)

- **Varios segmentadores por execucao**, cada um com `name` obrigatorio:
  `segformer` (Cityscapes, 19 classes: estrada, calcada, vegetacao, terreno,
  ceu...) e `twinlitenet` (TwinLiteNet+: area trafegavel e faixas, clone
  externo).
- **Exclusao por namespace:** `distance.exclude_semantic: [scene.road,
  lanes.lane]`; os mapas nao sao fundidos e a exclusao e a uniao. Prefixo
  desconhecido e erro ao ler a configuracao; classe desconhecida, na
  carga do modelo.
- **Video:** uma camada translucida por mapa (`render.semantic_alpha`,
  `render.semantic_overlay`), por baixo das mascaras e caixas.

### Novo: Saidas

- `detections.csv` ganha no fim `mask_source` (quem produziu a mascara) e
  `mask_score` (qualidade prevista pelo refinador). Nenhuma coluna
  semantica.
- `metrics.json` ganha `mask_sources`, `refined`, `semantic` (fracoes por
  classe e fracao excluida, por mapa e da uniao), os tempos
  `refine.<nome>`/`semantic.<nome>` e as versoes de `transformers` e
  `timm`.
- `depth.npz` (com `save_depth`) ganha `semantic_labels_<nome>` e
  `semantic_classes_<nome>`, na grade do mapa de profundidade.

### Novo: Exemplos de configuracao

- `configs/rfdetr_sam2_da3.yaml`, `configs/yolo_o365_sam2_da3.yaml`,
  `configs/rfdetr_semantic_da3.yaml` e `configs/rfdetr_lanes_scene_da3.yaml`
  (TwinLiteNet+ e SegFormer juntos).

### Atualizacao: Dependencias

- `transformers>=5.1,<6` passa de dependencia transitiva a declarada no
  extra `infer` (versao travada inalterada, 5.19.0).
- `timm>=1.0.17` entra no extra `infer` (trava 1.0.30), so para o EdgeTAM.
  Nenhuma outra versao do lock muda.

### Como usar

1. Baixar os pesos no servidor (comando no README, secao "Pesos dos
   refinadores e do SegFormer"); a execucao nunca baixa nada.
2. `uv run --extra infer python -m det_depth_distance --config configs/rfdetr_sam2_da3.yaml`.
3. Trocar o refinador: `--set 'refiners=[{type: edgetam, hf_cache: weights/hf}]'`.
4. Semantica: `--config configs/rfdetr_lanes_scene_da3.yaml`; desligar a
   exclusao com `--set 'distance.exclude_semantic=[]'`.

### Detalhes tecnicos

- Hierarquia de refinadores: `MaskRefiner` (contrato do pipeline) ->
  `BoxPromptRefiner` (numpy puro: um embedding por quadro, caixas em lote,
  mascara de maior IoU previsto) -> `HFSamRefiner` (API SAM do
  `transformers`) -> familias finas. Acrescentar um modelo e uma subclasse
  e um registro.
- Ordem dos estagios: deteccao -> concordancia -> agregacao -> refinadores
  -> profundidade -> semantica -> elegibilidade -> distancia.
- Os rotulos semanticos sao reamostrados na grade do mapa de profundidade
  antes do teste de exclusao.
- A fusao (NMS/WBF) passa a registrar em `mask_source` o detector dono da
  mascara da instancia fundida.

### Observacoes importantes

- A validacao na máquina com GPU (testes `-m gpu`, download dos pesos e
  execucoes de comparacao entre refinadores e de exclusao semantica) ficou
  pendente: o servidor estava fora do ar durante a implementacao.
- SegFormer e TwinLiteNet+ foram treinados em cenas de rua; em cenas
  agricolas, revisar a lista de exclusao (vegetacao e terreno cobrem a
  cultura).
- Licencas: SAM 3 tem licenca propria e acesso restrito; os pesos SegFormer
  da NVIDIA tem termos proprios; o Ultralytics e AGPL-3.0.

### O que nao muda

- Sem `refiners` e sem `semantic`, deteccoes, distancias e video sao os
  mesmos; o CSV so ganha as duas colunas no fim, vazias.
- Mascaras de instancia continuam sem ser gravadas em disco.

---

## [2026-10-13] Segmentacao de instancias e distancia pela mascara

Objetos finos ou inclinados (pessoas, bicicletas, postes, implementos)
ocupam pouco da propria caixa, e o fundo dominava a estimativa de
distancia. A task_0004 adiciona modelos de segmentacao de instancias
(RF-DETR Seg e YOLO26-seg) e passa a medir a distancia sobre a mascara de
cada objeto sempre que o modelo a fornece. Detectores so de caixas
continuam com exatamente o mesmo resultado de antes.

### Novo: Modelos de segmentacao de instancias

- **RF-DETR Seg:** `variant: seg-nano`, `seg-small`, `seg-medium`,
  `seg-large`, `seg-xlarge` ou `seg-2xlarge` (COCO). Os pesos (~135-155 MB)
  sao baixados pela biblioteca no primeiro uso.
- **YOLO26-seg:** o adaptador `yolo` aceita pesos de segmentacao
  (`yolo26*-seg.pt` COCO, `yolo26*-objv1-seg.pt` com vocabulario
  Objects365 esperado, ainda a conferir); as mascaras saem na resolucao do
  quadro. Antes, um peso de segmentacao era recusado.

### Novo: Distancia pela mascara (`distance.region`)

- **`mask` (padrao):** com mascara, as estatisticas usam so os pixels do
  objeto (mascara levada ao mapa de profundidade e erodida para descartar a
  borda); sem mascara, a caixa, como antes.
- **`box`:** sempre a caixa (as mascaras so sao desenhadas).
- **`both`:** como `mask`, e grava tambem a distancia pela caixa para
  comparar.
- **`distance.mask_erode`** (padrao 1): erosao em pixels do mapa de
  profundidade; recua sozinha em objetos finos demais.

### Novo: Mascaras no video e nas saidas

- Mascara translucida na cor da instancia, no quadro e no painel de
  profundidade (`render.mask_alpha`, padrao 0.35; 0 desliga). O painel de
  informacoes mostra `region <politica> . erode <k>` nos quadros em que
  alguma instancia usou a mascara.
- `detections.csv` ganha no fim `used_mask`, `mask_px`, `mask_erode` e
  `box_distance_m` (este so com `region: both`).
- `metrics.json` ganha `distance_regions` (quantas instancias por mascara e
  por caixa) e, com `both`, `region_comparison` (diferenca mascara - caixa
  por rotulo: mediana com sinal, mediana e p90 absolutas, mediana
  relativa).

### Novo: Exemplos de configuracao

- `configs/rfdetr_seg_da3.yaml`, `configs/yolo_seg_da3.yaml` e
  `configs/yolo_o365_seg_da3.yaml`, todos com `region: both` e mapa de
  profundidade lado a lado.

### Como usar

1. Baixar os pesos YOLO26-seg desejados em
   `~/dev/ultralytics/_testbed/weights/` (release `v8.4.0` de
   `ultralytics/assets`); os do RF-DETR Seg vem sozinhos.
2. `uv run --extra infer python -m det_depth_distance --config configs/rfdetr_seg_da3.yaml`.
3. Para comparar erosoes: `--set distance.mask_erode=0` (ou 2).

### Detalhes tecnicos

- Reamostragem da mascara (vizinho mais proximo no centro de cada pixel do
  mapa, escalas x/y separadas e deslocamento de crop) e erosao 3x3 em numpy
  puro (`distance.py`), sem OpenCV nem scipy; a estatistica e o mesmo nucleo
  da caixa (`_region_stats`).
- A janela da mascara e a caixa sem encolhimento; pixels da mascara fora
  da caixa sao ignorados.
- Mascara toda vazia equivale a "sem mascara": um detector com mascara e
  outro sem podem rodar juntos (`concat`), e a fusao ja escolhe a mascara do
  melhor membro que tem mascara.

### Observacoes importantes

- Objeto pequeno cuja mascara tem menos de `min_valid_px` pixels do mapa
  fica "n/a" pela mascara; com `both`, a distancia pela caixa fica no CSV.
- O vocabulario dos pesos `yolo26*-objv1-seg.pt` (esperado Objects365) e a
  validacao na máquina com GPU (testes `-m gpu`, regressao da task_0002 e as
  execucoes caixa x mascara) ficaram pendentes: o servidor estava fora do
  ar durante a implementacao.

### O que nao muda

- Detectores so de caixas produzem as mesmas deteccoes, distancias e
  painel do video de antes, com qualquer politica; as colunas novas do CSV
  ficam no fim (vazias, `used_mask = 0`).
- Mascaras nao sao gravadas em disco.
- Nenhuma dependencia nova.

---

## [2026-10-13] Harmonizacao de rotulos e fusao entre detectores

Com varios detectores no mesmo passo, o mesmo carro aparecia duas vezes e
com nomes diferentes conforme o vocabulario de cada modelo (`truck` no
COCO; `suv`, `van` ou `pickup truck` no Objects365). A task_0003 adiciona
mapas de rotulos por detector, que traduzem os nomes de cada modelo para
nomes canonicos comuns (ex.: tudo que e veiculo vira `vehicle`), e duas
estrategias de fusao (NMS e WBF) que produzem uma lista unica de objetos,
guardando quais detectores viram cada um e com quais scores. O
`metrics.json` passa a medir a concordancia entre detectores. O
comportamento padrao (`concat`) nao muda.

### Novo: Mapas de rotulos por detector

- **Opcao `labels`** em cada detector aponta um YAML de mapa
  (`canonical: {vehicle: [car, truck, bus], ...}`), com politica para
  rotulos fora do mapa (`unmapped: keep` ou `drop`).
- **Mapas versionados** em `configs/labels/`: `coco_driving.yaml` (RF-DETR,
  YOLO26 COCO) e `objects365_driving.yaml` (YOLO26 Objects365).
- **Conferidos na carga:** nome que o modelo nao conhece, ou mapa escrito
  para outro vocabulario, e erro antes do primeiro quadro.

### Novo: Fusao entre detectores (`aggregation: nms` / `wbf`)

- **`nms`:** supressao de nao-maximos sobre todas as caixas, entre
  detectores e tambem duplicatas de um mesmo detector, por rotulo canonico
  (ou sem olhar o rotulo, com `class_agnostic: true`).
- **`wbf`:** Weighted Boxes Fusion; cada grupo vira uma caixa media
  ponderada pelo score (implementacao propria, sem dependencia nova). O
  score fundido fica limitado a 1: com pesos desiguais e duplicatas de um
  mesmo detector, a formula de referencia passaria de 1.
- **Pesos por detector** (`aggregation.weights`) para favorecer um modelo;
  nao e calibracao de score.
- **Instancia fundida guarda seus membros:** detector, rotulo original e
  score original de cada deteccao que a compoe.

### Novo: Concordancia entre detectores

- Secao `agreement` no YAML (ligada por padrao; vale com 2 ou mais
  detectores, inclusive com `concat`).
- `metrics.json` ganha `agreement`: por par de detectores, quantos objetos
  casam por IoU, quantos sao exclusivos de cada um e a IoU media, no total
  e por quadro; por detector, quantas duplicatas internas foram removidas.

### Novo: Exemplo de configuracao

- `configs/multi_detector_fused.yaml`: RF-DETR large + YOLO26-L Objects365
  com mapas de rotulos, fusao `wbf` e concordancia.

### Como usar

1. `uv run --extra infer python -m det_depth_distance --config configs/multi_detector_fused.yaml`.
2. Mesmo YAML sem fusao: `--set 'aggregation={type: concat}'`; com NMS:
   `--set aggregation.type=nms`.

### Detalhes tecnicos

- `detections.csv` ganha no fim `n_sources`, `sources`, `source_scores` e
  `raw_labels` (um item por detector, separados por `|`); a coluna
  `class_name` passa a ser o rotulo canonico quando ha mapa.
- `metrics.json`: `detections_by_n_sources`, tempo `agree`,
  `pipeline.agreement` e, em cada detector, a descricao do mapa (`labels`);
  o agregador descreve os pesos efetivos.
- Video: com fusao, a cor e pelo rotulo canonico, o rotulo mostra o numero
  de detectores que viram o objeto (ex.: "vehicle 91% x2", com o sinal de
  multiplicacao) e o painel mostra o agregador e a concordancia.
- Geometria de caixas (IoU, NMS, WBF, casamento) em `core/boxes.py`, numpy
  puro, testada no ambiente leve; reutilizavel pelo rastreamento futuro.
- `--list-models` passa a listar `nms` e `wbf` com suas opcoes.

### Observacoes importantes

- Os scores de RF-DETR e YOLO26 Objects365 nao estao na mesma escala (nos
  mesmos veiculos, mediana de 0.90 contra 0.71); a fusao ponderada
  favorece o modelo mais confiante. Os pesos so reescalam.
- O YOLO26 Objects365 costuma emitir duas caixas de veiculo diferentes
  (ex.: `pickup truck` e `suv`) para o mesmo objeto; com o mapa para
  `vehicle`, a fusao as junta.
- A validacao de ponta a ponta na máquina com GPU (testes `-m gpu` dos mapas,
  regressao da task_0002 e execucoes concat/nms/wbf) ficou pendente: o
  servidor estava fora do ar durante a implementacao.

### O que nao muda

- `aggregation: concat` continua sendo o padrao; YAMLs sem `labels` e com
  um so detector produzem as mesmas deteccoes e distancias de antes (as
  colunas novas do CSV ficam no fim).
- Nenhuma dependencia nova.

---

## [2026-10-13] Pipeline modular com configuracao YAML e detector YOLO26

O pipeline deixa de ser amarrado ao RF-DETR e ao Depth Anything 3 (task_0002):
cada modelo vira um componente intercambiavel, varios detectores podem rodar
no mesmo quadro e a execucao inteira passa a ser descrita por um arquivo
YAML. Entra tambem o detector Ultralytics YOLO26, inclusive os pesos
treinados no Objects365 (365 classes, com `suv`, `van`, `pickup truck`,
`machinery vehicle`). Os resultados do pipeline antigo sao reproduzidos
exatamente: nos 60 quadros de referencia, as 28 colunas antigas do CSV saem
identicas byte a byte.

### Alteracao: Nova linha de comando (quebra de compatibilidade)

- **Uma configuracao por arquivo:** `det-depth-distance --config run.yaml
  [VIDEO] [--set chave=valor]...`; o video posicional substitui
  `source.video` e `--set` ajusta qualquer chave (ex.:
  `--set source.max_frames=300`, `--set detectors.0.threshold=0.3`).
- **As flags antigas foram removidas** (`--det-model`, `--depth-model`,
  `--focal-px`, `--hf-cache`, `--layout`, ...), sem aliases: cada uma tem
  uma chave equivalente no YAML (tabela no README).
- **`--list-models`** lista os detectores, modelos de profundidade e
  agregadores disponiveis com suas opcoes.
- **Erros de configuracao** (chave desconhecida, tipo errado, valor fora da
  faixa) aparecem com o caminho da chave e codigo de saida 2, antes de
  carregar qualquer modelo. Classe que o modelo nao conhece e peso YOLO
  inexistente sao detectados ao carregar o detector, com a mesma mensagem
  de erro e sem nenhum download.

### Novo: Detector YOLO26 e varios detectores no mesmo passo

- **YOLO26** com pesos COCO ou Objects365; preset de classes `driving`
  proprio de cada vocabulario (identificado pelas classes do modelo).
- **Varios detectores** (ex.: RF-DETR + YOLO26) no mesmo quadro: todas as
  caixas sao mantidas e marcadas com o modelo de origem; no video, a cor e
  a contagem do painel passam a ser por modelo.
- **Ultralytics isolado:** sem telemetria, sem instalacao automatica de
  pacotes e sem gravar configuracoes fora do projeto.

### Novo: Exemplos de configuracao

- `configs/rfdetr_da3.yaml` (comportamento da task_0001),
  `configs/yolo_o365_da3.yaml` e `configs/multi_detector.yaml`.

### Como usar

1. `uv sync --extra infer` (instala o ultralytics).
2. Rodar: `uv run --extra infer python -m det_depth_distance --config configs/multi_detector.yaml --set source.max_frames=300`.
3. Testes de GPU (contratos + regressao): `uv run --extra infer pytest -m gpu -v`.

### Detalhes tecnicos

- Novo layout: `core/` (contratos `Instances`/`DepthResult`, classes base
  `Detector`/`DepthEstimator`, registro de adaptadores, `Pipeline`),
  `adapters/` (`rfdetr`, `ultralytics_yolo`, `da3`), `io/` (video, fonte,
  sinks), `camera.py`, `config.py`, `runtime.py`; `detector.py`, `depth.py`
  e `video_io.py` foram removidos (conteudo movido).
- `detections.csv` mantem as colunas antigas na mesma ordem e ganha
  `source` e `raw_label` no fim.
- `metrics.json`: `detector`/`depth`/`args` substituidos por `components`
  (descricao de cada componente), `pipeline` e `config` (YAML resolvido);
  novos `detections_by_source` e tempos `det.<nome>` e `aggregate`.
- Testes de GPU em `tests/gpu/` (marcador `gpu`, fora do `uv run pytest`
  local): contrato comum dos adaptadores e regressao de 60 quadros contra a
  task_0001.

### Atualizacao: Dependencias

- `pyyaml>=6` no ambiente base; `ultralytics==8.4.175` no extra `infer`
  (+8 pacotes no lock, nenhuma versao existente alterada).

### O que nao muda

- O calculo da distancia (`distance.py`) e o mesmo da task_0001; a
  configuracao `configs/rfdetr_da3.yaml` reproduz a execucao anterior.
- As saidas continuam na mesma pasta e com os mesmos nomes de arquivo.

### Observacoes importantes

- O ultralytics e licenciado sob AGPL-3.0; aceitavel para este prototipo de
  pesquisa interno (dependencia opcional do extra `infer`).
- Os scores de RF-DETR e YOLO nao sao comparaveis: nesta versao nada e
  fundido entre detectores (a fusao fica para a task_0003).

---

## [2026-10-12] Pipeline RF-DETR + Depth Anything 3 com distancia por objeto

Primeira versao funcional do projeto (task_0001): um comando que processa um
video quadro a quadro com o detector RF-DETR e o modelo de profundidade
metrica Depth Anything 3 e estima, para cada objeto detectado, a distancia
ate a camera a partir dos pixels de profundidade dentro da caixa. O resultado
e um video anotado com caixas e distancias, uma planilha (CSV) com uma linha
por deteccao e um arquivo de metricas da execucao.

### Novo: Comando `det-depth-distance`

- **Uma unica execucao** roda os dois modelos sobre o mesmo quadro e gera o
  video anotado.
- **Detector configuravel:** variantes `nano`, `small`, `medium`, `large`
  (padrao) ou um checkpoint proprio; filtro de classes (`driving`, `all` ou
  lista) conferido contra as classes do modelo carregado.
- **Profundidade em metros:** o modelo `da3metric-large` (padrao) recebe a
  distancia focal da camera, informada pelo usuario ou estimada
  automaticamente; modelos que ja respondem em metros tambem sao aceitos, e
  modelos de profundidade relativa sao recusados com mensagem clara.

### Novo: Estimativa robusta de distancia por caixa

- **Somente o objeto conta:** a caixa e encolhida nas bordas, pixels de ceu
  e alem de um alcance maximo sao ignorados, e o fundo que sobra dentro da
  caixa e descartado como outlier.
- **Estatistica a escolher:** mediana (padrao), media, media aparada, moda
  (pico mais proximo) ou percentil; distancia ao longo do eixo da camera ou
  em linha reta.
- **Transparencia:** media, mediana, desvio-padrao, variancia e percentis de
  cada caixa ficam no CSV; caixas com poucos pixels validos aparecem como
  "n/a".

### Novo: Video anotado e metricas

- **Rotulo por objeto** com classe, confianca, distancia em metros e desvio-padrao, painel com
  tempos de cada etapa e, opcionalmente, o mapa de profundidade colorido ao
  lado do video, com legenda em metros e a regiao efetivamente amostrada.
- **Arquivo de metricas** com tempos, memoria de GPU, focal usada e sua
  origem, versoes e todos os parametros da execucao; mapas de profundidade
  opcionais para analise posterior.

### Melhoria: Validacao e robustez (revisao independente)

- **Parametros invalidos** (ex.: numero de quadros para a focal igual a zero,
  focal negativa, inicio alem do fim do video) sao recusados antes de
  carregar qualquer modelo, com mensagem curta em vez de erro interno.
- **Video ilegivel** agora gera erro explicito em vez de uma execucao vazia.
- **Modelo de profundidade em pasta local** e reconhecido pelo proprio
  arquivo de configuracao, estimando a focal como o alias equivalente.

### Como usar

1. Instalar a stack GPU: `uv sync --extra infer`.
2. Rodar: `uv run --extra infer python -m det_depth_distance VIDEO --layout side-by-side --show-stats`.

### Detalhes tecnicos

- Modulos em `src/det_depth_distance/`: `cli.py`, `detector.py`,
  `depth.py`, `distance.py` (numpy puro), `render.py`, `video_io.py`,
  `__main__.py`.
- Saidas por execucao: `video.mp4`, `detections.csv`, `metrics.json` e,
  com `--save-depth`, `depth.npz` (mapas `float16`, `frame_index`,
  `intrinsics` (N, 4) e `geometry` + `geometry_fields`).
- `metres = saida_DA3METRIC * (fx + fy) / 2 / 300`, com fx/fy na resolucao
  do mapa; tipo de modelo decidido pelo `model_name` do checkpoint
  (`peek_model_name` le o `config.json` de pastas locais).
- Decodificador ffmpeg encerrado com `kill()` ao parar cedo; falha do
  decodificador vira `RuntimeError`.
- Legenda de profundidade com passo de escala de fallback para `--max-depth`
  muito alto.
- 66 testes unitarios sem GPU (`uv run pytest`).

### Atualizacao: Dependencias

- `addict` adicionado ao extra `infer` (importado pelo depth-anything-3, mas
  nao declarado por ele).
- `[tool.ruff]` exclui `docs/` (o ruff 0.17 formata blocos de codigo em
  Markdown).

### Observacoes importantes

- A distancia depende diretamente da focal: em IMG_4021 e IMG_4022 a focal
  estimada foi 1390 px e 1503 px; para comparar videos da mesma camera,
  informar `--focal-px` fixo.

---

## [2026-10-12] Criacao do repositorio e adocao do padrao de projeto

Repositorio criado para combinar deteccao RF-DETR e profundidade metrica
Depth Anything 3 na estimativa de distancia por objeto em video. Esta
primeira entrada registra apenas o esqueleto e o padrao de projeto da casa;
o pipeline de inferencia chega na task_0001.

### Novo: Esqueleto do projeto Python com uv

- `pyproject.toml` com ambiente leve (numpy<2, OpenCV, Pillow) e extra
  `infer` com a stack GPU: torch 2.8.0 + torchvision 0.23.0 + xformers
  0.0.32.post2 do indice cu128, `rfdetr` (commit `1f2af2ea`) e
  `depth-anything-3` (commit `3d835ec`) fixados por git.
- `uv.lock` versionado; pacote `src/det_depth_distance/`.

### Novo: Padrao de projeto da casa

- `AGENTS.md`, `CLAUDE.md`, `README.md`, `DESCRIPTION.md`, `CHANGELOG.md`
  e `docs/task_TEMPLATE.md`, com ciclo de vida de tarefas, protecao do
  branch `main` e convencoes de commit.

---
