# Render class selection — design

Status: draft for approval. Nothing is built until this is approved.

## Intent

Turning on SegFormer paints all 19 Cityscapes classes over the video at
`render.semantic_alpha` (0.4), which buries the frame. Today the only render
controls are the global alpha and `render.semantic_overlay`, which picks whole
maps, not classes. The user wants to choose which classes are drawn, from a
list the interface already knows, with every class selected by default.

## Goals

- `render.semantic_classes`: choose which semantic classes are drawn, per map.
- Adapters declare their class list **before** loading, as a function of their
  options (weights), so the web editor can show it.
- The web editor shows a checkbox dropdown per semantic stage, all checked by
  default.
- Existing YAMLs and runs render exactly as before.

## Non-goals (now)

- Detector render filtering (see "Calls I made"). The API is designed so it can
  follow without rework.
- Presets in the dropdown; plain class names only.
- MCAP output (a separate feature; not implemented today: sinks are Csv, Video,
  Trajectory, Npz, Metrics and `io/mcap_reader.py` is read-only).
- Distance exclusion (`distance.exclude_semantic`) in the UI.
- Re-rendering a finished run without re-inference.

## Anti-goals

- Hiding a class in the render must never change distance, tracking, CSV, NPZ
  or metrics. Selection is presentation only.
- No model loading, and no unpickling of `.pt`/`.pth`, inside a web request.
- A wrong class list must fail loudly, not draw the wrong thing silently.

## Constraints (verified in the code)

- Class names exist only after `load()` today (`SemanticSegmenter.class_names`,
  `Detector.class_names`).
- The weights decide the classes: YOLO `weights`, RF-DETR `weights`, SegFormer
  `model` (any HF repo), TwinLiteNet+ `weights` (its list is a constant).
- `Component.requirements(cls, options)` is the precedent: a classmethod of
  validated options, callable without loading.
- The web catalog (`web/schema.py: module_catalog`) is a pure function of the
  registries and cannot carry option-dependent lists.
- Semantic class names collide across maps and detectors (`person`, `car`,
  `traffic light`); the existing key is `<map>.<class>`
  (`config.SEMANTIC_NAME_PATTERN`, `exclude_semantic`).
- Class names are validated late: prefix in `load_config`, class in
  `Pipeline.load`. Follow the same split.
- `Renderer._semantic_lut` is cached by `class_names`; the visible set must be
  part of the cache key.
- The weights root resolves from `set_weights_root`, `AVPL_WEIGHTS_ROOT`, or
  the repo `weights/` (gitignored).

## Approaches considered

1. **A. YAML option only, no UI.** Smallest; fixes the clutter; leaves class
   names to be typed by hand.
2. **B. A plus class discovery, a web endpoint and a dropdown. (Chosen.)**
   Delivered in this order: A, then discovery, then web.
3. **C. No code** (lower alpha, `semantic_overlay`). Cannot select individual
   classes.

For discovery: a constant per class (rejected: weights change the classes), an
abstract method (rejected: forces adapters without a cheap source to invent
one), a classmethod with a `None` default plus a load-time cache and a contract
test (chosen).

## Design

### Config (`config.py`)

`RenderConfig.semantic_classes: tuple[str, ...] | None = None`, entries
`<map>.<class>`. `None` = every class of every drawn map. `semantic_overlay`
keeps selecting maps; a map not in the overlay draws nothing regardless.
`RunConfig` checks the prefix against the segmenter names (like
`exclude_semantic`); `Pipeline.load`-time code checks the class against the
loaded `class_names` and raises `ConfigError` with the valid names.
A map with no entry in a non-`None` list draws no classes (allowlist), which is
stated in the field comment and the reference.

### Renderer (`render.py`, `io/sinks.py`)

`Renderer.render` receives the visible classes per map; hidden classes get
alpha 0 in the lookup table, the same mechanism as `SEMANTIC_BACKGROUND`. The
table cache key becomes `(class_names, visible)`. `VideoSink` passes the
resolved set. Nothing else reads it: CSV, NPZ, metrics and distance are
unchanged. `metrics.json["config"]` records the selection like any other field.

### Class declaration (`core/components.py`)

```python
@dataclass(frozen=True, slots=True)
class ClassCatalog:
    names: tuple[str, ...]
    source: Literal["static", "metadata", "cache"]

class Component:
    @classmethod
    def declared_classes(cls, options) -> ClassCatalog | None:
        return None   # unknown until loaded
```

Adapters:
- `TwinLiteNet`: returns its constant (`static`); `_load` additionally checks
  the head's output channels against it and raises `ContractError` on mismatch.
- `SegFormer`: reads `id2label` from `config.json` in the HF cache
  (`metadata`), reusing `_id2label_names`; returns `None` if the file is
  absent.
- Others (`YOLO`, `RF-DETR`): read the cache, below; the detector API exists
  now but no detector UI is built in this round.

Contract test: for every registered adapter, with test fakes, a non-`None`
`declared_classes` must equal `class_names` after `load()`.

### Load-time cache

After a successful `load()`, the base writes
`<weights root>/.classes/<key>.json` (`names`, `weights_id`). `key` derives from
the resolved weights path plus size and mtime (sha256 where the asset declares
one), so a replaced file is a cache miss. `declared_classes` falls back to it
(`source: cache`). A write failure (read-only root) is logged and ignored. The
cache is only a hint: `class_names` after load stays authoritative.

### Web

- `POST /api/classes` takes a stage spec (`kind`, `type`, options), builds the
  options dataclass, calls `declared_classes`, and returns
  `{names, source}` or `{names: null}`. It never loads a model. The path is
  validated with the same checks as the other endpoints.
- Editor: a checkbox dropdown on each semantic stage and in the Render panel,
  all checked by default, with select all / none and a search box. While
  `names` is null it shows "classes appear after the first load". The control
  writes `render.semantic_classes` only when the selection is not "all", so
  untouched configs stay unchanged.
- The `background` class is listed but unchecked-and-disabled with a note,
  matching the renderer's rule.

### Docs

`DESCRIPTION.md` (endpoint, options), `docs/reference.md` (YAML field),
`CHANGELOG.md`, `docs/web_extending.md` if the editor extension point changes,
and `docs/task_0035.md` created before the code.

### Tests

Config parse and validation (unknown prefix, unknown class, allowlist
semantics); renderer (hidden class alpha 0, cache key); `declared_classes`
for the two segmenters and the base default; cache write, hit, invalidation and
read-only root; contract test; endpoint (spec in, names out, no model loaded);
frontend component test (default all checked, search, none selected).

## Left to the builder

Cache key details and file layout; exact UI placement; the wording of the
"not yet known" message; naming of `ClassCatalog` fields.

## Calls I made (to check while reading)

1. **Semantic only this round.** The user asked for detectors too, but the
   clutter is a SegFormer problem and a detector render filter has an open
   label-space question (raw names vs. canonical labels after a `LabelMap`,
   and what happens to a hidden box's trajectory, BEV entry and HUD count).
   Say if you want detectors included.
2. **Allowlist, not denylist**, with `None` meaning all. It follows
   `semantic_overlay`; the cost is that adding a model to a config with a
   non-`None` list hides its classes until chosen.
3. **No model loading in the web endpoint**, to avoid unpickling arbitrary
   checkpoints on each editor change.
4. **Cache lives under the weights root**, so it follows the weights and
   survives restarts, but does not travel with git.
5. **`background` is not selectable.** It is already never drawn.

## Follow-ups (not in this feature)

Detector render filter on the same API; MCAP output sink (recording every
class by default; selection stays presentation-only); `exclude_semantic`
column next to "show" in the same dropdown; per-class colours.
