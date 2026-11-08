# Estendendo a interface web

A interface web é derivada dos registros do pipeline: o catálogo
(`GET /api/catalog`) é função de `core.registry` (adaptadores e estágios) e
dos metadados dos campos das dataclasses; os resultados de uma execução
(artefatos, métricas) são descobertos na pasta, não enumerados. Nada em `src/av_perception_lab/web/`
nomeia um modelo ou um estágio (o teste
`test_web_package_has_no_hardcoded_adapter_names` garante). Os testes de
`tests/web/test_extensibility.py` provam os itens abaixo.

## (a) Novo modelo (detector, profundidade, refinador, rastreador, ...)

1. Módulo novo com `@register("<família>", "<tipo>")` e classe com `Options`
   (dataclass `frozen, slots, kw_only`).
2. Uma entrada em `ADAPTER_MODULES` (`core/registry.py`).
3. Opcional: dicas de interface nos campos de `Options`:

   ```python
   zoom: float = field(default=1.5, metadata={"ui": {
       "label": "Zoom", "unit": "x", "min": 0, "max": 10, "step": 0.1,
       "advanced": True, "help": "..."}})
   ```

   Todas as chaves são opcionais; sem dica o campo usa o nome. O primeiro
   parágrafo da docstring da classe vira a descrição do módulo.

Nada mais: o modelo aparece em `catalog["modules"][família][tipo]`.
Um campo de arquivo/pasta anotado como `Path` (ou `Path | None`) entra
sozinho na política de caminhos do servidor (`web.security.path_problems`:
nunca na biblioteca nem em `out_root`; dentro de `--asset-root` quando
passado); um caminho guardado como `str` escapa dela, então use `Path`.

## (b) Novo estágio do pipeline

1. A execução do estágio fica em `Pipeline`/`cli.run` (trabalho de pipeline,
   não de interface) e o campo correspondente em `RunConfig`.
2. Registre o descritor, ao lado do campo (fim de `config.py`):
   `register_stage(Stage(yaml_key=..., title=..., order=..., kind=... ou
   config_cls=..., multiple=..., optional=..., general=..., group=...))`.
   `order` define a posição na pista; deixe folga entre os valores. `group`
   (opcional) junta estágios consecutivos com o mesmo valor numa só pista
   com esse título (ex.: `tracking` e `smoothing` em "Tracking & filter").

Nada sob `web/`: o estágio aparece em `catalog["stages"]` (e a seção em
`catalog["sections"]`, se tiver `config_cls`) na ordem registrada.

## (c) Visão ou inspetor personalizado

Estágio desconhecido vira uma pista genérica (cartão por especificação ou
cartão da seção); tipo de campo desconhecido, `map` e `any` viram um editor
YAML bruto. Uma visão específica entra pelo registro de extensões do
front-end (`web/src/editor/registry.ts`), sem tocar no backend:

```ts
import { registerFieldControl, registerStageView } from "./editor/registry";

// controle próprio para campos que a dica marca (metadata ui {"widget": "knob"})
registerFieldControl((f) => f.hint?.widget === "knob", KnobControl);
// pista própria para um estágio
registerStageView("planners", PlannerLane);
```

O registro mais recente vence; ambas as funções devolvem a função que
desfaz o registro. Sem registro, valem os padrões genéricos.

Uma pista que depende do estado do app (não só da configuração) lê o
`StageContext` do registro: o app o preenche (hoje com `{input}`, a entrada
atual do deck, vídeo ou MCAP) e o editor só o repassa, sem olhar dentro. A
opção `hidden(value, context)` de `registerStageView` tira o estágio do
canvas (a pista e a seta) quando devolve `true`. É assim que a pista
Rectification (`web/src/video/RectifyLane.tsx`) some com uma fonte de vídeo:

```ts
import { useContext } from "react";
import { registerStageView, StageContext } from "./editor/registry";

function RectifyLane(props: StageViewProps) {
  const input = useContext(StageContext).input as VideoRecord | null;
  // ...
}

// oferecida só para MCAP; continua visível se o YAML já a ligou
registerStageView("rectify", RectifyLane, {
  hidden: (value, context) => !isMcap(context.input as VideoRecord | null) && !(isRecord(value) && value.enabled === true),
});
```

O app fornece o contexto em volta do editor
(`<StageContext.Provider value={{ input, config }}>`, ver `App.tsx`); sem
provedor o contexto é `{}`.

The context also carries the current pipeline config under
`CONFIG_CONTEXT_KEY` (`"config"`, exported by `registry.ts`). It is meant for
field controls that depend on other sections: `FieldControlProps` gives a
control only its own field, value and path. The `render.semantic_classes`
dropdown (`web/src/editor/ClassSelect.tsx`) reads the `semantic` specs from
it to ask `POST /api/classes` for each map's classes. The control reads the
context rather than `useEditor()` so it keeps working (with no maps) outside
an `EditorProvider` and in tests with a plain provider; `useEditor()` stays
the way to change the config. Since the config changes on every edit, a
control that derives requests from it should key them on the part it reads
(the classes dropdown caches by the serialized spec and debounces new specs).

```ts
import { CONFIG_CONTEXT_KEY, registerFieldControl, StageContext } from "./editor/registry";

function ClassSelect(props: FieldControlProps) {
  const config = useContext(StageContext)[CONFIG_CONTEXT_KEY]; // ConfigDict | undefined
  // ...
}
registerFieldControl((f) => f.name === "semantic_classes", ClassSelect);
```

The catalog types `semantic_classes` as `any` (variants `str` and
`list<str>`), so the control matches the field by name, not by type.

## (d) Novo artefato ou nova seção de métricas

A vista de execuções (`#/runs/<id>`) não enumera arquivos: `GET
/api/runs/{id}/artifacts` lista toda a pasta da execução (nome, tamanho,
tipo de mídia, `role` quando é um arquivo conhecido de
`web.history.RUN_FILES`) e `GET /api/runs/{id}/files/{caminho}` serve
qualquer um deles. Um sink novo que grave `minha_saida.parquet` aparece na
lista com Abrir/Baixar sem nenhuma mudança; qualquer `.json` (o
`metrics.json` inclusive) vira uma árvore chave/valor recolhível, de modo que
uma seção nova de métricas aparece sozinha. Para uma visão própria, um
registro no front-end (`web/src/runs/artifactViews.tsx`):

```ts
import { registerArtifactView } from "./runs/artifactViews";

registerArtifactView((a) => a.name.endsWith(".parquet"), ParquetTable);
```

Arquivos de tipo ativo (HTML, SVG, JS, XML…) são servidos como download
(`Content-Disposition: attachment`, `nosniff`, `CSP: sandbox`): uma visão
própria de um tipo novo deve buscar o conteúdo (`fetch`) em vez de embutir a
URL num `iframe`. Só vídeo, PNG/JPEG/WebP/GIF, CSV, texto, JSON, NDJSON e
YAML abrem no navegador (`web.security.INLINE_MEDIA`).

O predicado recebe `{name, size, media_type, role, url}`; o registro mais
recente vence os embutidos (árvore JSON, imagem, MP4) e a função devolvida o
desfaz. Os indicadores do resumo e da comparação vêm de
`web.compare.SUMMARY_METRICS` (rótulo, unidade e se é melhor `lower` ou
`higher`): um indicador novo é uma entrada ali e um valor em `run_summary`;
o front-end renderiza a lista como vier (cartão, delta com sinal e cor de
melhora/piora).

## (e) Requirements of a new model (weights, code, packages)

A model's readiness is also derived from the registry, so a new model needs
no edit under `web/` or `src/av_perception_lab/web/` to get status, card
states and Prepare:

1. Override the classmethod `requirements(options) -> Requirements`
   (`core/requirements.py`) on the adapter: `assets` (registered kinds:
   `HuggingFaceAsset`, `UrlAsset` with `size_bytes` or `sha256`,
   `UserProvided`), `code` (`CodeRepo` pinned to a 7–64 hex commit),
   `packages` (import names) and `extra` (the `pyproject.toml` extra that
   provides them). It reads only the validated options, never `load()`, and
   imports no framework. `known` is `False` by default (on `CodeRepo` too);
   set `known=True` only when the source comes from a table in the adapter's
   code **and** the target is its canonical location under the weights root:
   the web server prepares only those (any other source reads "unknown
   source: prepare it from the CLI (--prepare)"). A file only the user can
   supply is `UserProvided`: the web shows its hint (`reason:
   "user_provided"`) and never offers Prepare for it.
2. In `_load`, check the files before importing the framework and raise
   `adapters._weights.not_prepared(...)` (a `ConfigError` naming `--prepare`)
   when a default file is missing: nothing is downloaded at run time.

That is all: `POST /api/models/status` lists the model with its layers,
source (each asset kind's `describe_source()`), size, licence and target;
the card turns yellow dashed with the alert icon until it is ready;
**Prepare model** / **Prepare all** obtain it through the registered sources;
a per-model extra gets **Install packages** after a clean dry-run (or the
state `conflict`); runs are refused with 409 while it is not ready. A new
source kind is one `Asset` subclass (`register_asset_kind`) plus one source
class (`core.provision.register_source`), again without web edits. Worked
example (a hypothetical Metric3D depth model), locations, states and the
dependency-conflict paths: `docs/model_environments.md`. Tests:
`tests/web/test_models.py` (status rows of fake adapters registered with
`monkeypatch.setitem(registry._REGISTRY[...])`).

## (f) Declaring the classes of a new model

A semantic segmenter or detector can tell the editor which classes it emits
before it is loaded, so the `render.semantic_classes` dropdown can list them
(`POST /api/classes`, `web/classes.py`). Override the classmethod
`Component.declared_classes(options) -> ClassCatalog | None`
(`core/components.py`):

- Declare the classes when they are cheaply knowable from the options or the
  weights without importing a framework: a constant of the architecture
  (`ClassCatalog(names, "static")`, as TwinLiteNet+), or plain metadata next to
  the weights such as a `config.json` (`"metadata"`, as SegFormer). It must
  never raise and never load the model.
- Otherwise return `None` (the default). Detectors get a fallback: a
  `Detector` that implements `weights_file(options)` (the checkpoint path,
  importable without frameworks) has its names recorded after every load in
  `<weights root>/.classes/<key>.json` and read back as `"cache"`.
- Contract: the declared names must equal `class_names`/the segmenter's
  classes after `load()`. If the architecture can vary (checkpoint with another
  head), check it at load and fail loudly (see `_check_heads` in
  `adapters/semantic/twinlitenet.py`). Add a test that compares the two.
- The endpoint parses the spec into the options dataclass and checks its
  paths like a run, but does not construct the component; a local model
  directory is confined to the weights root and the `--asset-root` folders.

## Versão do catálogo

`catalog_version` sobe só em mudança incompatível; adicionar campos ou
estágios não é incompatível.
