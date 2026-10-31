# Estendendo a interface web

A interface web é derivada dos registros do pipeline: o catálogo
(`GET /api/catalog`) é função de `core.registry` (adaptadores e estágios) e
dos metadados dos campos das dataclasses. Nada em `src/det_depth_distance/web/`
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

## Versão do catálogo

`catalog_version` sobe só em mudança incompatível; adicionar campos ou
estágios não é incompatível.
