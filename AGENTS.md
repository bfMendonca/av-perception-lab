# AGENTS.md — det-depth-distance

Guia autoritativo para desenvolvedores e agentes de IA que trabalham neste
repositório. Protótipo de pesquisa em Python que executa um ou mais
detectores (RF-DETR, YOLO26) e a profundidade métrica Depth Anything 3 sobre
os mesmos quadros de um vídeo e estima, para cada caixa detectada, a distância do objeto à câmera a
partir de estatísticas robustas dos pixels de profundidade dentro da caixa,
gerando um vídeo anotado. A inferência roda no servidor GPU `gpu-host`
(sincronizado por rsync) e, com o exemplo leve `configs/yolo_light_da3.yaml`,
também na GPU de 4 GB do notebook, num ambiente isolado; o desenvolvimento e os testes unitários rodam
localmente.
Detalhes de arquitetura, build e uso estão no `README.md`.

---

## Obrigatório: Manter DESCRIPTION.md Atualizado

O arquivo `DESCRIPTION.md` é uma descrição completa do projeto mantida para
consumo por LLMs externos. Ele **DEVE ser atualizado no mesmo commit** sempre
que qualquer mudança alterar:

- Modelos / esquema de dados
- Endpoints / rotas de API
- Dependências (adicionadas, removidas ou com major bump)
- Middleware, autenticação ou autorização
- Seções principais da UI ou fluxos de usuário
- Integrações externas ou variáveis de ambiente

Mudanças que não tocam nada disso (refactors internos, correções pontuais,
documentação) não exigem atualização.

---

## Comandos de Build & Execução

```bash
# Local: ambiente leve (numpy, OpenCV, Pillow + pytest/ruff)
uv sync

# Servidor GPU: envia o código e instala a stack de inferência (torch cu128, rfdetr, depth-anything-3, ultralytics)
utils/sync_gpu_host.sh push
ssh gpu-host 'cd ~/dev/det-depth-distance && ~/.local/bin/uv sync --extra infer'

# Executa o pipeline no servidor (`uv sync` sem --extra infer removeria a stack GPU)
ssh gpu-host 'cd ~/dev/det-depth-distance && ~/.local/bin/uv run --extra infer python -m det_depth_distance --config configs/rfdetr_da3.yaml _testbed/data/VIDEO.MOV'
# Ajustes rápidos sem editar o YAML; lista de adaptadores e opções
#   ... --config configs/multi_detector.yaml --set source.max_frames=300 --set render.show_stats=true
#   ... --list-models

# Notebook (GPU de 4 GB): ambiente isolado; nunca --extra infer no .venv (pesos e medição no README,
# "Execução no notebook")
#   export UV_PROJECT_ENVIRONMENT=_testbed/venv-infer && uv sync --extra infer
#   (com a variável exportada, todo uv run leva --extra infer; unset UV_PROJECT_ENVIRONMENT para voltar ao .venv)
#   uv run --extra infer python -m det_depth_distance --config configs/yolo_light_da3.yaml _testbed/data/VIDEO.MOV

# Pesos dos refinadores SAM e do SegFormer: só com confirmação do usuário (~3,9 GB); vão para o
# cache HF weights/hf (ignorado pelo git e pelo push); a execução nunca baixa nada.
# No gpu-host, dentro do repo (lista completa e opcionais no README, "Pesos dos refinadores e do SegFormer"):
#   ~/.local/bin/uv run --extra infer python -c "from huggingface_hub import snapshot_download as d; d('facebook/sam2.1-hiera-large', cache_dir='weights/hf', allow_patterns=['*.json', 'model.safetensors'])"

# Traz as saídas de volta para o _testbed/ local
utils/sync_gpu_host.sh pull [RUN_DIR]

# Calibração e avaliação (ambiente leve, local ou no gpu-host; protocolo em
# docs/refs/protocolo_calibracao_e_referencias.md)
uv run det-depth-distance-calibrate board --out _testbed/board.png              # alvo para impressão
uv run det-depth-distance-calibrate inspect _testbed/data/VIDEO.MOV             # lente, 35 mm-eq.
uv run det-depth-distance-calibrate calibrate _testbed/data/CALIB.MOV --square-mm 25 \
    --out configs/calibration/<aparelho>_<lente>_<modo>.yaml
uv run det-depth-distance-evaluate --refs REFS.yaml _testbed/outputs/RUN [RUN ...] [--out DIR]
uv sync --extra eval   # laptop, só para --plot (matplotlib); `uv sync` volta ao ambiente leve
# No gpu-host NUNCA `--extra eval` sozinho (removeria a stack GPU): o --extra infer já traz o matplotlib
#   ~/.local/bin/uv run --extra infer det-depth-distance-evaluate ... --plot
```

## Comandos de Teste

```bash
uv run pytest                      # local: unitários + smoke, sem GPU (pula -m gpu; ~18 s)
uv run pytest -m smoke             # só o smoke ponta a ponta (tests/smoke/, ~8 s)
uv run pytest -m "not gpu and not smoke"   # ciclo rápido: só os unitários (~10 s; um -m substitui o do addopts)
# gpu-host: contratos dos adaptadores + regressão + smoke gpu (precisa de --extra infer e pesos)
ssh gpu-host 'cd ~/dev/det-depth-distance && ~/.local/bin/uv run --extra infer pytest -m gpu -v'
# gpu-host: todas as pendências de GPU das tasks 0002-0008, em ordem, com logs
utils/gpu_pending.sh list                           # passos (local ou no gpu-host)
utils/gpu_pending.sh run --dry-run                  # só imprime os comandos
ssh -t gpu-host 'cd ~/dev/det-depth-distance && utils/gpu_pending.sh run'   # em tmux
```

Marcador `gpu` (`[tool.pytest.ini_options]`: `addopts = "-m 'not gpu'"`):
todo teste sob `tests/gpu/` é marcado automaticamente pelo
`tests/gpu/conftest.py`, então o `uv run pytest` local nunca roda testes de
GPU; `-m gpu` (a última opção `-m` vence) os seleciona no servidor. A
regressão compara 60 quadros de IMG_4021 com a referência da `task_0001`
congelada em `_testbed/regression/` (ou `$DDD_REGRESSION_REF`).

Marcador `smoke`: todo teste sob `tests/smoke/` é marcado pelo
`tests/smoke/conftest.py` e roda por padrão (o smoke cabe no limite de 10 s
da `task_0008`; se passar dele, vai para `addopts` e passa a ser obrigatório
rodar `-m smoke` antes de cada commit). O `gpu_pending.sh` nunca baixa pesos
sem `--allow-downloads` **e** confirmação por item; os downloads da
`task_0004` (fora do repo) só são impressos.

## Lint & Formatação

```bash
uv run ruff check --fix .
uv run ruff format .
```

---

## Convenções de Código

- **Formatação**: Ruff com defaults (88 colunas, aspas duplas), sem seção de
  configuração no `pyproject.toml` enquanto os defaults bastarem.
- **Imports**: 3 grupos PEP 8 (stdlib, terceiros, locais), ordenados.
  Imports pesados (`torch`, `rfdetr`, `depth_anything_3`, `ultralytics`,
  `transformers`, `timm`, o `model` do clone TwinLiteNet+)
  ficam dentro do `_load` dos adaptadores e dos métodos chamados depois
  dele (e de `runtime.DeviceMonitor`),
  para que `core/`, a configuração e o cálculo de distância sejam
  importáveis e testáveis sem a stack GPU.
- **Nomenclatura**:

  | Elemento | Padrão | Exemplo |
  |---|---|---|
  | Módulos / funções | `snake_case` | `distance.py`, `box_depth_stats()` |
  | Classes | `PascalCase` | `DepthEstimator` |
  | Constantes | `UPPER_SNAKE` | `DEPTH_MODELS` |
  | Helpers privados | prefixo `_` | `_scale_box()` |
  | Booleanos | `is_*` / `has_*` | `is_metric` |
  | Argumentos de CLI | `--kebab-case` | `--list-models` |
  | Chaves do YAML | `snake_case` | `box_shrink` |

- **Docstrings/comentários**: em inglês. Todo módulo tem docstring com o
  propósito e o contrato; comentários explicam o "porquê" (ex.: convenções
  de escala do modelo métrico), não o "quê".
- **Organização**: código em `src/det_depth_distance/`: `core/` (contratos,
  ABCs, registro, rótulos e mapas de rótulos, geometria de caixas,
  agregação/fusão, concordância, rastreamento, pipeline — sem frameworks),
  `adapters/` (um módulo por framework, registrado com `@register` e listado
  em `ADAPTER_MODULES`), `io/` (vídeo, fontes, sinks, metadados QuickTime),
  `evaluation/` (avaliação contra referências de campo, sem I/O no núcleo),
  `tools/` (CLIs auxiliares `det-depth-distance-calibrate` e
  `-evaluate`, finas), e módulos de topo `camera.py`, `calibration.py`,
  `config.py`, `distance.py`, `temporal.py`, `render.py`, `runtime.py`,
  `cli.py`. Configurações de exemplo em `configs/` (mapas de rótulos
  versionados em `configs/labels/`, modelo de referências em
  `configs/references/`, calibrações em `configs/calibration/`). Scripts
  utilitários bash em `utils/`.
- **Testes**: pytest em `tests/`, espelhando os módulos
  (`distance.py` → `tests/test_distance.py`). Testes unitários não dependem
  de GPU nem de pesos de modelos; `Instances` de teste são criadas pela
  fixture-fábrica `make_instances` (`tests/conftest.py`); calibração e
  avaliação usam as fixtures `synthetic_camera`, `render_board` (tabuleiro
  renderizado com distorção conhecida) e `fake_run_dir` (fábrica de pasta de
  execução fictícia), `distort_image` (imagem ideal vista pela lente
  distorcida) e `board_images` (pasta de tabuleiros renderizados). Testes que precisam
  de GPU/pesos ficam em `tests/gpu/` (marcador `gpu`); o smoke ponta a ponta
  fica em `tests/smoke/` (marcador `smoke`). Dublês de adaptadores entram no
  registro **só** por `monkeypatch.setitem(registry._REGISTRY[kind], tipo,
  cls)` depois de `registry.load_adapters()` (nunca `@register`, que muda o
  processo para sempre); toda pasta de execução é conferida por
  `tests/run_outputs.check_run_outputs`; cada execução da CLI num teste usa
  um `output.out_root` próprio (o nome da pasta tem resolução de segundos).
  Execuções reais da CLI nos testes `gpu` vão por subprocesso (fixture
  `run_cli_subprocess`).

---

## Ambiente Python (uv)

Ambientes virtuais e dependências são geridos **exclusivamente** com o
[Astral uv](https://docs.astral.sh/uv/):

- `pyproject.toml` na raiz do projeto é o manifesto único
  (metadados, dependências e config de tooling); `uv.lock` é versionado e
  `.venv/` fica no `.gitignore`.
- Comandos padrão:

```bash
uv sync                          # criar/sincronizar o ambiente
uv add <pkg> / uv remove <pkg>   # gerir dependências
uv run <comando>                 # executar no ambiente (ex.: uv run pytest)
```

- **NUNCA** `pip install` manual nem `requirements.txt` novo; toda mudança
  de dependência passa por `uv add`/`uv remove` (que atualizam o lock).
- Dependências da stack GPU vão no extra `infer`
  (`uv add --optional infer <pkg>`); `torch`, `torchvision` e `xformers` vêm
  do índice explícito `pytorch-cu128`. `rfdetr` e `depth-anything-3` são
  fixados por commit git em `[tool.uv.sources]` — atualizar o commit é uma
  mudança de dependência (exige `DESCRIPTION.md` atualizado).
- `numpy<2` é exigido pelo `depth-anything-3`; não remover a restrição.

---

## Documentação de Tarefas (`docs/`)

Toda melhoria ou correção planejada é documentada em arquivos
`docs/task_DDDD.md`. Cada arquivo contém o escopo completo, a análise de
impacto por arquivo, o plano de implementação passo a passo e os
procedimentos de verificação de uma única tarefa autocontida. Esses arquivos
são feitos para serem consumidos por qualquer desenvolvedor ou agente de IA
que execute a tarefa de forma independente.

**Formato do arquivo:**

```
docs/task_0001.md   # Primeira task (maior prioridade)
docs/task_0002.md   # Segunda task
...
```

**Cada arquivo contém:**

- **Descrição** — escopo breve, porém completo.
- **Escopo** — todos os arquivos alterados/criados/removidos, com detalhe.
- **Planejamento** — plano de implementação passo a passo.
- **Análise de Impacto** — matriz de risco e possíveis bugs.
- **Verificação** — procedimentos de teste automatizado e manual.
- **Dependências** — tarefas que precisam ser executadas antes desta.
- **Detalhes de Implementação** — (após a conclusão) o que foi realmente
  feito, incluindo desvios do plano original.
- **Caderno de Testes** — (após a conclusão) checklist manual para o usuário
  validar que as mudanças não introduziram regressões.

**Ordem de execução:** as tasks são numeradas na ordem recomendada de
execução. Números menores geralmente são executados primeiro. Dependências
entre tasks são documentadas explicitamente na seção "Dependências" de cada
arquivo. Use `docs/task_TEMPLATE.md` como ponto de partida para novas tasks.

---

## Obrigatório: Regras de Ciclo de Vida de Tarefas

As regras a seguir são **obrigatórias** para todo desenvolvedor ou agente que
trabalhe neste projeto.

### 1. Toda implementação DEVE ter um arquivo de tarefa

Antes de escrever qualquer código, é **obrigatório** verificar se existe um
arquivo `docs/task_DDDD.md` correspondente ao trabalho. Se não existir, um
novo arquivo **DEVE** ser criado antes de iniciar a implementação, seguindo o
formato padrão (Descrição, Escopo, Planejamento, Análise de Impacto,
Verificação, Dependências). Use o próximo número disponível na sequência.

### 2. Tarefas concluídas DEVEM ser marcadas com `(CONCLUÍDA)`

Ao finalizar uma tarefa, é **obrigatório** anexar `(CONCLUÍDA)` à linha de
título `#`. Exemplo:

```markdown
# Task 0001: Título Descritivo da Tarefa (CONCLUÍDA)
```

### 3. Tarefas concluídas DEVEM incluir "Detalhes de Implementação"

Após finalizar, adicionar ao fim do arquivo uma seção
`## Detalhes de Implementação` contendo:

- **O que foi feito** — resumo objetivo da implementação real.
- **Desvios do plano original** — qualquer alteração em relação ao
  Planejamento. Se não houve desvios, declarar isso explicitamente.
- **Observações** — notas, avisos ou itens de acompanhamento relevantes.

### 4. Tarefas concluídas DEVEM incluir "Caderno de Testes"

Após finalizar, adicionar uma seção `## Caderno de Testes` logo após
`## Detalhes de Implementação`. É um **checklist manual** que o usuário segue
para validar que as mudanças não introduziram regressões.

**Diretrizes:**

- Cada item é um passo concreto e acionável no app em execução (navegador,
  terminal ou cliente de API).
- Cobrir **todas as áreas funcionais afetadas** — não só o caminho feliz.
- Incluir **testes negativos** quando relevante.
- Itens curtos e objetivos, verbo no imperativo (Abrir, Clicar, Verificar).
- Usar checkbox Markdown (`- [ ]`).
- Escrever em **português (pt-BR)**.
- Não duplicar o que já é coberto por testes automatizados.

### 5. Atualizações do arquivo de tarefa fazem parte do commit da implementação

O marcador `(CONCLUÍDA)`, `Detalhes de Implementação` e `Caderno de Testes`
devem estar no **mesmo commit/PR** que entrega a implementação — nunca em
commit separado.

### 6. NUNCA commitar diretamente no branch de produção — sempre criar um branch

O branch `main` é o branch de produção. **NUNCA** commitar
diretamente nele. Antes de qualquer implementação:

1. Criar um novo branch a partir de `main` seguindo a convenção.
2. Realizar todo o trabalho nesse branch.
3. Integrar a `main` somente via pull request.

**Convenção de nomes de branch:**

| Tipo | Padrão | Exemplo |
|------|--------|---------|
| Nova funcionalidade | `feat/<descrição>` | `feat/graceful_shutdown` |
| Correção de bug | `fix/<descrição>` | `fix/finalize_timeout` |
| Correção (alt.) | `bug/<descrição>` ou `bugfix/<descrição>` | `bugfix/h4_display` |

- Usar **snake_case** na descrição; nomes curtos e descritivos.
- Derivar o nome do branch do assunto da task
  (ex.: task "Desligamento gracioso" → `feat/graceful_shutdown`).

---

## Convenções de Commit

Commits seguem **Conventional Commits** com assunto em **pt-BR**, no formato:

```
<type>(<scope>): <assunto em pt-BR, imperativo, ≤50 chars, sem ponto final>

<corpo opcional em pt-BR explicando o porquê, wrap em 72 chars>
```

- **Tipos** (inglês, minúsculo): `feat`, `fix`, `perf`, `refactor`, `docs`,
  `test`, `chore`.
- **Escopos válidos** (inglês, minúsculo, um por commit):

  | Escopo | Cobre |
  |--------|-------|
  | `core` | `src/det_depth_distance/core/` (contratos, ABCs, registro, rótulos, agregação, rastreamento, pipeline) |
  | `adapters` | `src/det_depth_distance/adapters/` (invólucros de framework) |
  | `io` | `src/det_depth_distance/io/` (vídeo, fontes, sinks) |
  | `distance` | `src/det_depth_distance/distance.py` (núcleo numpy de distância) e `temporal.py` (suavização temporal por trilha) |
  | `camera` | `src/det_depth_distance/camera.py` (intrínsecos, calibração, correção de distorção) e `calibration.py` (ferramenta de calibração) |
  | `render` | `src/det_depth_distance/render.py` |
  | `cli` | `src/det_depth_distance/cli.py`, `__main__.py`, `__init__.py`, `config.py`, `runtime.py` (composição, configuração, ambiente) e `tools/` (CLIs auxiliares) |
  | `eval` | `src/det_depth_distance/evaluation/` (avaliação contra referências de campo) |
  | `configs` | `configs/` |
  | `tests` | `tests/` (inclui `tests/gpu/`) |
  | `utils` | `utils/` |
  | `deps` | `pyproject.toml`, `uv.lock` |
  | `docs` | `docs/`, `README.md`, `AGENTS.md`, `CLAUDE.md`, `DESCRIPTION.md`, `CHANGELOG.md` |

  Omitir o escopo quando a mudança atravessa muitos componentes; nunca
  inventar escopo fora desta lista. A task que cria uma pasta ou módulo de
  topo novo em `src/det_depth_distance/` atualiza esta tabela na mesma
  entrega.
- Assunto no **imperativo** (`adicionar`, `corrigir`, `atualizar`,
  `remover`), primeira letra minúscula, sem ponto final.
- Ao trabalhar numa task documentada, referenciá-la no assunto ou corpo:
  `(task_DDDD)`.
- **Commits atômicos**: um commit = uma preocupação lógica. O fechamento do
  arquivo de tarefa (`(CONCLUÍDA)`, `Detalhes de Implementação`, `Caderno de
  Testes`) entra no **mesmo commit** da implementação.
- **`CHANGELOG.md` é obrigatório e vivo**: toda feature, correção ou mudança
  relevante entregue ganha entrada no `CHANGELOG.md` (formato
  `## [AAAA-MM-DD] Titulo Descritivo`, pt-BR), mantido homogêneo com os
  demais projetos da casa.

- **Casos especiais deste repositório:**
  - Sem git remote: o código chega ao servidor GPU por
    `utils/sync_gpu_host.sh push` (rsync, inclui `.git`). Não commitar no
    servidor; o repositório local é a fonte da verdade.
  - `_testbed/` (vídeos de teste, saídas, logs) e `weights/` nunca são
    versionados (`.gitignore`).
  - Sem CI: a validação de inferência é manual, no servidor GPU, seguindo o
    Caderno de Testes de cada task.
