# CLAUDE.md — det-depth-distance

Protótipo de pesquisa que combina detecção RF-DETR e profundidade métrica
Depth Anything 3 (DA3) para estimar a distância de cada objeto detectado num
vídeo e renderizar um vídeo anotado com caixas e distâncias.
Arquitetura, pré-requisitos, build e uso: veja o `README.md`. Regras
completas de processo: veja o `AGENTS.md`.

## Documentos companheiros (leia primeiro)

- **`AGENTS.md`** — convenções completas (estilo, nomenclatura, **regras de
  ciclo de vida de tarefas**, **branch protection**, **commits**). É o
  documento autoritativo; este arquivo é complemento, não substituto.
- **`DESCRIPTION.md`** — descrição do projeto para LLMs externos.
  **DEVE ser atualizado no mesmo commit** quando mudarem modelos, endpoints,
  dependências, middleware, auth ou seções principais da UI.
- **`README.md`** — documentação humana completa em pt-BR.
- **`CHANGELOG.md`** — changelog homogêneo da casa; toda entrega relevante
  ganha entrada.
- **`docs/task_DDDD.md`** — toda implementação exige um arquivo de tarefa
  (ver AGENTS.md § "Regras de Ciclo de Vida de Tarefas").

## Comandos comuns

```bash
uv sync                                   # ambiente leve local
uv run pytest                             # testes unitários (sem GPU)
uv run ruff check --fix . && uv run ruff format .
utils/sync_gpu_host.sh push                # código -> gpu-host:~/dev/det-depth-distance
ssh gpu-host 'cd ~/dev/det-depth-distance && ~/.local/bin/uv sync --extra infer'
utils/sync_gpu_host.sh pull [RUN_DIR]      # saídas <- gpu-host
```

## Arquitetura (big picture)

Dois ambientes, um manifesto: o laptop roda só o ambiente leve (numpy,
OpenCV, Pillow) para desenvolver e testar a lógica de distância; o servidor
GPU `gpu-host` (RTX PRO 6000 Blackwell, sm_120) instala o extra `infer`
(torch 2.8.0+cu128, xformers, `rfdetr` e `depth-anything-3` fixados por
commit git). Não há git remote: o código vai por rsync.

## Leia primeiro / Gotchas não-óbvios

- `DA3METRIC-LARGE` NÃO devolve metros: `metros = saída · (fx + fy) / 2 / 300`,
  com fx/fy na resolução do mapa de profundidade (não do vídeo). Ele também
  não tem cabeça de câmera (`pred.intrinsics is None`).
- `Prediction.is_metric` não é confiável (vem como `addict.Dict()` vazio no
  DA3METRIC); o tipo do modelo é decidido pelo `model_name` do checkpoint.
- O mapa do DA3 tem resolução própria (ex.: 280×504 para 1920×1080 com
  `process_res=504`) e proporção ligeiramente diferente; caixas precisam de
  escala separada em x e y, e os métodos `*_crop` ainda deslocam a origem
  (`depth.da3_map_geometry` replica o `InputProcessor` privado do DA3).
- No servidor, `uv sync` sem `--extra infer` remove a stack GPU (o sync é
  exato por padrão); use `--extra infer` no `uv sync` e no `uv run`.
- Decodificador ffmpeg encerrado antecipadamente (`--max-frames`) precisa de
  `kill()`: com SIGTERM ele trava escrevendo no pipe cheio.
- Passar uma lista de imagens para um modelo any-view do DA3 ativa o modo
  multi-view (resultado diferente); quadros são inferidos um a um.
- `numpy<2` é exigido pelo `depth-anything-3`.
- No `gpu-host`: sem sudo, `uv` em `~/.local/bin` (fora do PATH), disco quase
  cheio (preferir `/dev/shm` para temporários grandes), e `nvidia-smi` pode
  falhar por driver mismatch — usar `torch.cuda.mem_get_info()`.

## Ciclo de vida de tarefas e proteção de branch

- **Toda implementação exige um arquivo de tarefa** `docs/task_DDDD.md`
  criado **antes** do código (Descrição, Escopo, Planejamento, Análise de
  Impacto, Verificação, Dependências). Ao concluir, marque o título com
  `(CONCLUÍDA)` e adicione `Detalhes de Implementação` + `Caderno de Testes`
  — no **mesmo commit** da implementação. Regras completas em `AGENTS.md`
  § "Regras de Ciclo de Vida de Tarefas".
- **Nunca commite diretamente em `main`.** Trabalhe em
  `feat/` / `fix/` / `bug/<snake_case>` e integre via pull request.
