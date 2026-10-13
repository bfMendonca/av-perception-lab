# Changelog

Todas as mudancas relevantes do det-depth-distance serao documentadas neste arquivo.

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

1. No laptop: `utils/sync_gpu_host.sh push`.
2. No servidor: `uv sync --extra infer` (instala o ultralytics).
3. Rodar: `uv run --extra infer python -m det_depth_distance --config configs/multi_detector.yaml --set source.max_frames=300`.
4. Testes de GPU (contratos + regressao): `uv run --extra infer pytest -m gpu -v`.

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

1. No laptop, enviar o codigo: `utils/sync_gpu_host.sh push`.
2. No servidor, instalar a stack GPU: `uv sync --extra infer`.
3. Rodar: `uv run --extra infer python -m det_depth_distance VIDEO --layout side-by-side --show-stats`.
4. Trazer o resultado: `utils/sync_gpu_host.sh pull <PASTA_DA_EXECUCAO>`.

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

### Novo: Sincronizacao com o servidor GPU

- `utils/sync_gpu_host.sh push` envia o codigo para
  `gpu-host:~/dev/det-depth-distance` por rsync, sem tocar `.venv/`,
  `_testbed/` e `weights/` remotos.
- `utils/sync_gpu_host.sh pull [RUN]` traz as saidas de `_testbed/outputs/`.

### Novo: Padrao de projeto da casa

- `AGENTS.md`, `CLAUDE.md`, `README.md`, `DESCRIPTION.md`, `CHANGELOG.md`
  e `docs/task_TEMPLATE.md`, com ciclo de vida de tarefas, protecao do
  branch `main` e convencoes de commit.

---
