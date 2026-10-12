# Changelog

Todas as mudancas relevantes do det-depth-distance serao documentadas neste arquivo.

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
