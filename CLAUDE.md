# CLAUDE.md — av-perception-lab

Protótipo de pesquisa que combina detectores plugáveis (RF-DETR, YOLO26) e
profundidade métrica Depth Anything 3 (DA3) para estimar a distância de cada
objeto detectado num vídeo e renderizar um vídeo anotado com caixas e
distâncias. A execução é descrita por um YAML (`configs/`).
Arquitetura, pré-requisitos, build e uso: veja o `README.md`. Regras
completas de processo: veja o `AGENTS.md`.

## Documentos companheiros (leia primeiro)

- **`AGENTS.md`** — convenções completas (estilo, nomenclatura, **regras de
  ciclo de vida de tarefas**, **branch protection**, **commits**). É o
  documento autoritativo; este arquivo é complemento, não substituto.
- **`DESCRIPTION.md`** — descrição do projeto para LLMs externos.
  **DEVE ser atualizado no mesmo commit** quando mudarem modelos, endpoints,
  dependências, middleware, auth ou seções principais da UI.
- **`README.md`** — visão geral concisa em inglês (instalação e execução);
  a referência detalhada (YAML, calibração, avaliação, saídas) está em
  `docs/reference.md`, ainda em pt-BR.
- **`CHANGELOG.md`** — changelog homogêneo da casa; toda entrega relevante
  ganha entrada.
- **`docs/task_DDDD.md`** — toda implementação exige um arquivo de tarefa
  (ver AGENTS.md § "Regras de Ciclo de Vida de Tarefas").

## Comandos comuns

```bash
uv sync                                   # ambiente leve local
uv run pytest                             # unitários + smoke (sem GPU; pula -m gpu)
uv run pytest -m smoke                    # só o smoke ponta a ponta (tests/smoke/)
uv run ruff check --fix . && uv run ruff format .
uv sync --extra infer                     # stack GPU (no notebook: ver UV_PROJECT_ENVIRONMENT abaixo)
uv run --extra infer pytest -m gpu -v
uv run --extra infer python -m av_perception_lab --config configs/rfdetr_da3.yaml
utils/gpu_pending.sh {list,run} [--dry-run] [--from ID] ...   # pendências de GPU
uv run av-perception-lab-calibrate {calibrate,board,inspect} ...   # calibração (ambiente leve)
uv run av-perception-lab-evaluate --refs REFS.yaml RUN [RUN ...]   # erro contra referências de campo
```

## Arquitetura (big picture)

Dois ambientes, um manifesto: o ambiente leve (numpy, OpenCV, Pillow)
serve para desenvolver e testar a lógica de distância; o extra `infer`
(torch 2.8.0+cu128, xformers, `rfdetr` e `depth-anything-3` fixados por
commit git, `ultralytics==8.4.175`, `transformers` e `timm`) instala a stack
GPU (testada numa RTX PRO 6000 Blackwell, sm_120). Não há git remote.

Código em camadas: `core/` (contratos `Instances`/`DepthResult`/`SemanticMap`/
`MotionEstimate`, ABCs `Detector`/`DepthEstimator`/`MaskRefiner`/
`SemanticSegmenter`/`Tracker`, registro, `LabelMap`, agregadores
`concat`/`nms`/`wbf`, geometria de caixas, concordância, rastreador IoU,
`Pipeline`) não importa frameworks; `motion/` (numpy) estima o movimento 2D
por trilha e prevê trajetórias (modelos `cv`/`ctrv` e ego-motion `csv` no
registro); `adapters/` envolve cada framework (refinadores: `MaskRefiner` → `BoxPromptRefiner` → `HFSamRefiner` →
famílias finas, + `UltralyticsSamRefiner`); `io/` (vídeo, fonte, sinks,
metadados QuickTime); `cli.py` é só a raiz de composição (a câmera —
calibração, conferência da lente, correção de distorção — é composta por
`camera.build_camera`). Calibração (`calibration.py`, `tools/calibrate.py`)
e avaliação contra referências (`evaluation/`, `tools/evaluate.py`) rodam
no ambiente leve; só os gráficos pedem o extra `eval` (matplotlib).

## Leia primeiro / Gotchas não-óbvios

- `DA3METRIC-LARGE` NÃO devolve metros: `metros = saída · (fx + fy) / 2 / 300`,
  com fx/fy na resolução do mapa de profundidade (não do vídeo). Ele também
  não tem cabeça de câmera (`pred.intrinsics is None`).
- `Prediction.is_metric` não é confiável (vem como `addict.Dict()` vazio no
  DA3METRIC); o tipo do modelo é decidido pelo `model_name` do checkpoint.
- O mapa do DA3 tem resolução própria (ex.: 280×504 para 1920×1080 com
  `process_res=504`) e proporção ligeiramente diferente; caixas precisam de
  escala separada em x e y, e os métodos `*_crop` ainda deslocam a origem
  (`adapters/depth/da3.py: da3_map_geometry` replica o `InputProcessor` privado do DA3).
- Com a stack GPU instalada, `uv sync` sem `--extra infer` remove a stack GPU (o sync é
  exato por padrão); use `--extra infer` no `uv sync` e no `uv run` —
  também para `--plot` da avaliação (`--extra eval` sozinho removeria a
  stack; o `infer` já traz o matplotlib).
- No notebook (GPU de 4 GB, `configs/yolo_light_da3.yaml`) o extra `infer`
  vai para `UV_PROJECT_ENVIRONMENT=_testbed/venv-infer`: no `.venv` o próximo
  `uv run pytest` faria o sync exato e removeria a stack (com a variável
  exportada vale o mesmo para o `_testbed/venv-infer`: todo `uv run` leva
  `--extra infer`; `unset` para voltar ao `.venv`). O DA3 (ao contrário
  dos adaptadores `transformers`) **baixa** um checkpoint ausente em execução
  (`_load_da3` sem `local_files_only`): baixe antes. O pico do modelo da focal
  não entra no `metrics.json` (linha de base depois dele).
- Decodificador ffmpeg encerrado antecipadamente (`source.max_frames`) precisa de
  `kill()`: com SIGTERM ele trava escrevendo no pipe cheio.
- Passar uma lista de imagens para um modelo any-view do DA3 ativa o modo
  multi-view (resultado diferente); quadros são inferidos um a um.
- `numpy<2` é exigido pelo `depth-anything-3`.
- Imports de framework (`torch`, `rfdetr`, `depth_anything_3`, `ultralytics`,
  `transformers`, `timm`) só dentro do `_load` dos adaptadores e dos métodos
  chamados depois dele (e em `runtime.DeviceMonitor`):
  `--help`, `--list-models` e a validação do YAML rodam sem torch.
- YOLO (`ultralytics`) trata arrays numpy como **BGR**: o adaptador inverte o
  quadro RGB; esquecer a troca muda rótulos (ex.: `pickup truck` → `truck`).
- As variáveis `YOLO_OFFLINE`, `YOLO_AUTOINSTALL` e `YOLO_CONFIG_DIR` são
  lidas na importação do ultralytics; os módulos dos adaptadores YOLO e
  `ultralytics_sam` chamam `apply_ultralytics_env()` (`setdefault`) antes do
  import lazy (sem telemetria, sem `pip install`,
  configurações em `_testbed/config/`, nunca em `~/.config`). No 8.4 o
  argumento `half` está depreciado: usar `quantize=16`.
- Os nomes de classe do YOLO26 Objects365 vêm do **checkpoint**
  (`model.names`, minúsculos) e diferem do `Objects365.yaml` do ultralytics;
  nunca usar o YAML do dataset.
- Opções de adaptador são dataclasses `frozen, slots, kw_only`; uma filha
  com `__post_init__` próprio chama `DetectorOptions.__post_init__(self)` —
  `super().__post_init__()` sem argumentos levanta `TypeError` com `slots=True`.
- O YOLO26 Objects365 emite caixas sobrepostas de classes de veículo
  diferentes para o mesmo objeto (`pickup truck` × `suv`: 71 pares com
  IoU ≥ 0.5 em ~60 quadros); por isso os mapas agrupam veículos em
  `vehicle` e a fusão/concordância também atuam **dentro** de um detector.
- Scores do YOLO26 Objects365 ficam ~0.2 abaixo dos do RF-DETR nos mesmos
  objetos (mediana 0.71 × 0.90): `aggregation.weights` só reescala, não
  calibra.
- Mapas de rótulos (`configs/labels/`) declaram o `vocabulary` e são
  conferidos na carga (`LabelMap.check`): o mapa COCO num modelo Objects365
  é erro (`truck` tem outro sentido lá). Caminhos relativos do YAML
  (`labels`, `source.video`) partem do diretório de execução (raiz do repo).
- `Instances.sources` não tem `concat_fill`: concatenar instâncias fundidas
  com não fundidas zera o campo; consumidores leem `provenance()`.
- YOLO-seg sem `retina_masks=True` devolve máscaras na resolução do
  **letterbox** de entrada, não do quadro; o adaptador sempre passa
  `retina_masks=True` com pesos `segment`. Para `.pt`, o ultralytics ignora o
  argumento `task=` (a tarefa vem do checkpoint).
- Máscaras do YOLO são recortadas à caixa (`crop_mask`) e predições com
  máscara vazia são descartadas; as do RF-DETR Seg **não** são recortadas
  (a distância ignora pixels fora da caixa). RF-DETR Seg roda em quadrado
  312–768 px (estiramento anisotrópico) com máscara nativa em
  `resolution/4`, interpolada bilinear para o quadro.
- Máscara toda `False` = instância sem máscara (cai para a caixa). A erosão
  `distance.mask_erode` é em **pixels do mapa** de profundidade (1 px do mapa
  ≈ 3,8 px do quadro a 504×280), não do quadro.
- Rótulos de exclusão semântica têm namespace `<nome>.<classe>` (`name` é
  obrigatório em cada item de `semantic`, sem ponto/vírgula); os mapas de
  segmentadores diferentes **não** são fundidos — só a exclusão é a união.
  O prefixo é conferido no `load_config`, a classe só no `Pipeline.load`.
- Pesos de refinadores/SegFormer: `from_pretrained(..., local_files_only=True)`
  em `hf_cache` (`weights/hf`); nada é baixado em execução. Um
  `snapshot_download` sem `allow_patterns` baixa também o `.pt`/`.bin`
  duplicado (dobra o disco). `facebook/sam2.1-*` e `EdgeTAM-hf` são configs
  de vídeo (`sam2_video`/`edgetam_video`) mas carregam em `Sam2Model`/
  `EdgeTamModel`; `facebook/sam3` baixa 3,44 GB (detector + rastreador) para
  usar só o rastreador e é *gated*.
- EdgeTAM exige `timm` (*backbone* `timm_wrapper`). Processadores SAM 1/SAM-HQ
  não aceitam `images=None` e pós-processam com `reshaped_input_sizes`; o
  SAM-HQ devolve o *embedding* como tupla; a cabeça de IoU do SAM 1/SAM-HQ não
  tem sigmoide; `object_score_logits` não altera as máscaras no `transformers`.
  Em `transformers` 5 o `dtype` padrão do `from_pretrained` é `"auto"`: os
  adaptadores passam `float32` explícito.
- SAM e SegFormer redimensionam para quadrado (estiram 16:9); o SAM 1/SAM-HQ
  faz lado maior 1024 + *padding*.
- `ultralytics_sam`: `SAM.predict` descarta máscaras por `conf` (use
  `set_image` + `inference_features`), espera BGR e o peso precisa do nome
  canônico (`mobile_sam.pt`, `sam2.1_l.pt`…): a arquitetura sai do nome.
- TwinLiteNet+ não é pacote: o clone entra no `sys.path` só durante o import
  (o nome de topo `model` fica em `sys.modules`); a escala `int(1/ratio)` do
  demo original só vale para razão inteira (o adaptador interpola para o
  tamanho exato).
- Rastreamento e suavização (`core/tracking.py`, `temporal.py`): todo
  parâmetro temporal é em **segundos**, com `dt` de `Frame.t` (já correto com
  `stride`); nunca contar quadros. A velocidade da caixa é em px/s.
- O `std` por caixa superestima o ruído quadro a quadro da distância
  (1,3–3×) e `std/√n_inliers` o subestima 10–30× (pixels vizinhos do DA3 são
  correlacionados): a incerteza da medida é `max(0.5·std, 1 %·d)`.
- O ruído do DA3 é correlacionado no tempo (AR(1), ρ ≈ 0,7): a suavização
  tira *jitter*, não viés; > 30 m a distância oscila ±2–4 m em ~1 s e a
  velocidade relativa não é confiável.
- Rastreador do ultralytics (ByteTrack/BoT-SORT) **não** é usado: exige
  `lap` (fora do lock), importa `torch` e os efeitos globais do ultralytics,
  conta tempo em chamadas e é AGPL-3.0. O rastreador IoU é numpy puro.
- O rastreador é agnóstico de classe por padrão (detectores alternam
  `car`/`truck` no mesmo objeto) e associa por `source` com `concat`
  (cópias do mesmo objeto por dois detectores mantêm IDs próprios).
- Convenção de pixel: o projeto usa coordenadas **contínuas** (centro do
  pixel `i` em `i + 0,5`: caixas, `MapGeometry`, `ray_factors`); o OpenCV
  centra o pixel em `i`. Ao ler uma calibração, `cx/cy` ganham **+0,5**
  (`Calibration.from_opencv`) e perdem 0,5 ao voltar ao OpenCV
  (`camera_matrix()`); esquecer isso desloca o ponto principal meio pixel
  (e, reescalada por `s`, o erro vira `0,5·(1 − s)` px).
- O `ffprobe` só mostra as tags de `moov/meta`; a lente
  (`camera.lens_model`) e o 35 mm-eq. dos iPhones estão em `moov/trak/meta`
  e só são lidos pelo parser `io/quicktime.py`. O 35 mm-eq. **não** dá a
  focal do vídeo (24 mm-eq. → 1222 px pela diagonal contra 1390/1503 px
  estimados; recorte da estabilização não registrado): só diagnóstico.
- ArUco/ChArUco está no `opencv-python` (`cv2.aruco.CharucoDetector`,
  `Board.matchImagePoints`), mas **sem** `calibrateCameraCharuco`/
  `interpolateCornersCharuco` (API legada do contrib): ChArUco calibra com
  `calibrateCamera` como o tabuleiro. Não instalar `opencv-contrib-python`.
- Arquivos de calibração são `cv2.FileStorage` (`%YAML:1.0`, tags
  `!!opencv-matrix`): **não** são YAML válido para `yaml.safe_load`, e o
  `FileStorage` recusa YAML sem o cabeçalho (o binding levanta
  `SystemError`, não `cv2.error`).
- Com `camera.calibration.undistort` tudo a jusante (caixas do CSV, vídeo,
  `frame_intrinsics`) fica em pixels **corrigidos**; as referências de campo
  são anotadas no vídeo original e a avaliação as transforma com a K e a
  distorção gravadas no `metrics.json` (nunca relendo o arquivo de
  calibração, que pode ter sido refeito).
- `make_run_dir` nomeia a pasta com carimbo de **segundos**: duas execuções
  com as mesmas partes no mesmo segundo e `out_root` levantam
  `FileExistsError`; em `out_root` diferentes geram o mesmo nome, e a
  avaliação recusa (`ConfigError`) runs de mesmo nome — testes usam um
  `output.out_root` por execução e `output.tag` para distinguir runs.
- Dublês de adaptadores em testes: só `monkeypatch.setitem(registry._REGISTRY[kind], ...)`
  após `registry.load_adapters()`; `@register` recusa nome repetido e muda o
  processo para sempre.
- A CLI em processo sem torch precisa trocar `cli.DeviceMonitor` (o
  `start`/`summary` importam torch) e `DA3Depth._close` (importa torch); a
  rede do DA3 é trocável por `da3._load_da3` (ver `tests/smoke/smoke_fakes.py`).
- `motion:` (estágio de movimento 2D, `motion/`, `task_0009`) ≠
  `tracking.motion` (predição da caixa do `IouTracker`). Estado em `(x, z)`
  da câmera (x à direita, z à frente); rumo 0 = afastando, +90° = para a
  direita; taxa de giro > 0 = virando à direita. Sem `ego_motion` tudo é
  relativo à câmera (o giro do carro próprio vira movimento lateral dos
  objetos): use `cv` para todas as classes; `ctrv` sem ego gera aviso. O
  CTRV aquece com CV (`warmup_s`) porque com `v ≈ 0` não observa o lateral.
  Com ego-motion o filtro roda no mundo (REP-103) e as saídas voltam à
  câmera do quadro atual. Padrões do Passo 0: `rel_floor` 0,02 (o ruído do
  DA3 longe tem cauda pesada) e `nis_gate` 30, não 13,8.
- Coluna de texto nova no CSV entra em `evaluation/run_data.TEXT_COLUMNS`
  (a avaliação lê o resto como float) e `tests/run_outputs.py` fixa o
  número de colunas (61 desde a `task_0009`).
- O `MetricsSink` relê `Pipeline.describe()` no `close` (o `pipeline` do
  `metrics.json` é o final): contadores que só fecham no último quadro
  (`motion.ego_motion.clamped_frames`) entram em `describe()`.

## Ciclo de vida de tarefas e proteção de branch

- **Toda implementação exige um arquivo de tarefa** `docs/task_DDDD.md`
  criado **antes** do código (Description, Scope, Plan, Impact analysis,
  Verification, Dependencies). Ao concluir, marque o título com
  `(COMPLETED)` e adicione `Implementation details` + `Test notebook`
  — no **mesmo commit** da implementação. Regras completas em `AGENTS.md`
  § "Regras de Ciclo de Vida de Tarefas".
- **Idioma:** documentação nova e arquivos de tarefa novos (a partir da
  `task_0019`) são escritos em **inglês**; o que já existe em pt-BR continua em
  pt-BR (decisão de 2026-11-01).
- **Nunca commite diretamente em `main`.** Trabalhe em
  `feat/` / `fix/` / `bug/<snake_case>` e integre via pull request.
