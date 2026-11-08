> **Detailed reference.** This is the project's original long README, kept for its configuration, calibration, evaluation, output and distance-method details. It is still in Portuguese and has not been audited since the web interface and Prepare were added; for installing and running, use the [README](../README.md).

<p align="center">
  <h1 align="center">av-perception-lab</h1>
  <p align="center">
    Distância por objeto combinando detectores plugáveis (RF-DETR, YOLO26) e profundidade métrica Depth Anything 3
    <br />
    Python 3.11 · PyTorch 2.8 (cu128) · uv
    <br />
    <a href="#início-rápido"><strong>Início Rápido</strong></a>
    ·
    <a href="#uso-da-cli"><strong>Uso da CLI</strong></a>
    ·
    <a href="#saídas"><strong>Saídas</strong></a>
    ·
    <a href="#melhorias-futuras-todo"><strong>Roadmap</strong></a>
  </p>
</p>

---

## Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Funcionalidades](#funcionalidades)
- [Arquitetura e Stack](#arquitetura-e-stack)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Início Rápido](#início-rápido)
- [Testes](#testes)
- [Execução no notebook (GPU de 4 GB)](#execução-no-notebook-gpu-de-4-gb)
- [Uso da CLI](#uso-da-cli)
- [Configuração YAML](#configuração-yaml)
- [Refinadores de máscara](#refinadores-de-máscara)
- [Segmentação semântica](#segmentação-semântica)
- [Rastreamento e suavização](#rastreamento-e-suavização)
- [Estimação de movimento e trajetórias](#estimação-de-movimento-e-trajetórias)
- [Calibração da câmera](#calibração-da-câmera)
- [Entrada MCAP (ROS 2)](#entrada-mcap-ros-2)
- [Avaliação contra referências](#avaliação-contra-referências)
- [Interface web (API de esquema e validação)](#interface-web-api-de-esquema-e-validação)
- [Interface web: front-end e editor de pipeline](#interface-web-front-end-e-editor-de-pipeline)
- [Interface web: vídeos, trecho e processamento](#interface-web-vídeos-trecho-e-processamento)
- [Model weights](#model-weights)
- [Pesos dos refinadores e do SegFormer](#pesos-dos-refinadores-e-do-segformer)
- [Saídas](#saídas)
- [Como a distância é calculada](#como-a-distância-é-calculada)
- [Melhorias Futuras (TODO)](#melhorias-futuras-todo)

---

## Sobre o Projeto

Protótipo de pesquisa que executa, sobre os mesmos quadros de um vídeo, um
ou mais detectores de objetos (RF-DETR, Ultralytics YOLO26) e um estimador
de profundidade monocular métrica (Depth Anything 3). Para cada caixa detectada, os pixels de
profundidade dentro da caixa são agregados com estatísticas robustas (média,
mediana, variância, percentis, rejeição de outliers) para estimar a
distância do objeto à câmera. A saída é um vídeo anotado com caixas e
distâncias, mais um CSV por detecção.

O pipeline é organizado em **componentes com contratos abstratos**
(`Detector`, `DepthEstimator`, agregador, `MaskRefiner`,
`SemanticSegmenter`, `Tracker`): cada modelo é um adaptador que trata as
particularidades do seu framework e entrega sempre os mesmos tipos
(`Instances`, `DepthResult`, `SemanticMap`), e a execução inteira é descrita
por um arquivo YAML (`configs/`). Refinadores da família SAM transformam as
caixas de qualquer detector em máscaras, e segmentadores semânticos (vários
por execução) tornam inelegíveis para a distância os pixels de classes de
fundo (estrada, céu, vegetação, faixas). Um rastreador opcional dá a cada
objeto um ID estável entre quadros, e a distância de cada trilha pode ser
suavizada no tempo (com estimativa da velocidade relativa). A focal pode vir
de uma calibração da câmera (tabuleiro ou ChArUco), e as distâncias podem ser
avaliadas contra medidas de campo (trena laser, marcos, RTK).

O foco é **qualidade da estimativa**, não tempo real: os modelos rodam
em PyTorch, quadro a quadro. A inferência e o desenvolvimento rodam na máquina local;
os testes unitários não precisam de GPU.

---

## Funcionalidades

- **Detectar** objetos quadro a quadro com RF-DETR (`nano`, `small`, `medium`, `large`, ou checkpoint próprio) e/ou Ultralytics YOLO26 (pesos COCO ou Objects365), com filtro de classes por preset do vocabulário (`driving` para COCO e Objects365) validado contra as classes do modelo.
- **Segmentar** instâncias com RF-DETR Seg (`seg-nano` … `seg-2xlarge`, COCO) ou YOLO26-seg (`yolo26*-seg.pt` COCO, `yolo26*-objv1-seg.pt`, vocabulário Objects365 esperado e ainda não conferido em GPU): cada instância ganha uma máscara na resolução do quadro.
- **Combinar** vários detectores no mesmo passo (`aggregation: concat`): todas as caixas são mantidas e marcadas com o modelo de origem (`source`), no CSV e por cor no vídeo.
- **Harmonizar** rótulos entre vocabulários com mapas por detector (`labels: configs/labels/*.yaml`, ex. COCO `car`/`truck`/`bus` e Objects365 `suv`/`van`/`pickup truck` → `vehicle`), conferidos contra as classes do modelo na carga.
- **Fundir** as detecções do mesmo objeto (`aggregation: nms` ou `wbf`), entre detectores e também duplicatas de um mesmo detector, preservando os membros de cada instância fundida (detector, rótulo original, score original) e com peso fixo opcional por detector.
- **Medir a concordância** entre detectores (`agreement`): por par, quantos objetos casam por IoU e quantos são exclusivos de cada um, por quadro e no total do `metrics.json`.
- **Refinar** máscaras a partir das caixas de qualquer detector (`refiners`): SAM 2.1 (`sam2`, padrão dos exemplos), EdgeTAM, SAM-HQ, SAM, SAM 3 (rastreador; *gated*) via `transformers` e MobileSAM/SAM via Ultralytics (`ultralytics_sam`), todos sobre a mesma base `BoxPromptRefiner` (um *embedding* por quadro, caixas em lote, máscara de maior IoU previsto); por padrão só instâncias sem máscara (`targets: missing`), e uma máscara refinada vazia nunca substitui a existente.
- **Segmentar semanticamente** cada quadro com um ou mais segmentadores nomeados (`semantic`: SegFormer Cityscapes, TwinLiteNet+ área trafegável/faixas), sem fundir os mapas, e excluir classes do cálculo de distância por `distance.exclude_semantic: [<nome>.<classe>, …]` (união das exclusões).
- **Rastrear** objetos entre quadros (`tracking: {type: iou}`): associação gulosa por IoU com predição da caixa por velocidade constante (em px/s, tolera `stride` até 5 a 30 fps), agnóstica de classe por padrão (detectores alternam `car`/`truck` no mesmo objeto) e por detector com `concat`; `track_id` no CSV e `#id` no rótulo do vídeo; numpy puro, sem dependência nova.
- **Suavizar** a distância de cada trilha (`smoothing`): Kalman 1D com velocidade (padrão; estima também a velocidade relativa em m/s e as incertezas), média móvel exponencial ou mediana causal, com parâmetros em segundos; a distância bruta continua no CSV.
- **Estimar o movimento** de cada trilha no plano (`motion`): posição lateral e longitudinal (m) pelo ponto de referência da caixa, velocidade relativa (vetor e módulo), rumo e taxa de giro com incertezas, modelo de movimento escolhido pela classe canônica (velocidade constante `cv`; `ctrv` com giro, para uso com ego-motion), **previsão da trajetória** com covariância num horizonte em segundos e ego-motion opcional por arquivo CSV; numpy puro.
- **Calibrar** a câmera com tabuleiro ou ChArUco (`av-perception-lab-calibrate`): vídeo decodificado como nas execuções (ou imagens), rejeição de vistas atípicas, cobertura do quadro, arquivo OpenCV (`FileStorage` YAML/JSON) com K, distorção, desvios e os metadados QuickTime da gravação (lente, 35 mm-eq.); na execução (`camera.calibration`), K reescalada para o vídeo, conferência da lente contra o vídeo, correção de distorção opcional dos quadros e comparação opcional com a focal estimada.
- **Avaliar** as distâncias contra referências de campo (`av-perception-lab-evaluate`): referências por caixa, ponto ou `track_id` num intervalo de quadros/tempo, distância constante ou série (RTK), conversão alcance ↔ z, erros absolutos e relativos (média, mediana, p90, RMSE, viés, `within_2u`) por execução, detector, estimador, faixa de distância, classe, região e produtor da máscara, cobertura, trocas de ID e focal implícita; `eval.json`, `eval.md` e gráficos opcionais.
- **Estimar** profundidade métrica com Depth Anything 3 (`da3metric-large` com focal informada ou estimada; modelos NESTED já em metros).
- **Calcular** a distância de cada objeto com estatísticas robustas sobre a **máscara** da instância (quando o modelo a fornece; reamostrada para o mapa e erodida) ou sobre a caixa: encolhimento da caixa, exclusão de céu, rejeição de outliers (`iqr`, `mad`, `sigma`) e estatística configurável (`median`, `mean`, `trimmed`, `mode`, `pXX`), em profundidade `z` ou distância euclidiana; `distance.region: both` registra as duas distâncias para comparação.
- **Renderizar** vídeo anotado (rótulo `[#id] classe conf · D m ± σ [· ±v m/s] [· v V m/s]`, camadas semânticas e máscara translúcida por instância, trajetória prevista com elipses de incerteza no chão, HUD com tempos), opcionalmente o mapa de profundidade lado a lado com legenda em metros e um painel de vista superior (BEV) das trilhas.
- **Registrar** uma linha de CSV por caixa com todas as estatísticas, `metrics.json` com tempos, VRAM, focal e versões, e opcionalmente os mapas em `depth.npz`.
- **Configurar** a execução inteira por YAML, com ajustes rápidos por `--set chave.pontilhada=valor` e `--list-models` para ver adaptadores e opções.
- **Validar** a configuração (chaves desconhecidas, tipos, faixas, classes) antes de carregar modelos e reportar erros sem traceback (código de saída 2).

---

## Arquitetura e Stack

| Componente | Tecnologia | Versão / origem |
|---|---|---|
| Linguagem | Python | 3.11 |
| Ambiente e dependências | Astral uv | `pyproject.toml` + `uv.lock` |
| Framework de inferência | PyTorch + torchvision | 2.8.0 / 0.23.0 (índice cu128) |
| Detecção | `rfdetr` (Roboflow RF-DETR) | git `1f2af2ea` |
| Detecção | `ultralytics` (YOLO26; licença AGPL-3.0; extra opcional `yolo`, fora do `infer`) | 8.4.175 |
| Profundidade | `depth-anything-3` (ByteDance Seed) | git `3d835ec` |
| Refinadores SAM e SegFormer | `transformers` (Hugging Face) | `>=5.1,<6` (lock: 5.19.0) |
| *Backbone* do EdgeTAM | `timm` | `>=1.0.17` (lock: 1.0.30) |
| Área trafegável e faixas | TwinLiteNet+ (pinned clone `weights/_code/TwinLiteNetPlus`, cloned by `--prepare`; MIT; not a package) | `90f1b86` |
| Atenção eficiente (DA3) | xformers | 0.0.32.post2 |
| Vídeo | ffmpeg (sistema), OpenCV, Pillow | — |
| Configuração | PyYAML | ≥ 6 (ambiente leve) |
| Calibração (tabuleiro, ChArUco, correção de distorção) | OpenCV (`cv2.aruco` incluso; sem `opencv-contrib`) | ambiente leve |
| Gráficos da avaliação | matplotlib (extra opcional `eval`) | `>=3.8` (lock: 3.11.2) |

As dependências pesadas ficam no extra `infer`, instalado apenas no host de
inferência; localmente basta o ambiente leve (numpy, OpenCV, Pillow, PyYAML,
pytest, ruff). Calibração e avaliação rodam no ambiente leve; só os gráficos
da avaliação pedem o extra `eval` (`uv sync --extra eval`, matplotlib; com o ambiente de inferência
não use `--extra eval` sozinho — removeria a stack GPU —, o `--extra infer` já
traz o matplotlib). Os frameworks são importados só dentro do `_load` de cada
adaptador, então `--help`, `--list-models` e a validação da configuração
rodam sem torch.

---

## Estrutura do Projeto

```
av-perception-lab/
├── pyproject.toml           # manifesto uv; extra "infer" com a stack GPU, extra "eval" (matplotlib), extra "web" (FastAPI)
├── web/                     # front-end da interface web (React + TypeScript + Vite; `node_modules/` e `dist/` não versionados)
├── uv.lock                  # lock versionado
├── configs/                 # exemplos de YAML: rfdetr_da3, yolo_o365_da3, multi_detector(_fused), rfdetr_seg_da3, yolo_seg_da3, yolo_o365_seg_da3, rfdetr_sam2_da3, yolo_o365_sam2_da3, rfdetr_semantic_da3, rfdetr_lanes_scene_da3, rfdetr_da3_track, rfdetr_da3_calibrated, yolo_light_da3, rfdetr_da3_motion, rfdetr_sam2_lanes_motion_da3
│   ├── labels/              # mapas de rótulos por vocabulário: coco_driving, objects365_driving
│   └── references/          # example_references.yaml (modelo fictício do formato de referências de campo)
├── src/av_perception_lab/
│   ├── core/                # contratos e orquestração, sem frameworks
│   │   ├── types.py         # Frame, Instances, DepthResult, FrameResult, RefinedMasks, SemanticMap, TrackEstimate, MotionEstimate, Trajectory, MapGeometry, CameraIntrinsics
│   │   ├── options.py       # ConfigError, from_mapping (YAML -> dataclass com caminho nos erros)
│   │   ├── components.py    # Component, Detector, DepthEstimator, MaskRefiner, SemanticSegmenter, Tracker (ABCs) e opções base
│   │   ├── registry.py      # @register, build, available, load_adapters; Stage, register_stage, stages
│   │   ├── labels.py        # vocabulários COCO/Objects365, presets de classes, LabelMap, rótulos <nome>.<classe>
│   │   ├── boxes.py         # IoU, casamento guloso, clusters de NMS e WBF (numpy puro)
│   │   ├── aggregate.py     # Aggregator, Concat, FusionAggregator, Nms, Wbf
│   │   ├── agreement.py     # concordância entre detectores por quadro + resumo
│   │   ├── tracking.py      # IouTracker (IoU + velocidade constante) e TrackSummary (resumo por trilha)
│   │   └── pipeline.py      # Pipeline: detectores -> concordância -> agregação -> refinadores -> rastreador -> profundidade -> semântica -> elegibilidade -> distância -> suavização -> movimento
│   ├── adapters/
│   │   ├── detectors/       # rfdetr.py, ultralytics_yolo.py
│   │   ├── depth/           # da3.py (geometria do mapa, focal, conversão para metros)
│   │   ├── refiners/        # box_prompt.py (base numpy), hf_sam.py (5 famílias transformers), ultralytics_sam.py
│   │   ├── semantic/        # segformer.py, twinlitenet.py
│   │   ├── _hf_hub.py       # carga offline de checkpoints HF + comando de download
│   │   └── _ultralytics_env.py  # política de ambiente do ultralytics (offline, sem auto-instalação)
│   ├── motion/              # numpy puro: movimento 2D por trilha — config, measurement (caixa -> (x, z)), models (cv, ctrv), ego (ego-motion por CSV), estimator
│   ├── io/                  # video.py (ffmpeg), source.py (VideoSource, FrameTransform), sinks.py (CSV, vídeo, npz, métricas), quicktime.py (metadados de câmera do MOV)
│   ├── evaluation/          # references, run_data, matching, stats, evaluate, report: erro contra referências de campo
│   ├── tools/               # CLIs auxiliares: calibrate.py, evaluate.py
│   ├── web/                 # backend da interface web (extra `web`, sem torch): catálogo de módulos, esquema de opções, validação do YAML, fila de execuções, worker, histórico e biblioteca de vídeos; serve o build do front-end
│   ├── camera.py            # provedores de intrínsecos (fixa, estimada, calibrada), Calibration, Undistorter, build_camera
│   ├── calibration.py       # alvos (tabuleiro, ChArUco), detecção de vistas, ajuste com rejeição de vistas atípicas
│   ├── config.py            # RunConfig, load_config, --set; registra os estágios do pipeline
│   ├── distance.py          # numpy puro: caixa ou máscara -> pixels do mapa -> estatísticas robustas
│   ├── temporal.py          # numpy puro: suavização por trilha (Kalman 1D, EMA, mediana) e SmoothingConfig
│   ├── render.py            # caixas, máscaras, rótulos, HUD, painel de profundidade + legenda
│   ├── runtime.py           # memória da GPU, versões, commit
│   ├── cli.py               # --config/--set/--list-models; raiz de composição
│   └── __main__.py          # python -m av_perception_lab
├── tests/                   # pytest sem GPU; run_outputs.py confere pastas de execução
│   ├── smoke/               # smoke ponta a ponta (marcador smoke): cena sintética, dublês, matriz em pares, calibrar -> rodar -> avaliar
│   └── gpu/                 # marcador gpu, roda onde há GPU: contratos, regressão da task_0001, smoke com os modelos reais
├── utils/
│   └── gpu_pending.sh       # roda, em ordem, as pendências de GPU das tasks 0002-0009
├── docs/                    # tasks (docs/task_DDDD.md), template e refs/ (diagramas e protocolo de calibração e referências)
└── _testbed/                # (ignorado pelo git) vídeos, saídas, logs
```

---

## Pré-requisitos

| Requisito | Onde | Observação |
|---|---|---|
| Astral uv | local | gerencia os ambientes e roda os comandos |
| ffmpeg / ffprobe | local | decodificação e codificação por pipe; também usados por parte dos testes |
| GPU NVIDIA com CUDA 12.8 | local | testado numa RTX PRO 6000 Blackwell (sm_120) |
| Weights | local | Everything under the weights root (`weights/`, git-ignored; `AVPL_WEIGHTS_ROOT` overrides), fetched by `--prepare`; nothing is downloaded at run time. See [Model weights](#model-weights) |

> **Nota:** nunca instale o extra `infer` no `.venv` do laptop (a stack GPU ocupa ~9 GB e o próximo `uv run pytest` a removeria); para a GPU do notebook use o ambiente isolado de [Execução no notebook](#execução-no-notebook-gpu-de-4-gb).

---

## Início Rápido

Local (desenvolvimento, testes unitários e lint):

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Com GPU (inferência e testes de GPU: contrato dos adaptadores e
regressão contra a `task_0001`):

```bash
uv sync --extra infer
uv run --extra infer pytest -m gpu -v
```

O `uv run pytest` local nunca roda os testes de `tests/gpu/`
(`addopts = "-m 'not gpu'"`).

---

## Testes

Três níveis, todos com `pytest` (tempos medidos no laptop em 2026-10-23):

| Nível | O que prova | Comando | Onde | Tempo |
|---|---|---|---|---|
| Unitário | cada módulo isolado (dublês fixos, sem torch) | `uv run pytest -m "not gpu and not smoke"` | laptop | ~10 s |
| Smoke (`smoke`) | o sistema inteiro pelo ponto de entrada real: CLI, YAML e `--set`, registro, `Pipeline` com todas as etapas, sinks e arquivos de saída, mais o encadeamento calibrar → rodar → avaliar | `uv run pytest -m smoke` | laptop | ~10 s |
| GPU (`gpu`) | contratos dos adaptadores reais, regressão da `task_0001` e smoke com os modelos reais (5 quadros em 12 exemplos de `configs/`: um por família de adaptador e o exemplo leve, que só roda onde os pesos leves existem) | `uv run --extra infer pytest -m gpu -v` | máquina com GPU | minutos |

O `uv run pytest` padrão roda unitários **e** smoke (~21 s no total); o
`gpu` fica de fora.

**Smoke local** (`tests/smoke/`): um vídeo sintético 320×180 de 12
quadros, gerado no `tmp_path` com o próprio `VideoWriter` (`crf=0`), mostra
planos frontais de profundidade conhecida em tons de cinza (estrada, céu, um
carro que anda e dois anéis cujo furo mostra a estrada ou o céu). Dublês
determinísticos que "veem" o quadro pela paleta — detector (com e sem
máscaras), refinador (pela hierarquia `BoxPromptRefiner`) e segmentador
semântico — entram no registro só durante o teste (`monkeypatch.setitem`);
a profundidade é o adaptador `DA3Depth` **real** com só a rede trocada, que
devolve a profundidade no formato do DA3METRIC. Nove execuções cobrem todo
par de níveis de agregação × região × refinador × exclusão semântica ×
rastreamento+suavização × calibração+`undistort`; cada distância do CSV é
conferida contra o valor analítico da cena (tolerância 1e-4); dois casos
ligam também o estágio `motion` e conferem a velocidade lateral analítica
do carro (2,648 m/s) e a dos objetos parados, e um caso extra roda o
ego-motion por CSV (carro próprio a 5 m/s em linha reta). Um teste fixa
o invariante "tudo desligado = `task_0002`" e outro calibra tabuleiros
renderizados, processa o vídeo distorcido com a calibração e avalia contra
referências derivadas da geometria. `tests/run_outputs.py` confere a
estrutura de qualquer pasta de execução (arquivos, 61 colunas do CSV, chaves
do `metrics.json`, `ffprobe` do vídeo) e é usado também pelo smoke de GPU.

### Pendências de GPU

`utils/gpu_pending.sh` roda na máquina com GPU, na raiz do repo, todas as
execuções de GPU pendentes das tasks 0002–0009 na ordem de dependência
(regressão antes de qualquer download, `-m gpu`, execuções de
validação), com um log por passo e um `summary.tsv` em
`_testbed/outputs/gpu_pending_<carimbo>/`:

```bash
utils/gpu_pending.sh run   # use tmux: há execuções longas
```

| Opção | Efeito |
|---|---|
| `list` | lista os 19 passos (id, task de origem, tipo) |
| `run --dry-run` | imprime os comandos sem executar nem criar nada |
| `--from ID` / `--only ID` | retoma a partir de um passo / roda só um (id completo ou número) |
| `--skip-long` | pula o rastreamento nos vídeos completos (`10-track`) |
| `--allow-downloads` | permite os downloads da `task_0005` em `weights/hf`, com confirmação `[s/N]` por item (sem TTY, nada é baixado) |
| `--sam-hq base\|huge` | qual SAM-HQ baixar (decisão do usuário; sem a opção, nenhum) e usar nas validações E do `18-refiner-runs` (sem a opção, o que já estiver em `weights/hf`; nenhum → `sam_hq` pulado) |
| `--with-optional` | também os opcionais (`sam-vit-base`; `sam3` só com `HF_TOKEN`; MobileSAM só impresso) |

Os downloads da `task_0004` (pesos fora do repositório) nunca são
executados: o passo `12-dl-0004` só imprime os comandos (MANUAL). Execuções
que dependem desses pesos (caixa × máscara, validação B e comparações com
o YOLO-seg) são puladas enquanto eles faltarem — o `rfdetr` baixaria os
pesos Seg sozinho. Cada comando que falha é registrado e o passo segue;
o passo termina FAIL se algum falhou.

### Execução no notebook (GPU de 4 GB)

`configs/yolo_light_da3.yaml` (`task_0010`) é a versão leve da stack
completa para a RTX A1000 Laptop (4096 MiB): YOLO26s-seg COCO (máscara
nativa, sem refinador), DA3METRIC-LARGE, rastreamento IoU + Kalman,
TwinLiteNet+ `small` (`lanes`) e SegFormer-B0 Cityscapes `640-1280` em
640×1152 (`scene`). Pico **estimado** (não medido): ≈ 2,4 GB alocados pelo
PyTorch na fase principal (≈ 3,0 GB no dispositivo; o DA3 domina) e
≈ 3 GB na fase da focal (`da3-large-1.1`, 16 quadros), que roda e é
liberada antes dos modelos principais e **não** entra no `metrics.json`.
Alavancas: `--set camera.estimate.frames=8` ou `--set camera.focal_px=<px>`
(pula o modelo da focal).

Pré-requisito: ~13 GB livres (extra `infer` ~9 GB + pesos ~3 GB + vídeo). O
extra `infer` fica num ambiente isolado em `_testbed/` (ignorado pelo
git), nunca no `.venv`. Com a variável exportada, todo `uv run` leva
`--extra infer` (sem ele, o sync exato remove a stack também desse
ambiente); `unset UV_PROJECT_ENVIRONMENT` antes de voltar ao `.venv` leve.
Nothing is downloaded at run time (DA3 included since task_0021): fetch the
weights once with `--prepare`; TwinLiteNet+ `small.pth` comes from the
upstream Google Drive and is copied by hand (`--prepare` prints the command).

```bash
df -h ~
export UV_PROJECT_ENVIRONMENT=_testbed/venv-infer   # o .venv continua leve
uv sync --extra infer --extra yolo   # yolo = ultralytics (AGPL-3.0), opt-in
# YOLO26s-seg, DA3 (metric + focal), SegFormer-B0 and the TwinLiteNet+ clone under weights/
uv run --extra infer python -m av_perception_lab --config configs/yolo_light_da3.yaml --prepare
mkdir -p weights/_code/TwinLiteNetPlus/pretrained   # small.pth: upstream Google Drive
curl -fL -o weights/_code/TwinLiteNetPlus/pretrained/small.pth "https://drive.usercontent.google.com/download?id=1N7jNsa8P1dM-UevqPRsx6heUpN4L00FJ&export=download&confirm=t"
mkdir -p _testbed/data   # coloque IMG_4021.MOV aqui
uv run --extra infer pytest -m gpu tests/gpu/test_smoke_gpu.py -k yolo_light -v -rs
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -lms 200 > _testbed/outputs/light_vram_IMG_4021.csv &
uv run --extra infer python -m av_perception_lab --config configs/yolo_light_da3.yaml --set source.max_frames=300 --set output.tag=light
kill %1
```

Registrar: `peak_vram_torch_allocated_gb` e `device_mem_used_delta_gb_approx`
do `metrics.json` (fase principal), o máximo de `memory.used` do CSV (inclui
a fase da focal), os tempos por etapa e a focal estimada. **Status:
medição pendente.**

---

## Uso da CLI

Com a stack GPU (a partir da raiz do repo; vídeos em `_testbed/data/`):

```bash
uv run --extra infer python -m av_perception_lab --config configs/rfdetr_da3.yaml
uv run --extra infer python -m av_perception_lab --config configs/multi_detector.yaml \
    _testbed/data/IMG_4022.MOV --set source.max_frames=300 --set render.show_stats=true
uv run --extra infer python -m av_perception_lab --list-models
uv run --extra infer python -m av_perception_lab --config configs/rfdetr_da3.yaml --prepare
```

| Argumento | Efeito |
|---|---|
| `--config PATH` | Arquivo YAML com a execução inteira (obrigatório, exceto com `--list-models`) |
| `VIDEO` (opcional) | Sobrescreve `source.video` |
| `--set chave=valor` | Repetível; caminho pontilhado com índice de lista (`detectors.0.threshold=0.3`); valor lido como YAML (`null`, `true`, números); um valor-mapa ou lista substitui a seção inteira (`--set 'aggregation={type: concat}'`) |
| `--list-models` | Lista detectores, modelos de profundidade, agregadores, rastreadores, refinadores de máscara, segmentadores semânticos, modelos de movimento (`motion`) e provedores de ego-motion (`ego_motion`) registrados com suas opções e padrões |
| `--prepare` | Downloads the weights and pinned code every configured model declares (under `AVPL_WEIGHTS_ROOT`, default `weights/`; nothing is loaded and the video is not probed), prints a status table and exits: 0 only when every model is ready, 1 otherwise (the reason is listed: offline, HTTP 404, gated licence + `HF_TOKEN`, sha256 mismatch, "provide this file"). Downloads go to `<file>.part` and are renamed only when verified (declared size or sha256, or the full `Content-Length`); running it again resumes (only with a declared size or sha256) or is a no-op; Ctrl-C exits 130 and keeps the partial file. Python packages are reported, not installed: a missing base stack prints `uv sync --extra infer`; a model's own extra gets a `uv sync --dry-run` (clean -> the install command; failure -> state `conflict` with the resolver's reason). Also covers the focal model of `camera.estimate` when the focal is estimated |
| `--install-packages` | With `--prepare`: install a model's own `pyproject.toml` extra once its fresh dry-run is clean (`uv sync --locked --inexact` naming every installed extra plus the new one, so nothing is removed; `uv` output is printed). The base stack (`infer`) is never installed: run `uv sync --extra infer --extra web` once yourself |

Ferramentas auxiliares (ambiente leve, sem GPU; ver
[Calibração da câmera](#calibração-da-câmera) e
[Avaliação contra referências](#avaliação-contra-referências)):

| Comando | Efeito |
|---|---|
| `av-perception-lab-calibrate calibrate VIDEO\|--images GLOB --out ARQ` | Calibra a câmera e grava o arquivo de calibração |
| `av-perception-lab-calibrate board --out PNG` | Gera o alvo (tabuleiro ou ChArUco) para impressão |
| `av-perception-lab-calibrate inspect VIDEO [--calibration ARQ]` | Metadados da câmera do vídeo, focal pelo 35 mm-eq. e conferência com uma calibração (código 1 se a câmera não confere) |
| `av-perception-lab-evaluate --refs REFS.yaml RUN [RUN ...]` | Relatório de erro de uma ou mais execuções contra as referências |

Use `--extra infer` também no `uv sync`: sem ele o sync (exato por padrão)
remove a stack GPU. Erros de configuração terminam com `error: …` e código 2
antes de carregar qualquer modelo.

> **Quebra de compatibilidade (task_0002):** as flags antigas (`--det-model`,
> `--depth-model`, `--focal-px`, `--hf-cache`, …) foram substituídas pelo YAML,
> sem aliases. Cada flag antiga corresponde a uma chave da tabela abaixo.

---

## Configuração YAML

Exemplos versionados em `configs/`: `rfdetr_da3.yaml` (comportamento da
`task_0001`), `yolo_o365_da3.yaml` (YOLO26-L Objects365) e
`multi_detector.yaml` (os dois detectores, `layout: side-by-side`) e
`multi_detector_fused.yaml` (os mesmos dois detectores com mapas de rótulos,
fusão `wbf` e concordância), e os três de segmentação de instâncias com
`distance.region: both` e `layout: side-by-side`: `rfdetr_seg_da3.yaml`
(RF-DETR Seg Large), `yolo_seg_da3.yaml` (YOLO26-L-seg COCO) e
`yolo_o365_seg_da3.yaml` (YOLO26-L-objv1-seg, Objects365 esperado — a conferir em GPU —, com o mapa de
rótulos `objects365_driving`). Refinadores e semântica (`task_0005`):
`rfdetr_sam2_da3.yaml` e `yolo_o365_sam2_da3.yaml` (máscaras SAM 2.1 das
caixas, `region: both`), `rfdetr_semantic_da3.yaml` (SegFormer `scene`) e
`rfdetr_lanes_scene_da3.yaml` (TwinLiteNet+ `lanes` + SegFormer `scene`).
Rastreamento e suavização (`task_0006`): `rfdetr_da3_track.yaml`
(`rfdetr_da3.yaml` + rastreador IoU + Kalman, `layout: side-by-side`).
Calibração (`task_0007`): `rfdetr_da3_calibrated.yaml` (`rfdetr_da3.yaml` +
`camera.calibration` com `cross_check`; o arquivo de calibração é criado
pelo usuário, ver [Calibração da câmera](#calibração-da-câmera)).
Leve (`task_0010`): `yolo_light_da3.yaml` (YOLO26s-seg + DA3METRIC-L +
rastreamento/Kalman + TwinLiteNet+ `small` `lanes` + SegFormer-B0 `scene` em
640×1152, `layout: overlay`) para GPU de 4 GB, ver
[Execução no notebook](#execução-no-notebook-gpu-de-4-gb).
Movimento (`task_0009`): `rfdetr_da3_motion.yaml` (`rfdetr_da3_track.yaml` +
mapa de rótulos COCO + seção `motion` com `cv` por classe, `layout: overlay`
e painel BEV), ver
[Estimação de movimento e trajetórias](#estimação-de-movimento-e-trajetórias);
`rfdetr_sam2_lanes_motion_da3.yaml` é a stack completa
(`rfdetr_sam2_da3.yaml` + TwinLiteNet+ `lanes` excluindo `lanes.lane` +
rastreamento/Kalman + `motion` + BEV, `layout: side-by-side`).
Caminhos aceitam `~` e, quando relativos, são
resolvidos a partir do diretório de execução (a raiz do repositório); chaves
desconhecidas são erro (com o caminho, ex.
`detectors[0].treshold: unknown option`).

| Seção | Chaves (padrão) |
|---|---|
| topo | `device` (`cuda:0`) |
| `source` | `video` (obrigatório aqui ou como `VIDEO`) ou `mcap` + `topic` (ver [Entrada MCAP](#entrada-mcap-ros-2)); `start` (0 s); `duration` (0 = até o fim); `max_frames` (0 = todos); `stride` (1) |
| `detectors` | Lista (≥ 1); cada item tem `type`, `name` opcional (padrão: `rfdetr-<variant>`, `rfdetr-<arquivo de pesos>` ou, no YOLO, o nome do arquivo de pesos) e as opções do adaptador. Comuns: `threshold` (0.5, em (0, 1]), `classes` (`driving`, `all`, lista ou nomes separados por vírgula), `labels` (null; YAML de mapa de rótulos, ver abaixo). `rfdetr`: `variant` (`large`; detecção `nano`/`small`/`medium`/`large`, segmentação `seg-nano`/`seg-small`/`seg-medium`/`seg-large`/`seg-xlarge`/`seg-2xlarge`), `weights` (checkpoint próprio), `optimize` (TorchScript + FP16). `yolo`: `weights` (obrigatório, `.pt` de detecção ou de segmentação `*-seg.pt`), `imgsz` (640, múltiplo de 32), `half` (FP16) |
| `depth` | `type: da3`; `model` (`da3metric-large`; alias, repo HF ou diretório; modelos relativos são recusados); `process_res` (504, ≥ 14); `process_res_method` (`upper_bound_resize`); `hf_cache` |
| `camera` | Precedência `calibration` > `focal_px` > `estimate`. `calibration` (null; `path` ou `from_source` (true = K e distorção do `CameraInfo` do MCAP), `undistort` false, `alpha` 0.0, `check_lens` `error`, `cross_check` false — ver [Calibração da câmera](#calibração-da-câmera); exclusivo com `focal_px`); `focal_px` (null; focal em px do vídeo); `estimate` (`model: da3-large-1.1`, `frames: 16`, `process_res`, `process_res_method`, `hf_cache`; `null` desliga a estimativa; usado também por `calibration.cross_check`) |
| `aggregation` | `type`: `concat` (padrão; mantém todas as caixas, marcadas por `source`), `nms` ou `wbf` (opções abaixo) |
| `refiners` | Lista (padrão `[]`), aplicada em ordem depois da agregação (cada um vê as máscaras do anterior); item `{type, name?, …}`, ver [Refinadores de máscara](#refinadores-de-máscara) |
| `semantic` | Lista (padrão `[]`) de segmentadores, rodados em ordem sobre o mesmo quadro; item `{name, type, …}` com `name` **obrigatório** e único (`[a-z0-9][a-z0-9_-]*`), ver [Segmentação semântica](#segmentação-semântica) |
| `tracking` | `null` (padrão: sem rastreamento) ou `{type: iou, name?, …}`, ver [Rastreamento e suavização](#rastreamento-e-suavização) |
| `smoothing` | `null` (padrão) ou seção com `filter` (`kalman`) e parâmetros do filtro; exige `tracking` |
| `motion` | `null` (padrão) ou seção com `models`, `classes`, `default`, `measurement`, `prediction`, `ego_motion` …; exige `tracking`; ver [Estimação de movimento e trajetórias](#estimação-de-movimento-e-trajetórias) |
| `agreement` | `enabled` (true; só tem efeito com ≥ 2 detectores); `iou_thr` (0.5, em (0, 1)); `class_aware` (true: só casa rótulos canônicos iguais) |
| `distance` | `box_shrink` (0.1); `outlier` (`iqr`); `outlier_k` (null = padrão do método); `stat` (`median`, `mean`, `trimmed`, `mode`, `pXX`); `distance_type` (`z` ou `euclidean`); `min_valid_px` (20); `max_range` (0 = sem limite); `mask_erode` (1; erosão da máscara em pixels do mapa, ≥ 0); `exclude_sky` (true); `region` (`mask`: máscara quando a instância tem, senão caixa; `box`: sempre a caixa; `both`: como `mask` e grava também a distância da caixa); `exclude_semantic` (`[]`; rótulos `<nome>.<classe>` — lista YAML ou texto com vírgulas — cujos pixels saem do cálculo, ex. `[scene.road, lanes.lane]`) |
| `render` | `layout` (`overlay` ou `side-by-side`); `show_stats` (false); `draw_inner_box` (false); `max_depth` (30 m); `cmap` (`turbo`); `crf` (20); `hud` (true); `font` (null); `mask_alpha` (0.35, em [0, 1]; opacidade da máscara; 0 não desenha máscaras); `semantic_alpha` (0.4, em [0, 1]; 0 não desenha camadas semânticas); `semantic_overlay` (null = todos os mapas; lista de nomes de `semantic` a desenhar); `semantic_classes` (null = todas as classes; allowlist `<mapa>.<classe>` das classes pintadas, ver [Render class selection](#render-class-selection)); `show_smoothed` (true; rótulo com a distância suavizada quando há estimativa — a bruta segue no CSV); `show_speed` (true; velocidade relativa no rótulo, só com `kalman`); com `motion`: `show_trajectory` (true; trajetória prevista), `trajectory_alpha` (0.8, em [0, 1]; opacidade do 1º segmento, decrescente até o horizonte), `show_motion_speed` (true; `v <m/s>` no fim do rótulo), `bev_inset` (false; painel de vista superior no canto inferior direito — exige `motion`), `bev_size` (0.35 da altura do quadro, em (0, 1]), `bev_range_m` (40 m à frente), `bev_width_m` (20 m de largura) |
| `output` | `out_root` (`_testbed/outputs`); `tag` (""); `save_depth` (false) |

### Mapas de rótulos e fusão entre detectores

Cada detector pode ter um mapa `labels` que traduz os nomes do seu
vocabulário para nomes canônicos comuns. Versionados em `configs/labels/`:
`coco_driving.yaml` (RF-DETR e YOLO26 COCO) e `objects365_driving.yaml`
(YOLO26 Objects365); ambos agrupam as classes de veículo em `vehicle`, porque
os modelos trocam `car`/`truck`/`suv`/`pickup truck` no mesmo objeto.

```yaml
vocabulary: coco        # opcional; conferido contra o vocabulário do modelo carregado
unmapped: keep          # rótulos fora do mapa: keep (mantém o nome original) ou drop (descarta)
canonical:
  vehicle: [car, truck, bus]
  person: [person]
```

Nomes do mapa que não existem no modelo, ou um mapa escrito para outro
vocabulário (ex. o mapa COCO num modelo Objects365, onde `truck` tem outro
sentido), são erro na carga, antes do primeiro quadro. A coluna `class_name`
passa a ser o rótulo canônico; `raw_label` mantém o nome original.

| Agregador | Opções (padrão) | Comportamento |
|---|---|---|
| `concat` | — | Mantém todas as caixas de todos os detectores |
| `nms` | `iou_thr` (0.5); `class_agnostic` (false); `weights` ({}) | Supressão de não-máximos sobre a união das caixas (entre detectores e dentro de cada um), por rótulo canônico; a caixa de maior score ponderado sobrevive com sua caixa e score originais |
| `wbf` | `iou_thr` (0.55); `class_agnostic` (false); `weights` ({}); `conf_type` (`avg` ou `max`); `skip_box_thr` (0.0) | Weighted Boxes Fusion: cada grupo vira uma caixa média ponderada por score; com `avg`, o score é a média dos scores ponderados × min(T, n) / Σpesos (T = nº de detectores), então um objeto visto por só um de dois detectores de mesmo peso fica com metade do score; limitado a 1 (a referência passa de 1 com pesos desiguais) |

`weights` mapeia nome do detector → peso > 0 (ausente = 1); nomes que não
são detectores da execução são erro. **Limitação:** os scores de modelos
diferentes não estão na mesma escala (nos mesmos veículos, mediana de 0.90
no RF-DETR e 0.71 no YOLO26-L Objects365); o peso só reescala os scores para
decidir o vencedor do NMS e a média do WBF — não é calibração. Com fusão por
rótulo e detectores sem nenhum rótulo canônico em comum, o pipeline avisa
(`[warn]` no stderr) na carga. A instância fundida leva `class_id`,
`raw_label` e `source` do membro de maior score ponderado e a máscara (quando
houver) do melhor membro que tenha máscara.

Para rodar o mesmo YAML sem fusão, substitua a seção inteira:
`--set 'aggregation={type: concat}'`; as opções comuns (`iou_thr`,
`class_agnostic`, `weights`) permitem trocar `--set aggregation.type=nms`
sobre o `multi_detector_fused.yaml`.

### Refinadores de máscara

Um refinador recebe as instâncias já agregadas e devolve as **mesmas**
instâncias (mesmo N, mesma ordem) com máscaras na resolução do quadro,
usando cada caixa como *prompt*. O contrato do pipeline é só `MaskRefiner`
(alvos, fusão, proveniência, validação); os modelos "caixa → máscara"
derivam de `BoxPromptRefiner` (`adapters/refiners/box_prompt.py`, numpy
puro), que prepara o quadro uma vez (um *embedding* reusado entre lotes),
decodifica as caixas em lotes de `max_boxes` e escolhe a máscara de maior
IoU previsto. `HFSamRefiner` cobre os modelos `transformers` com a API do
SAM; cada família é uma subclasse fina.

| `type` | Família / classe `transformers` | `model` padrão | Bytes do peso | Licença | *Gated* | *Object score* |
|---|---|---|---|---|---|---|
| `sam2` | SAM 2.1 (`Sam2Model`) | `facebook/sam2.1-hiera-large` | 897 897 416 | Apache-2.0 | não | sim |
| `edgetam` | EdgeTAM (`EdgeTamModel`; exige `timm`) | `yonigozlan/EdgeTAM-hf` | 55 912 824 | Apache-2.0 | não | sim |
| `sam_hq` | SAM-HQ (`SamHQModel`) | `syscv-community/sam-hq-vit-huge` | 2 570 816 384 (`-vit-base`: 379 266 384) | Apache-2.0 | não | não |
| `sam` | SAM 1 (`SamModel`) | `facebook/sam-vit-huge` | 2 564 432 288 (`-vit-base`: 374 979 480) | Apache-2.0 | não | não |
| `sam3_tracker` | SAM 3, só o rastreador (`Sam3TrackerModel`) | `facebook/sam3` | 3 439 938 512 | "SAM License" | **sim** (pedir acesso + `HF_TOKEN`) | sim |
| `ultralytics_sam` | Ultralytics `SAM` (MobileSAM, SAM, SAM 2.x `.pt`) | — (`weights` obrigatório, nome canônico, ex. `mobile_sam.pt`, 40 728 226) | — | AGPL-3.0 (ultralytics); MobileSAM Apache-2.0 | não | não |

Opções em camadas (todas por chave do item da lista):

| Camada | Opções (padrão) |
|---|---|
| todo refinador | `targets` (`missing`: só instâncias sem máscara não vazia na caixa — preserva máscaras nativas de detectores Seg; `all`: todas, para comparar) |
| `BoxPromptRefiner` | `max_boxes` (32, ≥ 1); `multimask` (false; true = melhor de 3 máscaras pelo IoU previsto); `mask_threshold` (0.0, limiar do *logit*); `score` (`iou` ou `object` = sigmoide do *object score*, só famílias com esse sinal); `min_object_score` (null; em [0, 1], esvazia a máscara de caixas sem objeto, só famílias com o sinal) |
| famílias `transformers` | `model` (repo HF no formato `transformers`); `hf_cache` (null = `<weights root>/hf`); SAM-HQ: `hq_token_only` (false) |
| `ultralytics_sam` | `weights` (obrigatório; existência conferida antes do import); `imgsz` (1024, múltiplo de 32) |

`name` é opcional (padrão: nome do repo em minúsculas, ex.
`sam2.1-hiera-large`; no `ultralytics_sam`, o nome do arquivo sem extensão)
e aparece em `mask_source`, em `metrics.json` (`mask_sources`, `refined`,
`components.refiners`), em `timings_ms` (`refine.<nome>`) e no HUD; dois
refinadores do mesmo checkpoint exigem `name` explícito. Trocar o refinador
de um exemplo não exige YAML novo:
`--set 'refiners=[{type: edgetam}]'`.

Notas: os processadores do SAM 1/SAM-HQ só normalizam caixas junto com a
imagem e pós-processam com `reshaped_input_sizes`; a cabeça de IoU deles não
tem sigmoide (`mask_score` pode sair de [0, 1]). O `object_score_logits` não
altera as máscaras no `transformers` (só `min_object_score` o usa). O
`ultralytics_sam` usa `set_image` + `inference_features` (o `SAM.predict`
descarta máscaras por `conf` e quebraria a correspondência 1:1), recebe BGR
e faz *letterbox* — para SAM 2.1 prefira `sam2`.

#### Como adicionar um refinador

1. Criar uma subclasse de `BoxPromptRefiner` (ou de `HFSamRefiner`, para um
   modelo `transformers` com a API do SAM) declarando só o que difere
   (`model_class`/`processor_class`, `uses_reshaped_input_sizes`,
   `has_object_score`, *hooks* `_decoder_inputs`/`_forward_kwargs`; ou
   `_prepare`/`_decode` para outro framework).
2. Declarar a `Options` aninhada derivando de `BoxPromptOptions` (ou
   `HFSamOptions`) com os padrões; um `__post_init__` próprio chama
   `BoxPromptOptions.__post_init__(self)` primeiro (nunca `super()` sem
   argumentos em dataclass `slots=True`).
3. Decorar com `@register("refiner", "<type>")` e, se for um módulo novo,
   listá-lo em `core.registry.ADAPTER_MODULES` (import sem frameworks).
4. Acrescentar o caso a `REFINER_CASES` (`tests/gpu/conftest.py`) para rodar
   o mesmo teste de contrato `-m gpu`, e testes puros dos helpers.
5. Acrescentar o `type` ao conjunto esperado de
   `tests/test_registry.py::test_refiners_and_segmenters_are_registered` (e,
   numa família `HFSamRefiner`, a linha em `FAMILIES` de
   `tests/test_hf_sam.py`): os dois fixam de propósito os refinadores
   registrados.

### Segmentação semântica

Cada item de `semantic` é um segmentador com `name` obrigatório; os mapas
**não são fundidos** (um `SemanticMap` por nome em `FrameResult.semantic`,
cada um com suas classes). `distance.exclude_semantic` usa o *namespace*
`<nome>.<classe>` (separado no primeiro ponto; a classe pode ter espaço, ex.
`scene.traffic light`): o prefixo é conferido no `load_config` e a classe
após a carga do segmentador, ambos com `ConfigError`. Elegíveis =
`~céu ∧ ¬(∪ classes excluídas de todos os mapas)`; sem segmentador (ou sem
classe excluída) o resultado é idêntico ao da `task_0004`.

| `type` | Opções (padrão) | Classes |
|---|---|---|
| `segformer` | `model` (`nvidia/segformer-b5-finetuned-cityscapes-1024-1024`); `hf_cache` (null); `size` ((1024, 1024), altura × largura de entrada, múltiplos de 32 — quadrado estica quadros 16:9) | as 19 do Cityscapes: `road`, `sidewalk`, `building`, `wall`, `fence`, `pole`, `traffic light`, `traffic sign`, `vegetation`, `terrain`, `sky`, `person`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle` |
| `twinlitenet` | `repo` (null = pinned clone `<weights root>/_code/TwinLiteNetPlus`); `variant` (`large`; `nano`/`small`/`medium`); `weights` (null = `<repo>/pretrained/<variant>.pth`); `imgsz` (640); `half` (false) | `background`, `drivable`, `lane` (faixa vence a área trafegável) |

Domínios: Cityscapes e BDD100K são cenas de rua; em cena agrícola
`vegetation`/`terrain` cobrem a cultura — revisar a lista de exclusão.
`--set 'semantic=[{name: lanes, type: twinlitenet}]'` troca a lista inteira,
`--set semantic=[]` desliga a semântica e `--set semantic.<i>.size=[1024,1792]`
edita um item (`semantic.0` em `rfdetr_semantic_da3.yaml`, `semantic.1` em
`rfdetr_lanes_scene_da3.yaml`).

#### Render class selection

`render.semantic_classes` chooses which semantic classes are painted in
`video.mp4` (task_0035). It is an allowlist of `<map>.<class>` entries (split
at the first dot, like `distance.exclude_semantic`):

```yaml
render:
  semantic_overlay: null
  semantic_classes: [scene.road, scene.car, lanes.lane]   # only these are painted
```

- Absent or `null` (the default) paints every class, as before. `[]` paints
  nothing. A semantic map with no entry in the list draws none of its classes.
  A bare string is accepted as a one-item list (`--set render.semantic_classes=scene.road`).
- `background` (TwinLiteNet+) is never drawn, listed or not.
- The `<map>.<class>` form and the map prefix are checked in `load_config`
  (no dot, empty class or unknown map: error). The class
  names are checked right after `pipeline.load`, before the first frame
  (`ConfigError` listing the valid classes), since only a loaded segmenter is
  authoritative about its classes.
- It is presentation only: distances, tracking, `detections.csv`,
  `depth.npz` and `metrics.json` are identical with and without it. Use
  `distance.exclude_semantic` to change the distance. It combines with
  `semantic_overlay` (which picks whole maps) and `semantic_alpha`.
- Detector classes and the MCAP output are not filtered by this field (follow-ups).

Where the web editor gets the class list (`Component.declared_classes(options)`,
asked by `POST /api/classes`, no model is loaded):

| Source | Classes |
|---|---|
| `static` | TwinLiteNet+: `background`, `drivable`, `lane` (the load checks that both heads end in 2 channels) |
| `metadata` | SegFormer: `id2label` of the checkpoint's `config.json`, from the HF cache (`hf_cache`, default `<weights root>/hf`) or from a local `model` directory; plain JSON, no torch or transformers |
| `cache` | Detectors (YOLO, RF-DETR): checkpoints are pickles, so each successful load writes `<weights root>/.classes/<key>.json` (`{"names": [...]}`); `key` hashes the weights file's resolved path, size and mtime, so a replaced file misses |

The cache is a hint only: the loaded model's `class_names` stays
authoritative, a corrupt file is a miss and a failure to write it (read-only
weights root) is logged and ignored. SegFormer follows `refs/main` of the HF
cache like the loader (first sorted snapshot only without a usable ref). When
nothing is known the editor shows the classes after the first load and asks
again each time the dropdown is opened.

### Rastreamento e suavização

`tracking` atribui a cada instância um `track_id` (inteiro ≥ 1, estável
entre quadros) depois da agregação e dos refinadores; `smoothing` filtra, por
trilha, a distância já calculada. Os dois vêm desligados (`null`) e, assim,
as saídas são idênticas às da `task_0005` (colunas novas vazias).

- **Associação** (`iou`): a caixa de cada trilha é prevista no instante do
  quadro (`motion: constant_velocity`, velocidade em px/s por média móvel;
  `none` = caixa parada) e casada um-para-um, por IoU decrescente
  (≥ `iou_thr`), com as instâncias do mesmo **grupo**: o `source` para
  instâncias não fundidas (com `concat`, as cópias do mesmo objeto vistas
  por dois detectores mantêm IDs próprios) e um grupo único para instâncias
  fundidas (`nms`/`wbf`: uma trilha por objeto mesmo quando o representante
  troca de detector); o rótulo entra no grupo só com `class_aware: true`.
- **Ciclo de vida**: instância não casada abre trilha (toda instância tem ID
  desde o quadro em que aparece); trilha sem casamento por mais de
  `max_age_s` morre, e o objeto que reaparece ganha ID novo. O rótulo da
  trilha no resumo é a moda dos rótulos.
- **Tempo em segundos**: `dt` sai de `Frame.t`, então os parâmetros valem com
  qualquer `source.stride`. Numa reprodução *offline* sobre os
  `detections.csv` da `task_0001` (não é execução na GPU), a associação se
  mantém até `stride 5` a 30 fps e fragmenta em `stride 10`.
- **Filtros** (`smoothing.filter`): `kalman` (estado distância + taxa de
  variação, aceleração branca `accel_std`; único que estima
  `rel_speed_mps` — positiva = afastando, negativa = aproximando — e os
  desvios), `ema` (`α = 1 − exp(−dt/tau_s)`) e `median` (mediana causal das
  medidas com `t ≥ t_agora − window_s`). Sem distância no quadro (`n/a`) o
  Kalman só prediz e os demais mantêm o valor; trilha sem medida por mais de
  `reset_after_s` recomeça o filtro.
- **Incerteza da medida**: `σ = max(std_scale · std, rel_floor · d)`, com o
  `std` da caixa/máscara. Nos vídeos da `task_0001` o `std` superestima o
  ruído quadro a quadro 1,3–3× e `std/√n_inliers` o subestima 10–30×.
- **Limites**: o ruído do DA3 é correlacionado no tempo (AR(1), ρ ≈ 0,7): o
  filtro reduz o *jitter* 3,5–4,5× (desvio de `Δd` 0,213 → 0,060 m/quadro e
  0,568 → 0,127 m/quadro nos dois vídeos da `task_0001`, em reprodução
  *offline* dos seus `detections.csv`; validação na GPU pendente) mas só ~20 % do erro de nível, e
  não corrige viés do DA3 ou da focal (`task_0007`). Acima de ~30 m (caixas
  pequenas) a distância oscila ±2–4 m em ~1 s, indistinguível de movimento
  real: a velocidade relativa ali não é confiável (ver `rel_speed_std_mps`).
  O rastreador do ultralytics (ByteTrack/BoT-SORT) não é usado: exige `lap`,
  importa `torch`, conta tempo em chamadas e é AGPL-3.0.

| `tracking` (`type: iou`) | Padrão | Efeito |
|---|---|---|
| `iou_thr` | 0.3 | IoU mínima de um casamento, em (0, 1) |
| `max_age_s` | 0.5 | segundos sem casamento até a trilha morrer (> 0; limite inclusivo); na reprodução *offline* da `task_0001` a maior lacuna dentro de uma trilha foi 0,5 s (no limite) |
| `motion` | `constant_velocity` | predição da caixa; `none` = caixa parada |
| `velocity_alpha` | 0.5 | peso da velocidade mais recente, em (0, 1]; o 1º casamento define a velocidade |
| `class_aware` | false | true: só casa rótulos iguais (quebra trilhas quando o detector troca `car`/`truck`) |

| `smoothing` | Padrão | Efeito |
|---|---|---|
| `filter` | `kalman` | `kalman`, `ema` ou `median` |
| `accel_std` | 2.0 | `kalman`: desvio da aceleração (m/s², > 0); maior = segue mais rápido, suaviza menos |
| `std_scale` / `rel_floor` | 0.5 / 0.01 | modelo de incerteza acima (≥ 0, não ambos 0) |
| `init_speed_std` | 5.0 | `kalman`: desvio inicial da velocidade (m/s) |
| `tau_s` | 0.1 | `ema`: constante de tempo (s; α 0,28 a 30 fps) |
| `window_s` | 0.15 | `median`: janela causal (s; 5 quadros a 30 fps) |
| `reset_after_s` | 1.0 | estado descartado após esse tempo sem medida da trilha |

`--set 'tracking={type: iou, iou_thr: 0.5}'` troca a seção,
`--set tracking=null` desliga e `--set smoothing.filter=ema` troca o filtro.
`smoothing` sem `tracking` é erro de configuração.

### Estimação de movimento e trajetórias

`motion` (`task_0009`) estima, por trilha, o movimento do objeto **no plano**
e prevê sua trajetória. Roda depois da distância (e da suavização, quando
houver) e exige `tracking`; desligado (`null`, padrão), as saídas são as da
`task_0010` (colunas `motion_*` vazias, sem `trajectories.jsonl`).
`--set motion={}` liga com os padrões e `--set motion=null` desliga. Não
confundir com `tracking.motion`, que é a predição da **caixa** do rastreador
IoU.

- **Medida**: o ponto de referência da caixa (`bottom_center`, o contato com
  o chão, ou `center`) é levado ao mapa de profundidade e retroprojetado com
  as intrínsecas do mapa e a distância da instância: `x` (lateral, + à
  direita) e `z` (ao longo do eixo óptico) em metros. Com `distance_type:
  euclidean` o alcance vira `z` pelo fator do raio no centro da região
  amostrada. A covariância propaga `σ_d = max(std_scale · std, rel_floor · d)`
  e o desvio do ponto em pixels (`pixel_std`).
- **Modelos por classe**: `models` declara modelos nomeados (`{type, …}`);
  `classes` mapeia o rótulo canônico (dos mapas de `configs/labels/`, ex.
  `vehicle`, `person`) para um modelo e `default` vale para as demais classes
  (`null` = sem estimativa). O modelo é fixado no nascimento da trilha. `cv`
  é velocidade constante com aceleração branca por eixo; `ctrv` (velocidade e
  taxa de giro constantes, EKF) aquece com `cv` por `warmup_s` porque com
  velocidade ≈ 0 não observa o movimento lateral. Abaixo de
  `min_speed_for_heading` o rumo não é observável e sai `NaN`. Um aviso em
  stderr aparece quando um rótulo de `classes` não é emitido por nenhum
  detector.
- **Convenções**: rumo 0° = afastando-se, +90° = indo para a direita; taxa de
  giro > 0 = virando à direita (vista de cima). Tempos em segundos de
  `Frame.t` (valem com `stride`).
- **Reinício**: uma medida cujo NIS passa de `nis_gate` reinicia o filtro
  (troca de ID ou salto); trilha sem medida por mais de `reset_after_s`
  recomeça.
- **Previsão**: um ponto a cada `step_s` até `horizon_s`, cada um com
  covariância, no referencial da câmera do quadro atual; no vídeo, polilinha
  a partir do ponto de referência com opacidade decrescente e a elipse de
  `n_sigma` projetada no chão (mais achatada quanto mais longe) em ~3 pontos
  do horizonte; no painel BEV, as elipses de todos os pontos.
- **Sem ego-motion** (padrão), tudo é relativo à câmera: o giro ou a frenagem
  do carro próprio aparecem como movimento dos objetos, e um objeto parado à
  frente parece vir na velocidade do carro. Por isso o exemplo usa `cv` para
  todas as classes; um modelo `ctrv` sem `ego_motion` gera um aviso. A
  estimação do movimento próprio pelo vídeo fica para uma task futura.
- **Ego-motion por arquivo** (`ego_motion: {type: csv, path, time_offset_s}`):
  CSV com cabeçalho `t,x,y,yaw` — `t` em segundos no relógio do vídeo
  (`t_csv = Frame.t + time_offset_s`), `x/y` em metros num referencial do
  mundo REP-103 (x à frente, y à esquerda) e `yaw` em radianos, anti-horário,
  com `t` estritamente crescente. A pose é interpolada linearmente (yaw
  desembrulhado); fora do intervalo do arquivo vale a pose da borda com
  velocidade zero, com um único `[warn]` e a contagem em
  `pipeline.motion.ego_motion.clamped_frames`. `camera_mount` posiciona a
  câmera no veículo. Com ego-motion o filtro roda no mundo: a velocidade
  sobre o chão sai em `motion_abs_speed_mps`, e `motion_vx/vz_mps` seguem
  sendo a derivada da posição vista pela câmera. Já o rumo e a taxa de giro
  (e o limiar `min_speed_for_heading`) descrevem o movimento **sobre o
  chão**, com o rumo expresso nos eixos da câmera do quadro atual: com
  ego-motion, `motion_heading_deg` não é `atan2(vx, vz)` (um carro parado
  visto do carro próprio em movimento tem velocidade relativa, mas rumo vazio).
- **Padrões medidos**: na reprodução *offline* das trilhas da `task_0001`
  (Passo 0 do `docs/task_0009.md`), o ponto de referência oscila 0,7–1,6 px
  quadro a quadro (0,25 px robusto); com `rel_floor` 2 % o NIS médio no
  vídeo em linha reta fica ≈ 2 (o ruído do DA3 em longa distância tem cauda
  pesada), e `nis_gate` 30 reduz os reinícios 3–6× em relação a 13,8. Os
  `accel_std` são provisórios. Validado em GPU (passo
  `20-motion`); conferência visual dos vídeos pendente.

| `motion` | Padrão | Efeito |
|---|---|---|
| `models` | `{cv: {type: cv}}` | modelos nomeados `{type, …}` (registro `motion`, ver `--list-models`) |
| `classes` | `{}` | rótulo canônico → nome de modelo |
| `default` | `cv` | modelo das classes não mapeadas; `null` = sem estimativa |
| `reset_after_s` | 1.0 | estado descartado após esse tempo sem medida da trilha |
| `nis_gate` | 30.0 | NIS acima do qual o filtro reinicia (χ²(2) 99,9 % = 13,8) |
| `measurement` | `reference_point: bottom_center`, `std_scale: 0.5`, `rel_floor: 0.02`, `pixel_std: 2.0` | ponto da caixa e modelo de incerteza (`pixel_std` em px do quadro) |
| `prediction` | `horizon_s: 3.0`, `step_s: 0.25`, `n_sigma: 2.0` | horizonte e passo da previsão, incerteza desenhada |
| `ego_motion` | `null` | `{type: csv, path, time_offset_s: 0.0}` ou `null` (referencial da câmera) |
| `camera_mount` | `x_m: 0`, `y_m: 0`, `yaw_deg: 0` | posição da câmera no veículo (x à frente, y à esquerda; yaw + = à esquerda); só com ego-motion |

| Modelo (`type`) | Opções (padrão) |
|---|---|
| `cv` | `accel_std` (2.0 m/s² por eixo), `init_speed_std` (5.0 m/s), `min_speed_for_heading` (1.0 m/s) |
| `ctrv` | `accel_std` (3.0 m/s², longitudinal), `yaw_accel_std` (1.0 rad/s²), `init_yaw_rate_std` (0.5 rad/s), `warmup_s` (0.5 s), `init_speed_std` (5.0), `min_speed_for_heading` (1.0) |

### Calibração da câmera

A distância do `DA3METRIC-LARGE` é proporcional à focal, e a focal estimada
variou ~8 % entre IMG_4021 (1390 px) e IMG_4022 (1503 px), gravados com a
mesma lente segundo os metadados. A calibração mede K e a distorção da
câmera uma vez, no mesmo modo de gravação. Roteiro completo (gravação do
alvo, critérios de aceitação, coleta de referências):
[`docs/refs/protocolo_calibracao_e_referencias.md`](refs/protocolo_calibracao_e_referencias.md).

```bash
uv run av-perception-lab-calibrate board --out _testbed/board.png --dpi 300   # tabuleiro 9×6 cantos, 25 mm
uv run av-perception-lab-calibrate calibrate _testbed/data/CALIB_01.MOV --square-mm 24.9 \
    --note "estabilizacao padrao" --out configs/calibration/iphone14promax_main_1080p30.yaml
uv run av-perception-lab-calibrate inspect _testbed/data/IMG_4021.MOV \
    --calibration configs/calibration/iphone14promax_main_1080p30.yaml
```

- **Alvos**: `chessboard` (`--cols`/`--rows` = cantos internos;
  `findChessboardCornersSB`, recuo para o detector clássico + `cornerSubPix`)
  e `charuco` (`--cols`/`--rows` = quadrados, `--marker-mm`,
  `--dictionary`; aceita o alvo parcialmente visível). O ChArUco usa o
  `cv2.aruco` do `opencv-python` (`CharucoDetector` + `matchImagePoints` +
  `calibrateCamera`; o `calibrateCameraCharuco` legado não existe nele).
- **Ajuste**: um quadro a cada `--every-s` (0,5 s), até `--max-views` (60)
  vistas uniformes no tempo, mínimo `--min-views` (12); `k3` fixo por padrão
  (`--no-fix-k3` ajusta); vistas com RMS > max(3 · mediana, 1 px) são
  descartadas e o ajuste é refeito uma vez; `coverage` = fração de uma grade
  8×6 com cantos. Os critérios de aceitação do protocolo (RMS < 0,5 px,
  `coverage` ≥ 0,6, desvio de fx < 0,5 % de fx) são conferidos e impressos
  (`[accept] ok` ou aviso `[warn] not accepted`). Vistas ChArUco com cantos
  colineares (uma só fileira visível) são descartadas.
- **Arquivo** (`cv2.FileStorage`, YAML `%YAML:1.0` ou JSON pela extensão):
  `image_width`, `image_height`, `camera_matrix` (convenção OpenCV),
  `distortion_coefficients`, e `rms_px`, `std_intrinsics`, `camera`
  (metadados QuickTime da gravação), `target`, `recording` (`--note`),
  `n_views`, `coverage`, `per_view_rms`, `created`, `tool`. Não é YAML válido
  para `yaml.safe_load` (tags `!!opencv-matrix`).
- **Metadados da câmera**: `inspect` lê do MOV `make`, `model`, `software`,
  `lens_model`, `focal_length_35mm` e `lens_f_number` (os da trilha não
  aparecem no `ffprobe`) e mostra a focal pelo 35 mm-eq. só como
  diagnóstico (24 mm-eq. → 1222 px em 1920×1080 pela diagonal; não serve
  como focal do vídeo).

| `camera.calibration` | Padrão | Efeito |
|---|---|---|
| `path` | — (obrigatório) | arquivo de calibração; conferido antes de carregar qualquer modelo |
| `undistort` | false | corrige a distorção de cada quadro (`remap`) antes de todos os componentes; detecções, profundidade, CSV e vídeo ficam em pixels corrigidos, e a K entregue é a da câmera corrigida |
| `alpha` | 0.0 | `getOptimalNewCameraMatrix`, em [0, 1]: 0 = só pixels válidos (focal um pouco maior), 1 = campo inteiro (bordas pretas); só com `undistort` |
| `check_lens` | `error` | compara `model`, `lens_model` e `focal_length_35mm` da calibração com os do vídeo: `error` aborta, `warn` avisa, `off` não compara; sem metadados nos dois lados = `unavailable` |
| `cross_check` | false | também estima a focal (`camera.estimate`) e registra a estimada e a razão estimada/calibrada no `metrics.json` |

A calibração é reescalada para o tamanho do vídeo só com a mesma proporção
(outra proporção = outro recorte do sensor → erro); calibrar no modo exato
das gravações é o recomendado. Com `calibration: null` (padrão) a execução é
a da `task_0006` (o `metrics.json` só ganha `frame_intrinsics`, `camera: {}`
e `video.transform: null`).

### Entrada MCAP (ROS 2)

Em vez de um vídeo, a execução pode ler um tópico de imagem de um arquivo
`.mcap` (`sensor_msgs/msg/Image` ou `CompressedImage`, CDR; `task_0033`).
Arquivos indexados e não indexados são aceitos. Os módulos `mcap` e
`mcap-ros2-support` são dependências base, e o PyAV também (escreve o vídeo
de taxa variável).

```yaml
source:
  mcap: recordings/drive.mcap      # exclusivo com source.video
  topic: /camera/front/image_raw   # obrigatório com mcap
  duration: 10.0
  stride: 2                        # conta mensagens, não quadros de vídeo
detectors: [{type: rfdetr, variant: large}]
depth: {type: da3}
camera:
  calibration: {from_source: true}   # K e D do CameraInfo pareado
# ou, no lugar de camera: remover a distorção antes de todos os estágios
# rectify: {enabled: true, alpha: 0.0}
```

- **Chaves**: `source.mcap` e `source.topic` (um `topic` é obrigatório e
  `mcap` é exclusivo com `video`); `camera.calibration.from_source` (true =
  intrínsecos do `CameraInfo` do tópico; exclusivo com `path` e com
  `undistort`; exige `source.mcap`); seção `rectify` (`enabled` false,
  `alpha` 0.0, como o `alpha` da calibração) e estágio `rectify`
  (opcional, logo depois da fonte).
- **Base de tempo**: `Frame.t` usa o `header.stamp` quando todos os carimbos
  das primeiras 50 mensagens são > 0 e não decrescentes; senão, o `log_time`.
  Mensagens cujo tempo não é maior que o da última mantida são descartadas e
  contadas em `duplicates_dropped`; havendo descarte, o fim da execução
  imprime um `[warn] source.mcap: ...` no stderr (contagem, % do tópico, base
  de tempo) e, se o tempo volta atrás no meio do tópico (carimbo reiniciado
  depois das 50 primeiras mensagens: sim time, gravações concatenadas), a
  mensagem em que voltou (`stamp_regression_at`) e quantos descartes vêm
  depois dela (`dropped_after_regression`). O aviso também aparece no log da
  interface web. `t` conta a partir da primeira mensagem
  **selecionada** (o `t` de um vídeo inclui o `start`); `start`/`duration`
  selecionam por tempo e `stride` conta mensagens.
- **Decodificação**: os tempos saem dos bytes crus (o `header.stamp` logo
  depois do cabeçalho CDR), então só as mensagens selecionadas passam pelo
  decodificador de imagem; as puladas por `start`/`stride`, as descartadas e
  as não amostradas (`sample`, estimativa da focal) são lidas mas não
  decodificadas. A amostragem lê os carimbos da seleção uma vez e busca só as
  mensagens amostradas (pelo índice de chunks, quando há).
- **Codificações**: `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8` e, em
  `CompressedImage`, `jpeg`/`png`. Outras (`16UC1`, `compressedDepth`, ...)
  aparecem na lista de tópicos da interface, desabilitadas com o motivo, e
  recusadas na execução. Um quadro acima de `MAX_IMAGE_PIXELS` (2^27 ≈ 134 Mpx)
  é recusado antes de decodificar (`McapError`; na interface, 422): em
  `CompressedImage` o tamanho vem do cabeçalho PNG (IHDR) ou JPEG (SOFn) —
  dados que não são PNG nem JPEG são recusados —, em `Image` dos campos da
  mensagem.
- **`CameraInfo`**: pareado ao tópico de imagem pelo maior prefixo de
  namespace em comum. Uma mensagem crua (D não nulo) usa K e D; uma retificada
  (D vazio ou zero) usa o 3×3 à esquerda de P, sem distorção. Modelos
  `plumb_bob` e `rational_polynomial`; `equidistant`, `roi` parcial (zeros
  ou o quadro inteiro, na ordem `x_offset, y_offset, height, width`, valem
  como sem ROI),
  `binning` > 1, tamanho de D inválido ou tamanho diferente do dos quadros
  são recusados (`ConfigError`).
- **`rectify`**: exige um tópico de MCAP com `CameraInfo` **com distorção**
  (sem `CameraInfo`, ou já retificado, é erro antes de carregar os modelos) e
  é recusado com `camera.focal_px` ou `camera.calibration.path` (com
  `from_source` é permitido: é a mesma fonte). Os quadros são retificados
  antes de todos os componentes e a K entregue à profundidade é a retificada.
  Diferente de `camera.calibration.undistort`, que usa um arquivo de
  calibração e não existe com `from_source`. Na interface a faixa
  "Rectification" só aparece com entrada MCAP e a chave fica desabilitada
  quando o tópico não tem `CameraInfo` utilizável ou já é retificado.
- **Vídeo**: com MCAP o `video.mp4` tem **taxa variável** (cada quadro no seu
  `t`, base de 1/90000 s, sem áudio). Tamanhos ímpares são recusados pelo
  escritor.
- **Saídas**: a descrição da fonte fica em `metrics.json["video"]` (não há
  chave `source`): `path`, `topic`, `schema`, `encoding`, `width`, `height`,
  `count`, `nominal_fps`, `time_base` (`header` ou `log`),
  `duplicates_dropped`, `stamp_regression_at`, `dropped_after_regression`,
  `camera_info_topic`. Com `from_source` ou `rectify`
  o `metrics.json` grava a K e a distorção usadas, e a avaliação as usa para
  transformar as referências.
- **Limites conhecidos**: a avaliação de uma execução MCAP com
  `source.start > 0` é recusada (`ConfigError`: o `t` do MCAP não inclui o
  `start`); tamanhos de quadro ímpares são recusados pelo escritor de vídeo.
- **Risco aceito**: a descompressão dos chunks (zstd/lz4) é da biblioteca
  `mcap`, sem teto próprio além do limite de 4 GiB por registro; um MCAP
  forjado pode declarar chunks enormes (bomba de descompressão) e o upload
  na interface os descompacta no processo do servidor. É uma ferramenta de
  pesquisa local: abra só MCAPs de origem conhecida.

### Avaliação contra referências

`av-perception-lab-evaluate` cruza o `detections.csv` e o `metrics.json` de
uma ou mais execuções com um arquivo de referências de campo (um por vídeo;
modelo fictício em `configs/references/example_references.yaml`):

```bash
uv run av-perception-lab-evaluate --refs refs/IMG_4100.yaml _testbed/outputs/<execução>
uv run av-perception-lab-evaluate --refs refs/IMG_4100.yaml RUN_A RUN_B --out _testbed/eval_4100
uv run --extra eval av-perception-lab-evaluate --refs refs/IMG_4100.yaml RUN_A --plot   # laptop
# com o ambiente de inferência: uv run --extra infer av-perception-lab-evaluate ... --plot (nunca --extra eval sozinho)
```

- **Referências**: `frames: [a, b]` ou `t: [a, b]`; `select` por `box`,
  `point` (menor caixa que contém o ponto) ou `track_id`, com `label`
  opcional; `distance_m` ou `series: [[t, d], …]`; `instrument` (incerteza
  `hypot(uncertainty_m, uncertainty_rel · d)`) ou `uncertainty_m`;
  `quantity: range` (linha reta) ou `z`. Coordenadas em pixels do vídeo
  original; execuções com `undistort` têm as referências transformadas com a
  K e a distorção registradas no `metrics.json` da execução
  (`camera.calibration`), não com o arquivo de calibração — recalibrar
  depois não muda a avaliação de execuções antigas (a K corrigida
  reconstruída é conferida contra a registrada).
- **Casamento**: em cada quadro processado do intervalo e para cada detector
  (com fusão, a saída fundida é um grupo só), a instância de maior IoU
  (`--iou-min` 0.3), a menor caixa com o ponto ou a do `track_id`. Quadro sem
  candidato conta como "não detectado". Referências `range` viram `z` (e
  `z` vira alcance numa execução `euclidean`) pelo raio do centro da caixa.
  Referência por `track_id` numa execução sem rastreamento é ignorada, com
  aviso.
- **Estimadores** (os presentes no CSV): `distance_m`, `median`, `mean`,
  `p10`, `p25`, `p75`, `p90`, `min`, `box_distance_m`, `distance_smooth_m`.
- **Estatística**: erro = estimativa − referência; `n`, `n_refs`, viés,
  mediana com sinal, MAE, mediana e p90 absolutos, RMSE, relativos (médio,
  mediano, p90), `within_2u`, `scale_median`. Nível `reference` (padrão:
  uma mediana por referência, quadros vizinhos são correlacionados) ou
  `frame`. Tabelas por execução × detector × estimador e por faixa de
  distância (`--bins`, padrão 0,5,10,20,30,50 m), classe, região
  (máscara/caixa), produtor da máscara e referência; cobertura (válida,
  `n/a`, não detectada, `track_id` distintos) e focal implícita por
  detector (`metric_focal`: focal que zeraria o erro de escala).
- **Saídas**: `eval.json`, `eval.md` e, com `--plot` (extra `eval`),
  `eval_error_vs_distance.png` e `eval_abs_rel_by_bin.png`, em
  `<execução>/eval_<nome das referências>/` ou `--out` (obrigatório com
  várias execuções). Vídeo diferente entre referências e execução é erro
  (`--force` avalia mesmo assim).

### Interface web (API de esquema e validação)

A `task_0013` cria o pacote `av_perception_lab.web`, o primeiro passo da
interface web (editor visual de pipeline). Esta entrega é só o backend de
esquema e validação, sem torch:

```bash
export UV_PROJECT_ENVIRONMENT=_testbed/venv-infer
uv sync --extra infer --extra web
uv run --extra infer --extra web av-perception-lab-web    # 127.0.0.1:8000
curl -s localhost:8000/api/catalog | head -c 400
```

- `GET /api/catalog`: estágios (de `core.registry.stages()`), opções de cada
  adaptador por família e de cada seção de configuração, com `catalog_version`.
- `POST /api/config/validate`: `{config}` -> `{errors: [{path, message}]}`
  (sempre 200; vazio = válido).
- `POST /api/config/parse` / `dump`: YAML <-> mapeamento; chaves de topo
  desconhecidas voltam em `extras` (comentários e ordem das chaves se perdem).

Execuções (`task_0014`): `POST /api/runs` (`{config, video_id?}`) entra numa
fila de um job por vez; cada job é um subprocesso
`python -m av_perception_lab.web.worker` (o servidor continua sem torch).
`GET /api/jobs/{id}` dá o estado, `GET /api/jobs/{id}/events` o progresso por
SSE (`started`, `frame`, `done`, `error`, `end`; quem chega tarde recebe tudo
desde o início) e `DELETE /api/jobs/{id}` cancela (mata o worker e o ffmpeg).
O worker grava na pasta da execução `config.yaml` (efetivo), `web_run.json` e
`preview.jpg` (no máximo a cada 0,5 s). `GET /api/runs` lista as pastas de
`output.out_root` (inclusive as antigas ou incompletas),
`GET /api/runs/{id}/{video|preview|config|detections}` serve os arquivos e
`GET /api/gpu` informa a memória livre (`null` sem CUDA). Se um worker cair
(ex.: falta de memória na GPU) o job termina `failed` com o fim do log.

Nada em `web/` cita um modelo ou estágio: o catálogo é função dos registros.
Para dicas de interface (rótulo, unidade, faixa) use
`field(default=..., metadata={"ui": {...}})` no campo. Checklist para novos
modelos, estágios e visões em `docs/web_extending.md`.

### Interface web: segurança, rede local e desligamento

A `task_0018` endurece o servidor (continua sem autenticação):

- **Host e Origin**: só responde aos nomes `127.0.0.1`, `localhost`, `[::1]`,
  ao `--host` (se não for `0.0.0.0`/`::`) e a cada `--allowed-host NOME`
  (repetível; o endereço ou nome que outras máquinas usam no navegador);
  outro `Host` dá 400 (barra *DNS rebinding*). POST/PUT/PATCH/DELETE com
  `Origin` de outro site (ou `Sec-Fetch-Site: cross-site`) dá 403; sem
  `Origin` (curl) passa; `--dev-origin URL` (repetível) libera um front-end
  servido em outra origem.
- **Vídeo**: sem `video_id`, o `source.video` precisa estar numa `--data-root`
  ou ser o original de um vídeo da biblioteca (resolvido, symlink seguido);
  senão 422 com a mesma mensagem exista o arquivo ou não.
- **Outros caminhos** do YAML (`weights`, `labels`, `repo`, `hf_cache`,
  `camera.calibration.path`, `render.font`, e qualquer campo `Path` futuro:
  descobertos pelas anotações dos esquemas) não podem cair na biblioteca nem
  em `--out-root` (pastas que o cliente escreve). Com `--asset-root PASTA`
  (repetível) todos precisam estar dentro de uma delas — inclusive os
  padrões. Com `--host` fora do
  loopback e sem `--asset-root` (ou sem `--allowed-host` em `0.0.0.0`) o
  servidor avisa na partida. Na rede local, use as três flags, por exemplo:
  `--host 0.0.0.0 --allowed-host 192.168.0.7 --asset-root weights
  --asset-root configs --asset-root ~/dev`.
- **Envio**: 413 acima de 8 GiB (pelo `Content-Length`, antes de ler, ou
  contando), `meta` até 64 MiB, cabeçalho de parte até 8 KiB, até 16
  cabeçalhos e 8 partes (400), no máximo 2 envios simultâneos (429) e 507
  sem espaço livre (envio + 256 MiB) no disco da biblioteca. O original
  enviado guarda o sufixo só se for de vídeo (`.mp4 .m4v .mov .mkv .webm
  .avi .ts .h264 .264`; senão `original.bin`).
- **ffprobe/ffmpeg/PyAV** da biblioteca abrem só arquivos locais
  (`-protocol_whitelist file,pipe`) e só os demuxers aceitos
  (`-format_whitelist mov,matroska,avi,mpegts,h264,hevc`): listas de
  reprodução e arquivos concat são recusados. A prévia tem timeout
  (max(600 s, 10 × duração)); estourado, fica `failed` e vai para o log.
- **Arquivos servidos** (`/api/videos/{id}/…`, `/api/runs/{id}/…`): tipo
  explícito, `X-Content-Type-Options: nosniff` e `Content-Security-Policy:
  sandbox`; fora de vídeo, PNG/JPEG/WebP/GIF, CSV, texto, JSON, NDJSON e YAML
  o arquivo é baixado (`Content-Disposition: attachment`), nunca aberto como
  página. Ids de execução começando com `.` não existem (`.web_jobs`).
- **YAML e corpos**: corpos JSON de POST/PUT/PATCH/DELETE até 1 MiB (413);
  `POST /api/config/parse` recusa (422) documento que, com os aliases
  expandidos, passa de 100 000 valores, ou alias recursivo (âncoras comuns
  continuam valendo).
- **Ctrl-C**: o uvicorn espera no máximo 3 s pelas conexões (um SSE aberto
  nunca termina), depois fecha e mata o worker; todo job pendente termina
  `cancelled` ("the server shut down") no registro e no `web_run.json`. Um
  registro deixado `running`/`queued` por um servidor morto (kill -9) vira
  `failed`/`cancelled` ("interrupted…") na partida seguinte.
- **Jobs após recarregar**: `GET /api/jobs` lista os jobs não terminados e os
  20 últimos terminados (mesmo formato de `GET /api/jobs/{id}`, mais
  `created`, `video_id` e `video`); a página volta a seguir os não terminados.
  O SSE numera os eventos (`id:`) e uma reconexão com `Last-Event-ID`
  continua dali (o progresso não regride); um job que o servidor reiniciado
  não conhece mais (404) é encerrado no painel.
- **Histórico**: execuções de vídeos da biblioteca levam o nome do vídeo
  (não `original.mp4`).

### Interface web: front-end e editor de pipeline

A `task_0015` cria o front-end em `web/` (React 18 + TypeScript + Vite,
interface em inglês, só tema escuro). O build (`web/dist`) não é versionado;
o `av-perception-lab-web` o serve em `/` quando existe (rotas fora de `/api`
sem extensão caem no `index.html`; arquivo ausente em `assets/` ou com
extensão dá 404) e, sem ele, serve só a API.

Node sem sudo (tarball oficial LTS em `~/.local/share/node`, sem editar
arquivos de shell; o `PATH` vai em cada comando):

```bash
V=v24.22.2; D=~/.local/share/node; mkdir -p $D/dl && cd $D/dl
curl -fsSLO https://nodejs.org/dist/$V/node-$V-linux-x64.tar.xz
curl -fsSLO https://nodejs.org/dist/$V/SHASUMS256.txt
grep " node-$V-linux-x64.tar.xz\$" SHASUMS256.txt | sha256sum -c -
tar -xJf node-$V-linux-x64.tar.xz -C $D && ln -sfn $D/node-$V-linux-x64 $D/current
cd ~ && rm -rf $D/dl
export PATH=$D/current/bin:$PATH   # só na sessão atual
```

Build, testes e uso:

```bash
cd web
npm ci                 # dependências exatas do package-lock.json
npm test               # vitest (jsdom)
npm run build          # tsc -b + vite build -> web/dist
cd ..
export UV_PROJECT_ENVIRONMENT=_testbed/venv-infer
uv run --extra infer --extra web av-perception-lab-web --port 8000   # http://127.0.0.1:8000
# rede local: --host 0.0.0.0 --allowed-host 192.168.0.7 --asset-root weights ... (ver "segurança" abaixo)
# outro build: --static-dir DIR
# desenvolvimento: `cd web && npm run dev` (proxy de /api para AVPL_API, padrão http://127.0.0.1:8000;
#   o proxy repassa o Host do navegador; outro front-end: --dev-origin http://localhost:5173)
```

O editor:

- **Pistas** na ordem de `catalog.stages` (estágios com `general = false`;
  estágios consecutivos com o mesmo `group` dividem uma pista com o título do
  grupo — os nativos `tracking` e `smoothing` formam "Tracking & filter"):
  estágio com família de adaptador mostra um cartão por especificação
  (`{type, name?, ...}`) e "+ Add" com os tipos registrados; estágio com
  seção mostra um cartão da seção; opcionais vazios aparecem como "Off".
- **Inspetor** (à direita) segue o clique: tipo do módulo, nome da
  instância e os campos gerados do esquema — slider (int/float com
  `min`/`max` na dica), select (`enum`), switch (`bool`), texto/caminho,
  lista, grupo aninhado (`object`); campos com `hint.advanced` ficam em
  "Advanced", recolhido. `map`, `any` e qualquer tipo desconhecido viram um
  editor YAML bruto (validado ao sair do campo). Chaves fora do esquema
  aparecem como "Keys not in the schema". A engrenagem **General settings**
  abre as seções `general = true` (camera, distance, agreement, render,
  output), as chaves de topo sem estágio (ex.: `device`) e os `extras`.
- **Validação** no servidor 400 ms após a última mudança; o erro aparece no
  campo pelo caminho pontilhado, no cartão (contador) e na barra superior.
- **Abrir/Salvar YAML**: abrir usa `POST /api/config/parse` (botão, Ctrl+O ou
  arrastar o arquivo); salvar baixa o texto de `POST /api/config/dump` e, na
  primeira vez, avisa que comentários e ordem das chaves se perdem. O painel
  **Changes** mostra o diff contra o último YAML aberto ou salvo. O rascunho
  fica no `localStorage` do navegador entre recarregamentos.
- **Presets** (`task_0030`): o botão **Presets** lista as configurações
  prontas (`configs/*.yaml`, nomes e descrições em `configs/presets.yaml`,
  na ordem de exibição; um YAML novo em `configs/` precisa entrar ali, e um
  teste confere) e as do usuário. Carregar um preset troca o pipeline mas
  mantém `source` e `output` do editor; `device` faz parte do preset.
  **Save as preset...** grava o pipeline atual como
  `<--presets-root>/<slug>.yaml` (padrão `_testbed/presets/`), com o nome
  digitado num comentário `# name:` na primeira linha; `source` e `output`
  nunca são gravados. Nome já usado pede confirmação para sobrescrever (409
  sem `overwrite`); os embutidos são somente leitura (apagar dá 403). Rotas:
  `GET /api/presets`, `GET /api/presets/{id}`, `PUT /api/presets`,
  `DELETE /api/presets/{id}`.
- **Process** roda o trecho escolhido no painel de vídeo (próxima seção).

À prova de futuro: nada em `web/src` nomeia estágio, modelo ou campo; um
estágio novo vira uma pista genérica, um tipo de campo novo cai no editor
YAML e um `catalog_version` maior que o suportado mostra só um aviso. Telas
próprias entram por `registerStageView(yamlKey, view)` e
`registerFieldControl(match, control)` (`web/src/editor/registry.ts`).

### Interface web: vídeos, trecho e processamento

A `task_0016` liga o **Process**. Acima do editor fica o painel de vídeo:

- **Entrada**: solte um vídeo em qualquer ponto da página (ou "Choose
  video"); MP4/MOV/MKV tocam na hora. Um `.h264` da Raspberry Pi vai junto
  com o seu `.meta.json` (arraste os dois) e é convertido para `.mkv` de
  taxa variável com os `SensorTimestamp` (mesmo algoritmo de
  `data/s000032/remux_vfr.py`; com PyAV, extra `web`, os timestamps são
  idênticos aos do script). Também dá para colar um caminho do servidor —
  só dentro das pastas `--data-root` (padrão `data`; caminho relativo parte
  da raiz do repo, como na CLI; o arquivo é referenciado, não copiado; `../`
  e symlink para fora dão 403; se o original sumir ou mudar, a entrada vira
  "File missing" e reimportar o mesmo conteúdo a conserta) — ou escolher um
  vídeo já na biblioteca. "Use the pipeline's video" só aparece quando o
  `source.video` do YAML cai numa pasta permitida.
- **Biblioteca** em `--library-root` (padrão `_testbed/library`): uma pasta
  por vídeo com id = 12 hex do sha256 (reenvio não duplica), `record.json`,
  original, `thumb.jpg` e `preview.mp4` (H.264 que o navegador toca, gerado em
  segundo plano com `nice`; o registro mostra `preview: null` até ficar
  pronto). O envio é gravado em disco por blocos (nunca em memória), montado
  em `.incoming/` e só entra na biblioteca completo: arquivo que não é vídeo
  ou truncado dá 422 sem deixar nada.
- **Trecho**: barra com duas alças (arrastar alça ou o trecho, setas do
  teclado; Shift = ×10), cabeça de reprodução, campos Start/Length/Stride,
  botões In/Out na posição do player e estimativa ao vivo
  ("65 frames · ~33 s at 2 fps"). O player repete o trecho.
- **Process** (barra superior ou painel) envia `POST /api/runs` com o YAML do
  editor mais `source.video` (caminho do original no servidor),
  `source.start`, `source.duration`, `source.stride` e `video_id` (o servidor
  define `output.out_root` e um `output.tag` único); **Process entire video** usa `start = 0` e `duration = 0`. Com
  menos de 4000 MiB livres na GPU (`GET /api/gpu`) pede confirmação. Vários
  cliques entram na fila.
- **Jobs**: posição na fila, progresso (quadros / esperados), último quadro
  anotado (`preview.jpg`, atualizado pelos eventos SSE `frame`), Cancel; ao
  terminar abre a execução em `#/runs/<id>` (próxima seção); em
  falha mostra o erro (ex.: "no frames decoded … is source.start past the
  end of the video?") e a cauda do log.

Rotas: `GET /api/videos`, `GET /api/videos/roots`, `POST /api/videos`
(multipart: `file` e, para `.h264`, `meta`), `POST /api/videos/import`
(`{path}`), `GET /api/videos/{id}` e `GET /api/videos/{id}/{file|preview|thumb}`
(com `Range`). `POST /api/runs` recusa (422) um `video_id` que não está na
biblioteca ou cujo arquivo sumiu, e com `video_id` o `source.video` é sempre
o original do vídeo (o do cliente é sobrescrito).
Um `.mcap` (ROS 2) entra na biblioteca como `kind: "mcap"` (validado com
`probe_mcap`; sem tópico de imagem decodificável, corrompido ou truncado →
422; tamanho, taxa e miniatura vêm do primeiro tópico suportado), e
`GET /api/videos/{id}/topics` devolve os tópicos de imagem (resolução, taxa,
contagem, `CameraInfo` pareado, `has_distortion`, `supported` e
`unsupported_reason` — ex.: `16UC1`, `compressedDepth`); com esse `video_id`
o `POST /api/runs` define `source.mcap`, exige um `source.topic` suportado
entre os tópicos de imagem e recusa (422) `rectify.enabled`/`camera.calibration.from_source`
num tópico sem `CameraInfo` utilizável (task_0033).

### Interface web: execuções e comparação

A `task_0017` substitui a vista mínima pela tela **Runs & comparison**
(`#/runs/<id>`, ou `#/runs/<id>/vs/<id>` para comparar; ids na URL, nunca o
nome da pasta como caminho):

- **Histórico** à esquerda, com filtro: a execução selecionada fica em
  verde-água, a de comparação em violeta; o botão de comparar aparece ao
  passar o mouse sobre outra execução (ou use o menu "Compare with").
  Jobs que falharam antes de criar a pasta (`job-<id>`) mostram o erro e a
  cauda do log, sem player.
- **Vídeo anotado** (`video.mp4`), **indicadores** com delta assinado contra a
  execução de comparação (verde quando melhora, vermelho quando piora:
  *jitter* menor e vazão maior são melhores) e **diff do YAML efetivo**
  (caminhos pontilhados; `output.tag`/`out_root`, que o servidor sempre
  define, ficam escondidos atrás de um botão).
- **Distância ao longo do tempo** por trilha: esta execução suavizada sobre a
  medida bruta em cinza e a de comparação tracejada; passar o mouse mostra os
  valores, clicar (ou setas + Enter) salta o vídeo para aquele instante e a
  linha vertical acompanha o vídeo. Ao comparar, a trilha inicial é a mais
  longa cujo id também existe na outra execução com intervalo de tempo
  sobreposto (senão o id comum mais longo); ids são por execução e o mesmo id
  pode ser outro objeto, o que a nota sob o gráfico lembra.
- **Artefatos**: todos os arquivos da pasta (tamanho, tipo, abrir, baixar);
  JSON vira árvore recolhível, e o `metrics.json` aparece inteiro em
  **Metrics** — um sink ou seção nova aparece sem mudar a interface
  (`docs/web_extending.md`, item d).
- **Reopen in editor** carrega o YAML efetivo (`config.yaml`, senão o
  `config` do `metrics.json`) no editor; **Rerun** põe na fila a mesma
  configuração e o mesmo vídeo da biblioteca (sem `video_id`, só se o
  `source.video` estiver numa `--data-root`; senão 409 com a explicação).

Indicadores (`web/compare.py`, lidos do `detections.csv` por
`evaluation.run_data.read_columns` e do `metrics.json`): mediana de
`distance_m`; *jitter* = mediana, entre trilhas (`track_id`), da mediana de
|Δ`distance_m`| entre quadros processados consecutivos (passo =
`source.stride`; lacunas da trilha não contam); *jitter* suavizado (o mesmo
em `distance_smooth_m`); vazão (`end_to_end_fps`); quadros, caixas e trilhas.
Pastas antigas sem `metrics.json`, `config.yaml`, `track_id` ou colunas
suavizadas aparecem com `n/a` e uma nota. Rotas: `GET /api/runs/{id}/summary`,
`GET /api/compare?a=ID&b=ID` (deltas `b − a`), `GET
/api/runs/{id}/series?track=N` (≤ 1500 pontos, redução min/máx),
`GET /api/runs/{id}/artifacts`, `GET /api/runs/{id}/files/{caminho}`,
`POST /api/runs/{id}/reopen`, `POST /api/runs/{id}/rerun`,
`GET /api/runs/{id}/size` (`{bytes}`) e `DELETE /api/runs/{id}` (204; 404 se
não for uma execução listada; 409 enquanto a execução está na fila ou rodando).
Na vista de execuções, o botão de lixeira (na lista e no detalhe) abre um
diálogo que mostra o espaço liberado e, ao confirmar, apaga a pasta da
execução de forma permanente; uma entrada `job-<id>` apaga só o registro do job.

### Model weights

Every model declares what it needs (`requirements(options)`, task_0021), so a
fresh clone runs a sample YAML after one
`av-perception-lab --config configs/<x>.yaml --prepare`. Everything lives
under the **weights root** (`weights/` in the repo, git-ignored; override with
`AVPL_WEIGHTS_ROOT`); nothing is downloaded at run time, and a missing default
file is an error naming `--prepare`. The full reference (layers, asset kinds,
locations, states, the packages layer, how to add a model and how to resolve
dependency conflicts) is [`docs/model_environments.md`](model_environments.md).

#### From a fresh clone

Prerequisites: [uv](https://docs.astral.sh/uv/) (installs Python and the
packages), Node.js LTS with npm (to build the front-end once; see
[Interface web: front-end](#interface-web-front-end-e-editor-de-pipeline) for
an install without sudo), `git` (Prepare clones pinned code) and network
access to Hugging Face, GitHub and `repo.roboflow.com` while preparing.

1. Clone the repository and install the base Python stack **once** (the
   server never runs this command):

   ```bash
   uv sync --extra infer --extra web
   ```

2. Build the front-end once (Node, `cd web && npm ci && npm run build`; see
   [Interface web: front-end](#interface-web-front-end-e-editor-de-pipeline))
   and start the server:

   ```bash
   uv run --extra infer --extra web av-perception-lab-web --weights-root weights
   ```

3. Open a sample YAML from `configs/` in the editor (**Open YAML**); start
   with `configs/rfdetr_da3.yaml`, whose models are all downloadable. Models
   that are not ready get a yellow dashed outline with an alert icon.
4. Click **Prepare all** in the banner (or **Prepare model** in a model's
   inspector), check the list of downloads, sizes and licences in the
   confirmation, click **Download** and wait for the progress bars;
   **Process** unlocks when every model is ready. Configs that contain
   TwinLiteNet+ (for example `yolo_light_da3.yaml`,
   `rfdetr_sam2_lanes_motion_da3.yaml`, `rfdetr_lanes_scene_da3.yaml`) need one
   **manual download** (its `.pth` is on Google Drive, see below): Prepare
   fetches everything else, the TwinLiteNet+ row then shows "provide this
   file" with the command, and Process stays locked until the file is there.

Without the web interface, the same from the command line:

```bash
uv run --extra infer av-perception-lab --config configs/rfdetr_da3.yaml --prepare   # exit 0 = all ready
uv run --extra infer av-perception-lab --config configs/rfdetr_da3.yaml VIDEO.MOV
```

- **Weights root:** `--weights-root DIR` sets it for the web server, which
  passes it (absolute) to every run worker as `AVPL_WEIGHTS_ROOT`, so nothing
  else needs exporting; `AVPL_WEIGHTS_ROOT=DIR` sets it for the CLI (and for
  the server when `--weights-root` is absent). It must be a dedicated folder:
  `/`, the home folder and folders that contain the project, the active
  Python environment, the run outputs or the video library are refused.
- **Your own checkpoint under a release name:** a file at a default location
  whose size differs from the published one (for example a fine-tuned
  `weights/ultralytics/yolo26l.pt`) is never downloaded over: the model shows
  "exists with a different size at …; move or delete it, or point the config
  at another file".
- **Gated models** (today `facebook/sam3`, used by `sam3_tracker`): accept the
  licence on the model's Hugging Face page and make the credentials visible
  to the process that downloads (`HF_TOKEN` exported before starting the
  server or the CLI, or `hf auth login`); there is no token field in the UI.
  Without them Prepare fails with "gated - accept the licence at … and set
  HF_TOKEN".
- **Manual step, TwinLiteNet+** (`semantic` type `twinlitenet`): Prepare
  clones the pinned repository, but `pretrained/<variant>.pth` is on the
  upstream Google Drive (not a source in v1). After the first Prepare, save it
  in the clone; for `variant: small` the status message carries the exact
  command:

  ```bash
  mkdir -p weights/_code/TwinLiteNetPlus/pretrained
  curl -fL -o weights/_code/TwinLiteNetPlus/pretrained/small.pth "https://drive.usercontent.google.com/download?id=1N7jNsa8P1dM-UevqPRsx6heUpN4L00FJ&export=download&confirm=t"
  ```

  For the other variants the message links the upstream Drive folder.
- **Unknown sources** are prepared by the CLI (`--prepare`), not by the web
  server: a Hugging Face repo typed in a YAML, and a known file or repo at a
  location of the YAML's choosing (a `weights` path outside
  `weights/ultralytics/`, an `hf_cache` other than `weights/hf`).
- **Machines with older weights** (the defaults moved in task_0021: RF-DETR
  from `~/.roboflow`, DA3 and YOLO from machine paths): run `--prepare` once
  per sample config, or point the YAML options at the existing files.

| Model | Default location | Source (`--prepare`) |
|---|---|---|
| DA3 (`depth`, `camera.estimate`), SAM family, SegFormer | `weights/hf` (Hugging Face cache layout; an existing cache keeps working) | Hugging Face, weights file + `*.json` |
| RF-DETR (`variant`) | `weights/rfdetr/<library file name>`, e.g. `rf-detr-large-2026.pth` | `repo.roboflow.com` (library URL table), Apache-2.0 |
| YOLO / Ultralytics SAM (`weights`) | a bare file name in `weights` resolves under `weights/ultralytics/`; a path with a directory part keeps its usual meaning (relative to the working directory) | `ultralytics/assets` v8.4.0 for published names (YOLO26 COCO and Objects365, SAM, SAM 2.x, MobileSAM; AGPL-3.0 YOLO weights); any other file is *user-provided* |
| TwinLiteNet+ (`repo`) | `weights/_code/TwinLiteNetPlus` (pinned clone @ `90f1b86`) | git clone; `pretrained/<variant>.pth` is *user-provided* (upstream Google Drive) |

A run (without `--prepare`) whose declared models are not all ready stops
before loading anything, with exit code 2 and a list of what is missing
(paths relative to the weights root) naming `--prepare` (or, when only Python
packages are missing, `uv sync --extra infer`: `--prepare` does not install
packages).

#### Python packages (task_0024)

The packages a model imports come from a `pyproject.toml` extra. The base
stack (the `infer` extra, plus `web` for the server) is a one-time,
documented command that neither the server nor `--prepare` ever runs:

```bash
uv sync --extra infer --extra web
```

A future model with its own small extra (e.g. `model-xyz`) is installed in
three steps: the installed extras are detected (`importlib.metadata` and
`find_spec`, nothing imported); `uv sync --dry-run --locked --inexact` runs
with `--extra` for every detected extra plus the new one (never a bare
`uv sync`, which would remove the extras it does not name); only a clean
dry-run is followed by the same command without `--dry-run`, after an
explicit confirmation (`--prepare --install-packages`, or **Install
packages** in the web inspector). A dry-run that fails, or that would remove
or replace an installed package, leaves the environment untouched and puts
the model in state `conflict` with the resolver's reason; the ways to resolve
it are described in `docs/model_environments.md`. After an install, new runs
use the packages at once (each run starts a fresh worker process); restart
the web server only if it should use an upgraded package it had already
imported. `AVPL_UV` names the `uv` executable (default: `uv` on the
`PATH`).

An explicit path in the YAML overrides the default and behaves as before
(`hf_cache` relative to the working directory; an absolute or `~` YOLO path;
`repo: ~/dev/TwinLiteNetPlus`; RF-DETR `weights:` a checkpoint such as an
existing `~/.roboflow/models/rf-detr-large-2026.pth`). A location outside the
weights root is not tracked by `--prepare` (the adapter checks it when it
loads); existing `~/.roboflow`, `~/dev/...` and `weights/hf` files are never
moved or deleted.
Sources, sizes and licences of every declared file: `docs/task_0021.md`.
The GPU smoke and the task_0001 regression skip an example config until its
models are ready under the weights root: on a GPU host run `--prepare` once per
config (the adapter-contract tests keep their `AVPL_*` paths).

### Web interface: model status and Prepare

The web server exposes the same readiness and Prepare as the CLI (task_0022)
and the editor shows it on the pipeline (task_0023). Start it with the weights
root it should check and fill:

```bash
uv run --extra infer --extra web av-perception-lab-web --weights-root weights   # default: $AVPL_WEIGHTS_ROOT, else weights/
```

- `POST /api/models/status` with `{config}` returns one row per model the
  config would load (`stage`, `index`, `type`, `name`, `state`, `layers`,
  `missing`, `size_bytes`, `message`, plus `source`, `license`, `target`
  relative to the weights root, `reason` and `preparable`) and
  `total_size_bytes` still to download. `GET /api/models/prepare` lists the
  recent Prepare jobs (a reloaded page re-attaches to the unfinished ones). Nothing is loaded and torch is not imported.
- `POST /api/models/prepare` with `{config, models: "all"}` (or a list of
  `{stage, index}` from the status rows) queues a download; follow it with
  `GET /api/models/prepare/{id}/events` (SSE, per-model progress) and cancel
  it with `DELETE /api/models/prepare/{id}`. One Prepare runs at a time and
  never blocks the processing queue; a cancelled or failed model stays not
  ready (a partial file is never counted).
- A run whose models are not all ready is refused with 409
  `{models_not_ready: [...]}`.
- Rows whose Python packages are missing carry `packages` (`action` base,
  install, conflict or invalid; `extra`; `command`; `message`). A failed
  dry-run makes the row `conflict`. `POST /api/models/prepare` with
  `packages: true` and named models installs a per-model extra as a job of
  `kind: "packages"` (fresh dry-run first; `uv` output as `output` SSE
  events; cancel kills `uv`); then the status body has
  `restart_required: true`. The base stack is never installed (422), nor an
  extra whose server-side plan is not a clean dry-run (422 with the reason);
  the job runs only if its fresh dry-run gives the confirmed command. While a
  run is queued or running an install answers 409, and while an install is
  queued or running `POST /api/runs` answers 409. A row whose dry-run could
  not run (uv not found, timeout) is `failed` with the reason and checked
  again next time (set `AVPL_UV` when uv is not on the `PATH`).
- From the web, Prepare only downloads sources the adapters vouch for: a
  Hugging Face repo typed in `depth.model`, `refiners[].model` or
  `semantic[].model` that the project does not ship reads "unknown source:
  prepare it from the CLI (--prepare)" (422 when named; `models: "all"`
  prepares the rest and lists it under `skipped`). Run
  `av-perception-lab --config X.yaml --prepare` on the server for it.
- The client only names models: URLs and target folders come from the
  adapters' code, downloads land only under the weights root, and messages
  show `<weights root>` instead of the server's absolute path. The weights
  root may not overlap `--library-root` or `--out-root`; with `--asset-root`
  it counts as an asset folder (a bare YOLO `weights: yolo26l.pt` is checked
  at `<weights root>/ultralytics/yolo26l.pt`, where the adapter opens it).

In the editor (task_0023):

- Every card whose model is not ready gets a **yellow dashed outline and an
  alert icon at its top-right corner** (hover or focus it for the reason); a
  model being prepared shows a progress bar on its card; a failed Prepare is
  red dashed with its reason; a dependency conflict (or any state the
  interface does not know yet) uses the yellow warning style with the
  server's message. Ready cards stay solid.
- A banner above the lanes reads "N models need preparing · X to download"
  (N and X: what **Prepare all** would download, with their licences) and
  lists the models; while a Prepare runs
  it shows the overall progress and a **Cancel** button. You can keep editing
  meanwhile.
- The inspector of a model has a **Weights & requirements** section: layers
  (assets, code, packages), what is missing, size, where the files go, and
  **Prepare model** (or **Retry** after a failure). A model whose source the
  web may not download shows the command to run on the server instead,
  `av-perception-lab --config <file> --prepare`, with a copy button. To use
  your own file, set the module's path option (e.g. `weights`) in Options.
- Python packages (task_0024): a missing base stack shows
  `uv sync --extra infer --extra web` with a copy button (no install button);
  a model's own extra with a clean dry-run gets **Install packages**, which
  opens a confirmation dialog stating the exact command; while it runs the
  inspector shows the `uv` output, and afterwards a note says new runs use
  the packages (a server restart is only for an upgraded package the server
  had already imported). A conflict shows the resolver's reason and points to
  `docs/model_environments.md`.
- **Process** stays disabled ("N models are not ready — Prepare them first.")
  until every model is ready; if the server still refuses the run (409, e.g.
  a file removed meanwhile), a dialog lists the models and offers Prepare all.

### Pesos dos refinadores e do SegFormer

Nada é baixado em tempo de execução (`local_files_only=True`); um peso
ausente gera `ConfigError` com o comando exato. Dentro do repo
(o filtro evita as cópias `.pt`/`.bin` duplicadas):

```bash
uv run --extra infer python -c "from huggingface_hub import snapshot_download as d; [d(r, cache_dir='weights/hf', allow_patterns=['*.json', 'model.safetensors']) for r in ('facebook/sam2.1-hiera-large', 'yonigozlan/EdgeTAM-hf', 'syscv-community/sam-hq-vit-huge')]; d('nvidia/segformer-b5-finetuned-cityscapes-1024-1024', cache_dir='weights/hf', allow_patterns=['*.json', 'pytorch_model.bin'])"
# opcionais: d('facebook/sam-vit-base', …) / d('facebook/sam3', …) com o mesmo filtro (sam3: HF_TOKEN com acesso aprovado)
curl -fL -o weights/ultralytics/mobile_sam.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt   # or --prepare
```

| Peso | Bytes | Uso |
|---|---|---|
| `facebook/sam2.1-hiera-large` | 897 897 416 | `sam2` (exemplos) |
| `yonigozlan/EdgeTAM-hf` | 55 912 824 | `edgetam` |
| `syscv-community/sam-hq-vit-huge` (ou `-vit-base`) | 2 570 816 384 (379 266 384) | `sam_hq` |
| `nvidia/segformer-b5-finetuned-cityscapes-1024-1024` (`pytorch_model.bin`) | 338 896 453 | `segformer` (licença `other`, NVIDIA: ler os termos antes de uso não acadêmico) |
| `facebook/sam-vit-base` (opcional) | 374 979 480 | `sam` |
| `mobile_sam.pt` (opcional) | 40 728 226 | `ultralytics_sam` |
| `facebook/sam3` (opcional, *gated*) | 3 439 938 512 | `sam3_tracker` |
| `nvidia/segformer-b0-finetuned-cityscapes-640-1280` (`pytorch_model.bin`) | 14 957 601 | `segformer` do exemplo leve (licença `other`) |
| `yolo26s-seg.pt` (ultralytics/assets v8.4.0) | 23 467 933 | detector do exemplo leve |
| TwinLiteNet+ `small.pth` (Google Drive do upstream) | 594 777 | `twinlitenet` `variant: small` |
| `depth-anything/DA3METRIC-LARGE` / `DA3-LARGE-1.1` (`model.safetensors`) | 1 336 734 448 / 1 643 843 860 | DA3 no notebook (`weights/hf`) |

As quatro últimas linhas são os pesos do exemplo leve, baixados pelos
comandos de [Execução no notebook](#execução-no-notebook-gpu-de-4-gb). Since
task_0021 DA3 also reads with `local_files_only=True`: a missing checkpoint is
an error naming `--prepare`.

---

## Saídas

Cada execução cria `<output.out_root>/<AAAAmmdd_HHMMSS>_<vídeo>_<detectores>_<profundidade>[_tag]/`:

| Arquivo | Conteúdo |
|---|---|
| `video.mp4` | Vídeo anotado (H.264; taxa variável com entrada MCAP); com mais de um detector e `concat`, cor e contagem do HUD por modelo de origem; com fusão, cor pelo rótulo canônico, `×N` no rótulo quando N detectores viram o objeto e linhas de HUD com o agregador e a concordância (`agree <m> · only <a> <x> · only <b> <y>`); máscaras translúcidas na cor da instância, no quadro e no painel de profundidade, e `· region <política> · erode <k>` na linha de distância do HUD nos quadros em que alguma instância usou a máscara; camadas semânticas translúcidas (cor pelo nome da classe, por baixo das máscaras e caixas, só no painel do quadro, na ordem de `semantic`) e linhas de HUD `refine <nome> · <preenchidas>/<alvos>` e `semantic <nome> · excl <classes>`; com rastreamento, rótulo `#<id> classe conf · D m · ±v m/s` (distância suavizada com `show_smoothed`, velocidade com `show_speed`) e última linha de HUD `track iou · IoU ≥ 0.3 · age 0.5 s · kalman q 2 · ids <trilhas abertas>`; com `motion`, `· v <m/s>` no fim do rótulo, trajetória prevista a partir da base da caixa (opacidade decrescente, elipses de `n_sigma` no chão em ~3 pontos), painel BEV opcional e `· motion <tipos> · <horizonte> s` na linha de rastreamento; áudio copiado quando o vídeo inteiro é processado sem `stride` |
| `detections.csv` | Uma linha por caixa: quadro, tempo, classe, confiança, caixa, região amostrada no mapa, `n_valid`/`n_inliers`/`inlier_frac`, média, mediana, desvio, variância, p10–p90, mín/máx, `distance_m` (vazio = n/a) e, no fim, `source` (detector de origem), `raw_label` (nome no vocabulário do modelo), `n_sources` (nº de detectores que viram o objeto; 1 sem fusão), `sources`, `source_scores` e `raw_labels` (um item por detector, o seu melhor membro, separados por `\|`, em score decrescente), e por fim `used_mask` (1 = estatísticas sobre a máscara), `mask_px` (pixels do mapa da máscara erodida; `mask_px / n_box_px` = ocupação da caixa), `mask_erode` (erosão aplicada) e `box_distance_m` (só com `region: both`, distância pela caixa da instância que usou máscara) e, por último, `mask_source` (componente que produziu a máscara — detector, refinador ou membro da fusão dono da máscara —; vazio = sem máscara) e `mask_score` (qualidade prevista pelo refinador, 4 casas; vazio para máscaras de detector) e, no fim, `track_id`, `distance_smooth_m`, `distance_smooth_std_m`, `rel_speed_mps` e `rel_speed_std_mps` (vazias sem rastreamento/suavização; desvios e velocidade só com `kalman`; `distance_m` segue sendo a distância bruta) e, por fim, as 16 colunas de movimento (`motion`; vazias sem o estágio): `motion_model`, `motion_x_m`, `motion_z_m`, `motion_x_std_m`, `motion_z_std_m`, `motion_vx_mps`, `motion_vz_mps`, `motion_speed_mps`, `motion_speed_std_mps`, `motion_heading_deg`, `motion_heading_std_deg` (vazios abaixo de `min_speed_for_heading`), `motion_yaw_rate_dps`, `motion_yaw_rate_std_dps` (só com `ctrv`), `motion_abs_speed_mps` (só com ego-motion), `motion_nis` (vazio sem medida no quadro) e `motion_n_updates`. Com máscara, as colunas de estatística descrevem a máscara e `n_box_px` é a janela da caixa sem encolhimento. A semântica não acrescenta colunas |
| `trajectories.jsonl` | Só com `motion`: uma linha JSON por quadro × instância com estimativa — `frame`, `t`, `track_id`, `label`, `model`, `x`, `z`, `vx`, `vz`, `y` e a trajetória prevista (`t_ahead`, `xz`, `cov` 2×2 por ponto), no referencial da câmera do quadro, 4 casas |
| `metrics.json` | `components` (descrição de cada detector, da profundidade, do agregador, dos `refiners` e dos `semantic`), `pipeline` (política: `exclude_sky`, `agreement`, `region`, `mask_erode`, `exclude_semantic` como `{nome: [classes]}`, `smoothing`, `motion` — modelos, classes, medida, previsão e `ego_motion` com `clamped_frames`), `components.tracker` (null sem rastreamento), tempos por etapa (`det`, `det.<nome>`, `agree`, `aggregate`, `refine`/`refine.<nome>`, `track`, `semantic`/`semantic.<nome>`, `smooth` e `motion` quando existem, `depth`, `dist`, `render`), `tracks` (só com rastreador: `n_tracks`, `short_tracks` — trilhas com menos de `min_duration_s` = 1 s, fora da lista — e `per_track` com `id`, `label` (moda), `labels`, `label_changes`, `sources`, `first_frame`/`last_frame`/`first_t`, `duration_s`, `n_frames`, `coverage`, `gaps`, `max_gap_s`, `distance_m` {min, median, max}, `jitter_raw_m`/`jitter_smooth_m` (desvio da diferença quadro a quadro), `rel_speed_mps` {p5, median, p95} e, só com `motion`, `motion` {`model`, `speed_mps` {p5, median, p95}, `nis_mean`, `restarts`} ou `null` sem estimativa), `mask_sources` (instâncias com máscara por produtor), `refined` (por refinador: `targets` e `filled`), `frame_intrinsics` (K do quadro entregue ao modelo de profundidade, `null` quando ele prevê K por quadro), `camera` (`{}` sem `camera.calibration`; com ela: `calibration` (caminho, tamanho, K contínua, distorção, RMS, desvios, câmera), `scaled_from`, `lens_check`, `undistort` e, com `cross_check`, `estimated_focal_frame_px`, `estimated_source` e `estimated_over_calibrated`), `video.transform` (correção de distorção aplicada aos quadros ou `null`), `semantic` (só com segmentador: por mapa `exclude`, `class_frac_mean`, `excluded_frac_mean`; no topo, `excluded_frac_mean` da união das exclusões), VRAM, focal (px do quadro e intrínsecos do mapa) e origem, parâmetros de distância, contagens (`detections_total`, `detections_by_source`, `detections_by_n_sources`), `distance_regions` (`{"mask": n, "box": n}`), `region_comparison` (só com `region: both`: diferença máscara − caixa com `n`, `median_signed_m`, `median_abs_m`, `p90_abs_m`, `median_rel`, total e `by_label`), `agreement` (com ≥ 2 detectores: por detector `instances`/`duplicates`; por par `matched`/`only_a`/`only_b`/`mean_iou`; séries por quadro em `per_frame`), versões (inclui `transformers` e `timm`), commit e `config` (YAML resolvido) |
| `depth.npz` | Só com `output.save_depth: true`: mapas em metros (`float16`), índices dos quadros, intrínsecos por quadro (`fx, fy, cx, cy` na resolução do mapa) e geometria (`geometry` + `geometry_fields`); com segmentadores, por nome, `semantic_labels_<nome>` (`(F, map_h, map_w)`, `uint8` até 256 classes, rótulos amostrados na grade do mapa como na elegibilidade) e `semantic_classes_<nome>` |

Prévia leve para trazer ao laptop:
`ffmpeg -i video.mp4 -vf scale=-2:720 -crf 26 preview_720p.mp4`.

---

## Como a distância é calculada

1. O DA3 gera um mapa de profundidade em resolução própria (ex.: 504×280 para
   1920×1080); cada caixa detectada é levada ao mapa com escala separada em
   x/y (e deslocamento nos métodos `*_crop`).
2. A região amostrada depende de `distance.region`. Com máscara de instância
   (`mask`/`both`), a máscara (resolução do quadro) é reamostrada por vizinho
   mais próximo no centro de cada pixel do mapa dentro da caixa **sem**
   encolhimento (pixels da máscara fora da caixa são ignorados) e erodida
   `distance.mask_erode` pixels do mapa (elemento 3×3; a erosão recua enquanto
   deixaria menos de `min_valid_px` pixels, para objetos finos). Sem máscara
   (ou com `box`), a caixa é encolhida (`distance.box_shrink`). Pixels
   inválidos, além do limite (`distance.max_range`), de céu
   (`distance.exclude_sky`) ou de classe semântica excluída
   (`distance.exclude_semantic`, rótulos reamostrados para a grade do mapa,
   união entre os mapas) são descartados nos dois casos. Máscaras de
   refinadores (SAM) são máscaras de instância como as nativas.
3. Outliers (fundo dentro da caixa) são rejeitados (`distance.outlier`).
4. A distância é a estatística escolhida (`distance.stat`) dos pixels
   restantes; com menos de `distance.min_valid_px` pixels, o rótulo mostra `n/a`.

O `DA3METRIC-LARGE` só fornece metros com a focal da câmera: use uma
calibração (`camera.calibration`, K completa com `fx ≠ fy` e ponto principal
fora do centro), informe `camera.focal_px` ou deixe que um modelo any-view a
estime (`camera.estimate`, mediana de `frames` quadros). Os intrínsecos
usam coordenadas contínuas (centro do pixel `i` em `i + 0,5`); o `cx`/`cy`
do OpenCV é 0,5 menor e é convertido na leitura da calibração. `distance_type: euclidean`
converte a profundidade ao longo do eixo óptico em distância em linha reta.

---

## Melhorias Futuras (TODO)

- [x] **Padrão de projeto** — esqueleto do repositório, AGENTS/CLAUDE/DESCRIPTION/CHANGELOG.
- [x] **Pipeline RF-DETR + DA3** (`task_0001`) — distância por caixa, vídeo anotado, CSV e métricas.
- [x] **Pipeline modular** (`task_0002`) — componentes abstratos, configuração YAML, detector YOLO26 e vários detectores no mesmo passo.
- [x] **Fusão entre detectores** (`task_0003`) — harmonização de rótulos, NMS/WBF entre modelos e métricas de concordância.
- [x] **Segmentação de instâncias** (`task_0004`) — YOLO26-seg e RF-DETR-Seg; distância pela máscara (`distance.region`, `mask_erode`), máscaras no vídeo e colunas de máscara no CSV.
- [x] **Refinadores de máscara e segmentação semântica** (`task_0005`) — família SAM (SAM2, EdgeTAM, SAM-HQ, SAM, SAM3) e MobileSAM via hierarquia `BoxPromptRefiner`; vários segmentadores semânticos nomeados (SegFormer, TwinLiteNet+) com exclusão de classes por `<nome>.<classe>`. Validação no servidor GPU pendente (testes `-m gpu`, pesos e execuções A–E).
- [x] **Rastreamento e suavização temporal** (`task_0006`) — rastreador IoU com velocidade constante (`track_id`), suavização por trilha (Kalman 1D, EMA, mediana), velocidade relativa e resumo por trilha no `metrics.json`. Validação no servidor GPU pendente (regressão `-m gpu` e execuções nos vídeos reais).
- [x] **Calibração e avaliação** (`task_0007`) — calibração por tabuleiro/ChArUco, `camera.calibration` (K reescalada, conferência da lente, correção de distorção, comparação com a focal estimada), protocolo de verdade de campo e relatório de erro contra referências. Pendentes: calibração real do iPhone 14 Pro Max, coleta de referências de campo e validação no servidor GPU.
- [x] **Smoke test ponta a ponta** (`task_0008`) — smoke local sem GPU (cena sintética, dublês, matriz de configurações em pares, calibrar → rodar → avaliar), smoke `gpu` com os modelos reais e `utils/gpu_pending.sh` com todas as pendências de GPU em ordem. Execução no servidor GPU pendente.
- [x] **Movimento por trilha e previsão de trajetória** (`task_0009`) — estágio `motion`: posição no plano, velocidade, rumo e taxa de giro com incertezas, modelos por classe (`cv`, `ctrv`), trajetória prevista com covariância no vídeo, no painel BEV e em `trajectories.jsonl`, ego-motion opcional por CSV. Validado no servidor GPU (`-m gpu` e passo `20-motion` do `gpu_pending.sh`); conferência visual dos vídeos pendente.
- [x] **Model requirements and status check** (`task_0019`) — `core/requirements.py` declares what a model needs (weights, pinned code clones, packages) and `check()` reports readiness without importing torch; the provisioning engine, adapter migration and UI follow in `task_0020`–`task_0025`. New documentation is written in English.
- [x] **Provisioning engine and `--prepare`** (`task_0020`) — `core/provision/` obtains a model's declared weights and code with registered sources (Hugging Face, URL with resume and sha256, pinned git clone; a new source kind is one class + `register_source`; Google Drive not yet), atomically, cancellably and idempotently; `av-perception-lab --config X.yaml --prepare` prepares every model of a YAML. Adapters declare their requirements in `task_0021`.
- [x] **Web API for model status and Prepare** (`task_0022`) — `POST /api/models/status`, a prepare queue with SSE progress and cancel (independent from the GPU queue), 409 for runs with models not ready, `--weights-root`.
- [x] **Model readiness in the editor** (`task_0023`) — card states (not ready: yellow dashed + alert icon; preparing: progress bar; failed: red dashed), "N models need preparing · X GB" banner with Prepare all, inspector "Weights & requirements" with Prepare model / Retry, Process gated until every model is ready, 409 dialog. Real download through the UI on a networked host pending.
- [x] **Packages layer** (`task_0024`) — per-model extras detected without imports, `uv sync --dry-run --locked --inexact` before any install, state `conflict` with the resolver's reason, `--prepare --install-packages` and **Install packages** in the inspector; the base stack (`uv sync --extra infer --extra web`) is never installed by the server.
- [x] **Model readiness documentation** (`task_0025`) — [`docs/model_environments.md`](model_environments.md): the model recipe with a worked example, layers and asset kinds, locations and overrides, states, CLI and web flows, packages layer and the five ways to resolve a dependency conflict.
- [ ] **Dependency-conflict resolution beyond detection** (future; v1 only detects conflicts) — paths 2–5 of [`docs/model_environments.md`](model_environments.md#8-dependency-conflicts-the-five-resolution-paths): (2) mutually exclusive per-model extras via `[tool.uv] conflicts` with a confirmed switch; (3) one environment per profile, used by the web worker subprocess; (4) the conflicting model served as its own process behind a thin adapter; (5) manual pinning (constraints/overrides) or forking the upstream.
- [ ] **Ego-motion pelo vídeo** (`task_0011`, futura) — odometria visual RGB-D com a profundidade do DA3 no fundo estático (ou só o *yaw* pela homografia do fundo distante) como provedor `ego_motion`.
- [x] **Configuração leve** (`task_0010`) — `yolo_light_da3.yaml` para GPU de 4 GB (YOLO26s-seg, DA3METRIC-L, rastreamento, TwinLiteNet+ small, SegFormer-B0). Medição de VRAM no notebook pendente.
- [x] **MCAP input, CameraInfo intrinsics and rectification** (`task_0033`) — `source.mcap` / `source.topic` read a ROS 2 (CDR) `Image` or `CompressedImage` topic directly (`io/mcap_reader.py`, `McapSource`), with the message times as `Frame.t`; `camera.calibration.from_source` takes K from the topic's `CameraInfo`; the optional `rectify` stage undistorts with it; the rendered video keeps the real timestamps (`TimestampedVideoWriter`); the web interface takes a dropped `.mcap`, lists its topics and offers the CameraInfo and Rectification options; evaluation loads MCAP runs. See [MCAP input](#entrada-mcap-ros-2).
- [ ] **Stereo input and stereo depth** (future, asked 2026-11-04) — load a ZED (Stereolabs) SVO and run a stereo depth model such as Fast Foundation Stereo; when a stereo pair is provided, run it and DA3 in the same pipeline (stereo as a second depth estimator, also to cross-check DA3). `Frame` would get an optional right image; the depth stage would choose DA3, a stereo model, or both.
- [ ] **Several camera streams and stereo pairs from one MCAP** (future) — today one topic per run. Select several image topics (and a left/right pair) from the same file; needs the stereo item above for the pair.
- [ ] **Ego-motion from an MCAP** (future) — `/tf`, `/tf_static`, odometry and IMU topics as an `ego_motion` provider for the motion stage, next to the visual odometry item above and the CSV provider.
- [ ] **Ground truth from an MCAP** (future) — LiDAR or depth topics and simulator 3D boxes as evaluation references, instead of the manual field references.
- [ ] **Write results back to an MCAP** (future) — annotated images and detections as topics a viewer such as Foxglove can show; Foxglove protobuf/JSON schemas as inputs through the same reader.
- [ ] **Evaluate MCAP runs with `source.start > 0`** (`task_0033` follow-up) — `load_run` refuses them today because the frame indices and times would be wrong. Record the first selected index and the time offset in `McapSource.describe()` and use them in the evaluation.
- [x] **Optional YOLO extra** (`task_0034`, asked 2026-11-01) — moved `ultralytics` (AGPL-3.0) out of the `infer` extra into an optional one, as the README says is planned.
- [ ] **Retake the web interface screenshot** (`docs/assets/ui-editor.png`; needs a person) — it predates the MCAP input: no Rectification lane, no topic picker, and the README alt text says the stages run "from source to semantic segmentation". Run `utils/run_web.sh --gpu`, load `configs/rfdetr_sam2_lanes_motion_da3.yaml`, run **Prepare all** (about 3.1 GB), drop an `.mcap` whose topic has a `CameraInfo` with distortion, select a three-second segment, open the Kalman smoothing panel, and overwrite the image at the same size. Update the alt text in `README.md`. Optionally add `docs/assets/ui-mcap.png` for the topic picker.
- [ ] **Browser pass over the MCAP interface** (`task_0033` follow-up) — nobody has looked at the drop zone, the topic picker and the Rectification toggle in a real browser yet.
- [ ] **MCAP interface polish** (`task_0033` follow-up) — show a notice when turning on the CameraInfo option or Rectification removes `camera.focal_px`, a calibration file or a custom `alpha` (or keep `rectify.enabled: false` so `alpha` survives); re-clamp the segment when the topic changes; make `useMcapBinding` independent of load order; show the error when the topic list fails to load; say "recording" instead of "video" where the input is an MCAP; use the message count for the frame estimate.
- [ ] **MCAP reader robustness** (`task_0033` follow-up) — a topic with one message has fps 0 and duration 0; the `max_frames` drop warning gives only a lower-bound percentage; the unindexed-file docstring says "one pass" but the topic is read fully; a raw `Image` topic above the pixel cap is listed as supported and only fails at decode.
- [ ] **Odd frame sizes** (`task_0033` follow-up) — `VideoWriter` and `TimestampedVideoWriter` refuse odd widths and heights (libx264 limit); pad the frame in the renderer.
- [ ] **Extensibility guard without the allowlist** (`task_0033` follow-up) — read the `rectify` and `camera` keys from the catalog or stage registry and drop the exception for `mcapCamera.ts` and `RectifyLane.tsx` in `tests/web/test_extensibility.py`.
- [ ] **Split `web/library.py`** (`task_0033` follow-up) — move the MCAP entry logic (about 120 lines) to a `web/mcap_entry.py` and add one `source_path(config)` helper for the repeated `source.get("video") or source.get("mcap")` in `jobs.py`, `history.py` and `app.py`.
- [ ] **Re-measure the README numbers** — the frames per second (about 7, and 2 to 3) and the download sizes (3.1 GB, 136 MB, 1.3 GB, 1.6 GB) were not re-checked in the 2026-11-05 README audit.
