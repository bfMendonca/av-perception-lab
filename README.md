<p align="center">
  <h1 align="center">det-depth-distance</h1>
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
- [Uso da CLI](#uso-da-cli)
- [Configuração YAML](#configuração-yaml)
- [Refinadores de máscara](#refinadores-de-máscara)
- [Segmentação semântica](#segmentação-semântica)
- [Rastreamento e suavização](#rastreamento-e-suavização)
- [Pesos dos refinadores e do SegFormer](#pesos-dos-refinadores-e-do-segformer)
- [Saídas](#saídas)
- [Como a distância é calculada](#como-a-distância-é-calculada)
- [Sincronização com o servidor GPU](#sincronização-com-o-servidor-gpu)
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
suavizada no tempo (com estimativa da velocidade relativa).

O foco é **qualidade da estimativa**, não tempo real: os modelos rodam
em PyTorch, quadro a quadro. A inferência é feita no servidor GPU `gpu-host`;
o desenvolvimento e os testes unitários rodam localmente.

---

## Funcionalidades

- **Detectar** objetos quadro a quadro com RF-DETR (`nano`, `small`, `medium`, `large`, ou checkpoint próprio) e/ou Ultralytics YOLO26 (pesos COCO ou Objects365), com filtro de classes por preset do vocabulário (`driving` para COCO e Objects365) validado contra as classes do modelo.
- **Segmentar** instâncias com RF-DETR Seg (`seg-nano` … `seg-2xlarge`, COCO) ou YOLO26-seg (`yolo26*-seg.pt` COCO, `yolo26*-objv1-seg.pt`, vocabulário Objects365 esperado e ainda não conferido no servidor GPU): cada instância ganha uma máscara na resolução do quadro.
- **Combinar** vários detectores no mesmo passo (`aggregation: concat`): todas as caixas são mantidas e marcadas com o modelo de origem (`source`), no CSV e por cor no vídeo.
- **Harmonizar** rótulos entre vocabulários com mapas por detector (`labels: configs/labels/*.yaml`, ex. COCO `car`/`truck`/`bus` e Objects365 `suv`/`van`/`pickup truck` → `vehicle`), conferidos contra as classes do modelo na carga.
- **Fundir** as detecções do mesmo objeto (`aggregation: nms` ou `wbf`), entre detectores e também duplicatas de um mesmo detector, preservando os membros de cada instância fundida (detector, rótulo original, score original) e com peso fixo opcional por detector.
- **Medir a concordância** entre detectores (`agreement`): por par, quantos objetos casam por IoU e quantos são exclusivos de cada um, por quadro e no total do `metrics.json`.
- **Refinar** máscaras a partir das caixas de qualquer detector (`refiners`): SAM 2.1 (`sam2`, padrão dos exemplos), EdgeTAM, SAM-HQ, SAM, SAM 3 (rastreador; *gated*) via `transformers` e MobileSAM/SAM via Ultralytics (`ultralytics_sam`), todos sobre a mesma base `BoxPromptRefiner` (um *embedding* por quadro, caixas em lote, máscara de maior IoU previsto); por padrão só instâncias sem máscara (`targets: missing`), e uma máscara refinada vazia nunca substitui a existente.
- **Segmentar semanticamente** cada quadro com um ou mais segmentadores nomeados (`semantic`: SegFormer Cityscapes, TwinLiteNet+ área trafegável/faixas), sem fundir os mapas, e excluir classes do cálculo de distância por `distance.exclude_semantic: [<nome>.<classe>, …]` (união das exclusões).
- **Rastrear** objetos entre quadros (`tracking: {type: iou}`): associação gulosa por IoU com predição da caixa por velocidade constante (em px/s, tolera `stride` até 5 a 30 fps), agnóstica de classe por padrão (detectores alternam `car`/`truck` no mesmo objeto) e por detector com `concat`; `track_id` no CSV e `#id` no rótulo do vídeo; numpy puro, sem dependência nova.
- **Suavizar** a distância de cada trilha (`smoothing`): Kalman 1D com velocidade (padrão; estima também a velocidade relativa em m/s e as incertezas), média móvel exponencial ou mediana causal, com parâmetros em segundos; a distância bruta continua no CSV.
- **Estimar** profundidade métrica com Depth Anything 3 (`da3metric-large` com focal informada ou estimada; modelos NESTED já em metros).
- **Calcular** a distância de cada objeto com estatísticas robustas sobre a **máscara** da instância (quando o modelo a fornece; reamostrada para o mapa e erodida) ou sobre a caixa: encolhimento da caixa, exclusão de céu, rejeição de outliers (`iqr`, `mad`, `sigma`) e estatística configurável (`median`, `mean`, `trimmed`, `mode`, `pXX`), em profundidade `z` ou distância euclidiana; `distance.region: both` registra as duas distâncias para comparação.
- **Renderizar** vídeo anotado (rótulo `[#id] classe conf · D m ± σ [· v m/s]`, camadas semânticas e máscara translúcida por instância, HUD com tempos) e, opcionalmente, o mapa de profundidade lado a lado com legenda em metros.
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
| Detecção | `ultralytics` (YOLO26; licença AGPL-3.0) | 8.4.175 |
| Profundidade | `depth-anything-3` (ByteDance Seed) | git `3d835ec` |
| Refinadores SAM e SegFormer | `transformers` (Hugging Face) | `>=5.1,<6` (lock: 5.19.0) |
| *Backbone* do EdgeTAM | `timm` | `>=1.0.17` (lock: 1.0.30) |
| Área trafegável e faixas | TwinLiteNet+ (clone externo `~/dev/TwinLiteNetPlus`, MIT; não é pacote) | `90f1b86` |
| Atenção eficiente (DA3) | xformers | 0.0.32.post2 |
| Vídeo | ffmpeg (sistema), OpenCV, Pillow | — |
| Configuração | PyYAML | ≥ 6 (ambiente leve) |

As dependências pesadas ficam no extra `infer`, instalado apenas no host de
inferência; localmente basta o ambiente leve (numpy, OpenCV, Pillow, PyYAML,
pytest, ruff). Os frameworks são importados só dentro do `_load` de cada
adaptador, então `--help`, `--list-models` e a validação da configuração
rodam sem torch.

---

## Estrutura do Projeto

```
det-depth-distance/
├── pyproject.toml           # manifesto uv; extra "infer" com a stack GPU
├── uv.lock                  # lock versionado
├── configs/                 # exemplos de YAML: rfdetr_da3, yolo_o365_da3, multi_detector(_fused), rfdetr_seg_da3, yolo_seg_da3, yolo_o365_seg_da3, rfdetr_sam2_da3, yolo_o365_sam2_da3, rfdetr_semantic_da3, rfdetr_lanes_scene_da3, rfdetr_da3_track
│   └── labels/              # mapas de rótulos por vocabulário: coco_driving, objects365_driving
├── src/det_depth_distance/
│   ├── core/                # contratos e orquestração, sem frameworks
│   │   ├── types.py         # Frame, Instances, DepthResult, FrameResult, RefinedMasks, SemanticMap, TrackEstimate, MapGeometry, CameraIntrinsics
│   │   ├── options.py       # ConfigError, from_mapping (YAML -> dataclass com caminho nos erros)
│   │   ├── components.py    # Component, Detector, DepthEstimator, MaskRefiner, SemanticSegmenter, Tracker (ABCs) e opções base
│   │   ├── registry.py      # @register, build, available, load_adapters
│   │   ├── labels.py        # vocabulários COCO/Objects365, presets de classes, LabelMap, rótulos <nome>.<classe>
│   │   ├── boxes.py         # IoU, casamento guloso, clusters de NMS e WBF (numpy puro)
│   │   ├── aggregate.py     # Aggregator, Concat, FusionAggregator, Nms, Wbf
│   │   ├── agreement.py     # concordância entre detectores por quadro + resumo
│   │   ├── tracking.py      # IouTracker (IoU + velocidade constante) e TrackSummary (resumo por trilha)
│   │   └── pipeline.py      # Pipeline: detectores -> concordância -> agregação -> refinadores -> rastreador -> profundidade -> semântica -> elegibilidade -> distância -> suavização
│   ├── adapters/
│   │   ├── detectors/       # rfdetr.py, ultralytics_yolo.py
│   │   ├── depth/           # da3.py (geometria do mapa, focal, conversão para metros)
│   │   ├── refiners/        # box_prompt.py (base numpy), hf_sam.py (5 famílias transformers), ultralytics_sam.py
│   │   ├── semantic/        # segformer.py, twinlitenet.py
│   │   ├── _hf_hub.py       # carga offline de checkpoints HF + comando de download
│   │   └── _ultralytics_env.py  # política de ambiente do ultralytics (offline, sem auto-instalação)
│   ├── io/                  # video.py (ffmpeg), source.py (VideoSource), sinks.py (CSV, vídeo, npz, métricas)
│   ├── camera.py            # provedores de intrínsecos: focal fixa ou estimada
│   ├── config.py            # RunConfig, load_config, --set
│   ├── distance.py          # numpy puro: caixa ou máscara -> pixels do mapa -> estatísticas robustas
│   ├── temporal.py          # numpy puro: suavização por trilha (Kalman 1D, EMA, mediana) e SmoothingConfig
│   ├── render.py            # caixas, máscaras, rótulos, HUD, painel de profundidade + legenda
│   ├── runtime.py           # memória da GPU, versões, commit
│   ├── cli.py               # --config/--set/--list-models; raiz de composição
│   └── __main__.py          # python -m det_depth_distance
├── tests/                   # pytest sem GPU; tests/gpu/ (marcador gpu) roda no gpu-host
├── utils/
│   └── sync_gpu_host.sh      # rsync push/pull com o servidor GPU
├── docs/                    # tasks (docs/task_DDDD.md), template e refs/ (diagramas de arquitetura e do fluxo por quadro)
└── _testbed/                # (ignorado pelo git) vídeos, saídas, logs
```

---

## Pré-requisitos

| Requisito | Onde | Observação |
|---|---|---|
| Astral uv | laptop e servidor | no `gpu-host` fica em `~/.local/bin/uv` (fora do PATH) |
| ffmpeg / ffprobe | laptop e servidor | decodificação e codificação por pipe; também usados por parte dos testes |
| GPU NVIDIA com CUDA 12.8 | servidor | testado na RTX PRO 6000 Blackwell (sm_120) do `gpu-host` |
| Acesso `ssh gpu-host` | laptop | sincronização por rsync (`utils/sync_gpu_host.sh`) |
| Cache de pesos | servidor | DA3 via `hf_cache` no YAML (ex.: `~/dev/depth-anything-3/_testbed/weights/hf`); RF-DETR em `~/.roboflow/models` (as variantes Seg, ~135–155 MB cada, são baixadas pela biblioteca no primeiro uso, com MD5 conferido); YOLO26 em `~/dev/ultralytics/_testbed/weights` (caminho em `weights`; nada é baixado — pesos `-seg` vêm de `https://github.com/ultralytics/assets/releases/download/v8.4.0/<nome>.pt`); refinadores `transformers` e SegFormer no cache HF do projeto `weights/hf` (ignorado pelo git e pelo `push`; ver [Pesos](#pesos-dos-refinadores-e-do-segformer)); TwinLiteNet+ lido de `~/dev/TwinLiteNetPlus/pretrained/` |

> **Nota:** nunca instale o extra `infer` no laptop; a stack GPU ocupa ~9 GB.

---

## Início Rápido

Local (desenvolvimento, testes unitários e lint):

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Servidor GPU (inferência e testes de GPU: contrato dos adaptadores e
regressão contra a `task_0001`):

```bash
utils/sync_gpu_host.sh push
ssh gpu-host 'cd ~/dev/det-depth-distance && ~/.local/bin/uv sync --extra infer'
ssh gpu-host 'cd ~/dev/det-depth-distance && ~/.local/bin/uv run --extra infer pytest -m gpu -v'
```

O `uv run pytest` local nunca roda os testes de `tests/gpu/`
(`addopts = "-m 'not gpu'"`).

---

## Uso da CLI

No servidor GPU (a partir da raiz do repo; vídeos em `_testbed/data/`):

```bash
uv run --extra infer python -m det_depth_distance --config configs/rfdetr_da3.yaml
uv run --extra infer python -m det_depth_distance --config configs/multi_detector.yaml \
    _testbed/data/IMG_4022.MOV --set source.max_frames=300 --set render.show_stats=true
uv run --extra infer python -m det_depth_distance --list-models
```

| Argumento | Efeito |
|---|---|
| `--config PATH` | Arquivo YAML com a execução inteira (obrigatório, exceto com `--list-models`) |
| `VIDEO` (opcional) | Sobrescreve `source.video` |
| `--set chave=valor` | Repetível; caminho pontilhado com índice de lista (`detectors.0.threshold=0.3`); valor lido como YAML (`null`, `true`, números); um valor-mapa ou lista substitui a seção inteira (`--set 'aggregation={type: concat}'`) |
| `--list-models` | Lista detectores, modelos de profundidade, agregadores, rastreadores, refinadores de máscara e segmentadores semânticos registrados com suas opções e padrões |

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
`yolo_o365_seg_da3.yaml` (YOLO26-L-objv1-seg, Objects365 esperado — a conferir no servidor GPU —, com o mapa de
rótulos `objects365_driving`). Refinadores e semântica (`task_0005`):
`rfdetr_sam2_da3.yaml` e `yolo_o365_sam2_da3.yaml` (máscaras SAM 2.1 das
caixas, `region: both`), `rfdetr_semantic_da3.yaml` (SegFormer `scene`) e
`rfdetr_lanes_scene_da3.yaml` (TwinLiteNet+ `lanes` + SegFormer `scene`).
Rastreamento e suavização (`task_0006`): `rfdetr_da3_track.yaml`
(`rfdetr_da3.yaml` + rastreador IoU + Kalman, `layout: side-by-side`).
Caminhos aceitam `~` e, quando relativos, são
resolvidos a partir do diretório de execução (a raiz do repositório); chaves
desconhecidas são erro (com o caminho, ex.
`detectors[0].treshold: unknown option`).

| Seção | Chaves (padrão) |
|---|---|
| topo | `device` (`cuda:0`) |
| `source` | `video` (obrigatório aqui ou como `VIDEO`); `start` (0 s); `duration` (0 = até o fim); `max_frames` (0 = todos); `stride` (1) |
| `detectors` | Lista (≥ 1); cada item tem `type`, `name` opcional (padrão: `rfdetr-<variant>`, `rfdetr-<arquivo de pesos>` ou, no YOLO, o nome do arquivo de pesos) e as opções do adaptador. Comuns: `threshold` (0.5, em (0, 1]), `classes` (`driving`, `all`, lista ou nomes separados por vírgula), `labels` (null; YAML de mapa de rótulos, ver abaixo). `rfdetr`: `variant` (`large`; detecção `nano`/`small`/`medium`/`large`, segmentação `seg-nano`/`seg-small`/`seg-medium`/`seg-large`/`seg-xlarge`/`seg-2xlarge`), `weights` (checkpoint próprio), `optimize` (TorchScript + FP16). `yolo`: `weights` (obrigatório, `.pt` de detecção ou de segmentação `*-seg.pt`), `imgsz` (640, múltiplo de 32), `half` (FP16) |
| `depth` | `type: da3`; `model` (`da3metric-large`; alias, repo HF ou diretório; modelos relativos são recusados); `process_res` (504, ≥ 14); `process_res_method` (`upper_bound_resize`); `hf_cache` |
| `camera` | `focal_px` (null; focal em px do vídeo, tem precedência); `estimate` (`model: da3-large-1.1`, `frames: 16`, `process_res`, `process_res_method`, `hf_cache`; `null` desliga a estimativa) |
| `aggregation` | `type`: `concat` (padrão; mantém todas as caixas, marcadas por `source`), `nms` ou `wbf` (opções abaixo) |
| `refiners` | Lista (padrão `[]`), aplicada em ordem depois da agregação (cada um vê as máscaras do anterior); item `{type, name?, …}`, ver [Refinadores de máscara](#refinadores-de-máscara) |
| `semantic` | Lista (padrão `[]`) de segmentadores, rodados em ordem sobre o mesmo quadro; item `{name, type, …}` com `name` **obrigatório** e único (`[a-z0-9][a-z0-9_-]*`), ver [Segmentação semântica](#segmentação-semântica) |
| `tracking` | `null` (padrão: sem rastreamento) ou `{type: iou, name?, …}`, ver [Rastreamento e suavização](#rastreamento-e-suavização) |
| `smoothing` | `null` (padrão) ou seção com `filter` (`kalman`) e parâmetros do filtro; exige `tracking` |
| `agreement` | `enabled` (true; só tem efeito com ≥ 2 detectores); `iou_thr` (0.5, em (0, 1)); `class_aware` (true: só casa rótulos canônicos iguais) |
| `distance` | `box_shrink` (0.1); `outlier` (`iqr`); `outlier_k` (null = padrão do método); `stat` (`median`, `mean`, `trimmed`, `mode`, `pXX`); `distance_type` (`z` ou `euclidean`); `min_valid_px` (20); `max_range` (0 = sem limite); `mask_erode` (1; erosão da máscara em pixels do mapa, ≥ 0); `exclude_sky` (true); `region` (`mask`: máscara quando a instância tem, senão caixa; `box`: sempre a caixa; `both`: como `mask` e grava também a distância da caixa); `exclude_semantic` (`[]`; rótulos `<nome>.<classe>` — lista YAML ou texto com vírgulas — cujos pixels saem do cálculo, ex. `[scene.road, lanes.lane]`) |
| `render` | `layout` (`overlay` ou `side-by-side`); `show_stats` (false); `draw_inner_box` (false); `max_depth` (30 m); `cmap` (`turbo`); `crf` (20); `hud` (true); `font` (null); `mask_alpha` (0.35, em [0, 1]; opacidade da máscara; 0 não desenha máscaras); `semantic_alpha` (0.4, em [0, 1]; 0 não desenha camadas semânticas); `semantic_overlay` (null = todos os mapas; lista de nomes de `semantic` a desenhar); `show_smoothed` (true; rótulo com a distância suavizada quando há estimativa — a bruta segue no CSV); `show_speed` (true; velocidade relativa no rótulo, só com `kalman`) |
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
| famílias `transformers` | `model` (repo HF no formato `transformers`); `hf_cache` (null = cache HF padrão; os exemplos usam `weights/hf`); SAM-HQ: `hq_token_only` (false) |
| `ultralytics_sam` | `weights` (obrigatório; existência conferida antes do import); `imgsz` (1024, múltiplo de 32) |

`name` é opcional (padrão: nome do repo em minúsculas, ex.
`sam2.1-hiera-large`; no `ultralytics_sam`, o nome do arquivo sem extensão)
e aparece em `mask_source`, em `metrics.json` (`mask_sources`, `refined`,
`components.refiners`), em `timings_ms` (`refine.<nome>`) e no HUD; dois
refinadores do mesmo checkpoint exigem `name` explícito. Trocar o refinador
de um exemplo não exige YAML novo:
`--set 'refiners=[{type: edgetam, hf_cache: weights/hf}]'`.

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
| `twinlitenet` | `repo` (`~/dev/TwinLiteNetPlus`); `variant` (`large`; `nano`/`small`/`medium`); `weights` (null = `<repo>/pretrained/<variant>.pth`); `imgsz` (640); `half` (false) | `background`, `drivable`, `lane` (faixa vence a área trafegável) |

Domínios: Cityscapes e BDD100K são cenas de rua; em cena agrícola
`vegetation`/`terrain` cobrem a cultura — revisar a lista de exclusão.
`--set 'semantic=[{name: lanes, type: twinlitenet}]'` troca a lista inteira,
`--set semantic=[]` desliga a semântica e `--set semantic.<i>.size=[1024,1792]`
edita um item (`semantic.0` em `rfdetr_semantic_da3.yaml`, `semantic.1` em
`rfdetr_lanes_scene_da3.yaml`).

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

### Pesos dos refinadores e do SegFormer

Nada é baixado em tempo de execução (`local_files_only=True`); um peso
ausente gera `ConfigError` com o comando exato. No servidor, dentro do repo
(o filtro evita as cópias `.pt`/`.bin` duplicadas):

```bash
~/.local/bin/uv run --extra infer python -c "from huggingface_hub import snapshot_download as d; [d(r, cache_dir='weights/hf', allow_patterns=['*.json', 'model.safetensors']) for r in ('facebook/sam2.1-hiera-large', 'yonigozlan/EdgeTAM-hf', 'syscv-community/sam-hq-vit-huge')]; d('nvidia/segformer-b5-finetuned-cityscapes-1024-1024', cache_dir='weights/hf', allow_patterns=['*.json', 'pytorch_model.bin'])"
# opcionais: d('facebook/sam-vit-base', …) / d('facebook/sam3', …) com o mesmo filtro (sam3: HF_TOKEN com acesso aprovado)
curl -fL -o ~/dev/ultralytics/_testbed/weights/mobile_sam.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt
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

---

## Saídas

Cada execução cria `<output.out_root>/<AAAAmmdd_HHMMSS>_<vídeo>_<detectores>_<profundidade>[_tag]/`:

| Arquivo | Conteúdo |
|---|---|
| `video.mp4` | Vídeo anotado (H.264); com mais de um detector e `concat`, cor e contagem do HUD por modelo de origem; com fusão, cor pelo rótulo canônico, `×N` no rótulo quando N detectores viram o objeto e linhas de HUD com o agregador e a concordância (`agree <m> · only <a> <x> · only <b> <y>`); máscaras translúcidas na cor da instância, no quadro e no painel de profundidade, e `· region <política> · erode <k>` na linha de distância do HUD nos quadros em que alguma instância usou a máscara; camadas semânticas translúcidas (cor pelo nome da classe, por baixo das máscaras e caixas, só no painel do quadro, na ordem de `semantic`) e linhas de HUD `refine <nome> · <preenchidas>/<alvos>` e `semantic <nome> · excl <classes>`; com rastreamento, rótulo `#<id> classe conf · D m · ±v m/s` (distância suavizada com `show_smoothed`, velocidade com `show_speed`) e última linha de HUD `track iou · IoU ≥ 0.3 · age 0.5 s · kalman q 2 · ids <trilhas abertas>`; áudio copiado quando o vídeo inteiro é processado sem `stride` |
| `detections.csv` | Uma linha por caixa: quadro, tempo, classe, confiança, caixa, região amostrada no mapa, `n_valid`/`n_inliers`/`inlier_frac`, média, mediana, desvio, variância, p10–p90, mín/máx, `distance_m` (vazio = n/a) e, no fim, `source` (detector de origem), `raw_label` (nome no vocabulário do modelo), `n_sources` (nº de detectores que viram o objeto; 1 sem fusão), `sources`, `source_scores` e `raw_labels` (um item por detector, o seu melhor membro, separados por `\|`, em score decrescente), e por fim `used_mask` (1 = estatísticas sobre a máscara), `mask_px` (pixels do mapa da máscara erodida; `mask_px / n_box_px` = ocupação da caixa), `mask_erode` (erosão aplicada) e `box_distance_m` (só com `region: both`, distância pela caixa da instância que usou máscara) e, por último, `mask_source` (componente que produziu a máscara — detector, refinador ou membro da fusão dono da máscara —; vazio = sem máscara) e `mask_score` (qualidade prevista pelo refinador, 4 casas; vazio para máscaras de detector) e, no fim, `track_id`, `distance_smooth_m`, `distance_smooth_std_m`, `rel_speed_mps` e `rel_speed_std_mps` (vazias sem rastreamento/suavização; desvios e velocidade só com `kalman`; `distance_m` segue sendo a distância bruta). Com máscara, as colunas de estatística descrevem a máscara e `n_box_px` é a janela da caixa sem encolhimento. A semântica não acrescenta colunas |
| `metrics.json` | `components` (descrição de cada detector, da profundidade, do agregador, dos `refiners` e dos `semantic`), `pipeline` (política: `exclude_sky`, `agreement`, `region`, `mask_erode`, `exclude_semantic` como `{nome: [classes]}`, `smoothing`), `components.tracker` (null sem rastreamento), tempos por etapa (`det`, `det.<nome>`, `agree`, `aggregate`, `refine`/`refine.<nome>`, `track`, `semantic`/`semantic.<nome>` e `smooth` quando existem, `depth`, `dist`, `render`), `tracks` (só com rastreador: `n_tracks`, `short_tracks` — trilhas com menos de `min_duration_s` = 1 s, fora da lista — e `per_track` com `id`, `label` (moda), `labels`, `label_changes`, `sources`, `first_frame`/`last_frame`/`first_t`, `duration_s`, `n_frames`, `coverage`, `gaps`, `max_gap_s`, `distance_m` {min, median, max}, `jitter_raw_m`/`jitter_smooth_m` (desvio da diferença quadro a quadro) e `rel_speed_mps` {p5, median, p95}), `mask_sources` (instâncias com máscara por produtor), `refined` (por refinador: `targets` e `filled`), `semantic` (só com segmentador: por mapa `exclude`, `class_frac_mean`, `excluded_frac_mean`; no topo, `excluded_frac_mean` da união das exclusões), VRAM, focal (px do quadro e intrínsecos do mapa) e origem, parâmetros de distância, contagens (`detections_total`, `detections_by_source`, `detections_by_n_sources`), `distance_regions` (`{"mask": n, "box": n}`), `region_comparison` (só com `region: both`: diferença máscara − caixa com `n`, `median_signed_m`, `median_abs_m`, `p90_abs_m`, `median_rel`, total e `by_label`), `agreement` (com ≥ 2 detectores: por detector `instances`/`duplicates`; por par `matched`/`only_a`/`only_b`/`mean_iou`; séries por quadro em `per_frame`), versões (inclui `transformers` e `timm`), commit e `config` (YAML resolvido) |
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

O `DA3METRIC-LARGE` só fornece metros com a focal da câmera: informe
`camera.focal_px` (calibração) ou deixe que um modelo any-view a estime
(`camera.estimate`, mediana de `frames` quadros). `distance_type: euclidean`
converte a profundidade ao longo do eixo óptico em distância em linha reta.

---

## Sincronização com o servidor GPU

| Comando | Efeito |
|---|---|
| `utils/sync_gpu_host.sh push` | Envia o working tree (incluindo `.git`) para `gpu-host:~/dev/det-depth-distance`, sem tocar `.venv/`, `_testbed/` e `weights/` remotos |
| `utils/sync_gpu_host.sh pull [RUN]` | Traz `_testbed/outputs[/RUN]` do servidor para o `_testbed/outputs/` local |

Variáveis opcionais: `SYNC_HOST` (padrão `gpu-host`) e `SYNC_DEST` (padrão
`dev/det-depth-distance`, relativo ao `$HOME` remoto).

---

## Melhorias Futuras (TODO)

- [x] **Padrão de projeto** — esqueleto do repositório, AGENTS/CLAUDE/DESCRIPTION/CHANGELOG.
- [x] **Pipeline RF-DETR + DA3** (`task_0001`) — distância por caixa, vídeo anotado, CSV e métricas.
- [x] **Pipeline modular** (`task_0002`) — componentes abstratos, configuração YAML, detector YOLO26 e vários detectores no mesmo passo.
- [x] **Fusão entre detectores** (`task_0003`) — harmonização de rótulos, NMS/WBF entre modelos e métricas de concordância.
- [x] **Segmentação de instâncias** (`task_0004`) — YOLO26-seg e RF-DETR-Seg; distância pela máscara (`distance.region`, `mask_erode`), máscaras no vídeo e colunas de máscara no CSV.
- [x] **Refinadores de máscara e segmentação semântica** (`task_0005`) — família SAM (SAM2, EdgeTAM, SAM-HQ, SAM, SAM3) e MobileSAM via hierarquia `BoxPromptRefiner`; vários segmentadores semânticos nomeados (SegFormer, TwinLiteNet+) com exclusão de classes por `<nome>.<classe>`. Validação no servidor GPU pendente (testes `-m gpu`, pesos e execuções A–E).
- [x] **Rastreamento e suavização temporal** (`task_0006`) — rastreador IoU com velocidade constante (`track_id`), suavização por trilha (Kalman 1D, EMA, mediana), velocidade relativa e resumo por trilha no `metrics.json`. Validação no servidor GPU pendente (regressão `-m gpu` e execuções nos vídeos reais).
- [ ] **Calibração e avaliação** (`task_0007`) — focal calibrada, protocolo de verdade de campo e relatório de erro.
