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
(`Detector`, `DepthEstimator`, agregador): cada modelo é um adaptador que
trata as particularidades do seu framework e entrega sempre os mesmos tipos
(`Instances`, `DepthResult`), e a execução inteira é descrita por um arquivo
YAML (`configs/`).

O foco é **qualidade da estimativa**, não tempo real: os modelos rodam
em PyTorch, quadro a quadro. A inferência é feita no servidor GPU `gpu-host`;
o desenvolvimento e os testes unitários rodam localmente.

---

## Funcionalidades

- **Detectar** objetos quadro a quadro com RF-DETR (`nano`, `small`, `medium`, `large`, ou checkpoint próprio) e/ou Ultralytics YOLO26 (pesos COCO ou Objects365), com filtro de classes por preset do vocabulário (`driving` para COCO e Objects365) validado contra as classes do modelo.
- **Combinar** vários detectores no mesmo passo (`aggregation: concat`): todas as caixas são mantidas e marcadas com o modelo de origem (`source`), no CSV e por cor no vídeo.
- **Estimar** profundidade métrica com Depth Anything 3 (`da3metric-large` com focal informada ou estimada; modelos NESTED já em metros).
- **Calcular** a distância de cada caixa com estatísticas robustas: encolhimento da caixa, exclusão de céu, rejeição de outliers (`iqr`, `mad`, `sigma`) e estatística configurável (`median`, `mean`, `trimmed`, `mode`, `pXX`), em profundidade `z` ou distância euclidiana.
- **Renderizar** vídeo anotado (rótulo `classe conf · D m ± σ`, HUD com tempos) e, opcionalmente, o mapa de profundidade lado a lado com legenda em metros.
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
├── configs/                 # exemplos de YAML: rfdetr_da3, yolo_o365_da3, multi_detector
├── src/det_depth_distance/
│   ├── core/                # contratos e orquestração, sem frameworks
│   │   ├── types.py         # Frame, Instances, DepthResult, FrameResult, MapGeometry, CameraIntrinsics
│   │   ├── options.py       # ConfigError, from_mapping (YAML -> dataclass com caminho nos erros)
│   │   ├── components.py    # Component, Detector, DetectorOptions, DepthEstimator (ABCs)
│   │   ├── registry.py      # @register, build, available, load_adapters
│   │   ├── labels.py        # vocabulários COCO/Objects365 e presets de classes
│   │   ├── aggregate.py     # Aggregator, Concat
│   │   └── pipeline.py      # Pipeline: detectores -> agregação -> profundidade -> distância
│   ├── adapters/
│   │   ├── detectors/       # rfdetr.py, ultralytics_yolo.py
│   │   └── depth/           # da3.py (geometria do mapa, focal, conversão para metros)
│   ├── io/                  # video.py (ffmpeg), source.py (VideoSource), sinks.py (CSV, vídeo, npz, métricas)
│   ├── camera.py            # provedores de intrínsecos: focal fixa ou estimada
│   ├── config.py            # RunConfig, load_config, --set
│   ├── distance.py          # numpy puro: caixa -> pixels do mapa -> estatísticas robustas
│   ├── render.py            # caixas, rótulos, HUD, painel de profundidade + legenda
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
| Cache de pesos | servidor | DA3 via `hf_cache` no YAML (ex.: `~/dev/depth-anything-3/_testbed/weights/hf`); RF-DETR em `~/.roboflow/models`; YOLO26 em `~/dev/ultralytics/_testbed/weights` (caminho em `weights`; nada é baixado) |

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
| `--list-models` | Lista detectores, modelos de profundidade e agregadores registrados com suas opções e padrões |

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
`multi_detector.yaml` (os dois detectores, `layout: side-by-side`). Caminhos
aceitam `~`; chaves desconhecidas são erro (com o caminho, ex.
`detectors[0].treshold: unknown option`).

| Seção | Chaves (padrão) |
|---|---|
| topo | `device` (`cuda:0`) |
| `source` | `video` (obrigatório aqui ou como `VIDEO`); `start` (0 s); `duration` (0 = até o fim); `max_frames` (0 = todos); `stride` (1) |
| `detectors` | Lista (≥ 1); cada item tem `type`, `name` opcional (padrão: `rfdetr-<variant>`, `rfdetr-<arquivo de pesos>` ou, no YOLO, o nome do arquivo de pesos) e as opções do adaptador. Comuns: `threshold` (0.5, em (0, 1]), `classes` (`driving`, `all`, lista ou nomes separados por vírgula), `labels` (reservado para os mapas de rótulos da `task_0003`; hoje só aceita `null`). `rfdetr`: `variant` (`large`), `weights` (checkpoint próprio), `optimize` (TorchScript + FP16). `yolo`: `weights` (obrigatório, `.pt`), `imgsz` (640, múltiplo de 32), `half` (FP16) |
| `depth` | `type: da3`; `model` (`da3metric-large`; alias, repo HF ou diretório; modelos relativos são recusados); `process_res` (504, ≥ 14); `process_res_method` (`upper_bound_resize`); `hf_cache` |
| `camera` | `focal_px` (null; focal em px do vídeo, tem precedência); `estimate` (`model: da3-large-1.1`, `frames: 16`, `process_res`, `process_res_method`, `hf_cache`; `null` desliga a estimativa) |
| `aggregation` | `type: concat` (mantém todas as caixas, marcadas por `source`) |
| `distance` | `box_shrink` (0.1); `outlier` (`iqr`); `outlier_k` (null = padrão do método); `stat` (`median`, `mean`, `trimmed`, `mode`, `pXX`); `distance_type` (`z` ou `euclidean`); `min_valid_px` (20); `max_range` (0 = sem limite); `exclude_sky` (true) |
| `render` | `layout` (`overlay` ou `side-by-side`); `show_stats` (false); `draw_inner_box` (false); `max_depth` (30 m); `cmap` (`turbo`); `crf` (20); `hud` (true); `font` (null) |
| `output` | `out_root` (`_testbed/outputs`); `tag` (""); `save_depth` (false) |

---

## Saídas

Cada execução cria `<output.out_root>/<AAAAmmdd_HHMMSS>_<vídeo>_<detectores>_<profundidade>[_tag]/`:

| Arquivo | Conteúdo |
|---|---|
| `video.mp4` | Vídeo anotado (H.264); com mais de um detector, cor e contagem do HUD por modelo de origem; áudio copiado quando o vídeo inteiro é processado sem `stride` |
| `detections.csv` | Uma linha por caixa: quadro, tempo, classe, confiança, caixa, região amostrada no mapa, `n_valid`/`n_inliers`/`inlier_frac`, média, mediana, desvio, variância, p10–p90, mín/máx, `distance_m` (vazio = n/a) e, no fim, `source` (detector de origem) e `raw_label` (nome no vocabulário do modelo) |
| `metrics.json` | `components` (descrição de cada detector, da profundidade e do agregador), `pipeline` (política, ex. `exclude_sky`), tempos por etapa (`det`, `det.<nome>`, `aggregate`, `depth`, `dist`, `render`), VRAM, focal (px do quadro e intrínsecos do mapa) e origem, parâmetros de distância, contagens (`detections_total`, `detections_by_source`), versões, commit e `config` (YAML resolvido) |
| `depth.npz` | Só com `output.save_depth: true`: mapas em metros (`float16`), índices dos quadros, intrínsecos por quadro (`fx, fy, cx, cy` na resolução do mapa) e geometria (`geometry` + `geometry_fields`) |

Prévia leve para trazer ao laptop:
`ffmpeg -i video.mp4 -vf scale=-2:720 -crf 26 preview_720p.mp4`.

---

## Como a distância é calculada

1. O DA3 gera um mapa de profundidade em resolução própria (ex.: 504×280 para
   1920×1080); cada caixa detectada é levada ao mapa com escala separada em
   x/y (e deslocamento nos métodos `*_crop`).
2. A caixa é encolhida (`distance.box_shrink`), e pixels inválidos, além do
   limite (`distance.max_range`) ou de céu (`distance.exclude_sky`), são
   descartados.
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
- [ ] **Fusão entre detectores** (`task_0003`) — harmonização de rótulos e NMS/WBF entre modelos.
- [ ] **Segmentação de instâncias** (`task_0004`) — YOLO26-seg e RF-DETR-Seg; distância pela máscara.
- [ ] **Refinadores de máscara e segmentação semântica** (`task_0005`) — família SAM (SAM2, EdgeTAM, SAM-HQ, SAM, SAM3) e MobileSAM via hierarquia `BoxPromptRefiner`; vários segmentadores semânticos nomeados (SegFormer, TwinLiteNet+) com exclusão de classes por `<nome>.<classe>`.
- [ ] **Rastreamento e suavização temporal** (`task_0006`) — ID por objeto, distância suavizada e velocidade relativa.
- [ ] **Calibração e avaliação** (`task_0007`) — focal calibrada, protocolo de verdade de campo e relatório de erro.
