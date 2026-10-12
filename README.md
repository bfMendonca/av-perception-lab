<p align="center">
  <h1 align="center">det-depth-distance</h1>
  <p align="center">
    Distância por objeto combinando detecção RF-DETR e profundidade métrica Depth Anything 3
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
- [Saídas](#saídas)
- [Como a distância é calculada](#como-a-distância-é-calculada)
- [Sincronização com o servidor GPU](#sincronização-com-o-servidor-gpu)
- [Melhorias Futuras (TODO)](#melhorias-futuras-todo)

---

## Sobre o Projeto

Protótipo de pesquisa que executa, sobre os mesmos quadros de um vídeo, um
detector de objetos (RF-DETR) e um estimador de profundidade monocular
métrica (Depth Anything 3). Para cada caixa detectada, os pixels de
profundidade dentro da caixa são agregados com estatísticas robustas (média,
mediana, variância, percentis, rejeição de outliers) para estimar a
distância do objeto à câmera. A saída é um vídeo anotado com caixas e
distâncias, mais um CSV por detecção.

O foco é **qualidade da estimativa**, não tempo real: ambos os modelos rodam
em PyTorch, quadro a quadro. A inferência é feita no servidor GPU `gpu-host`;
o desenvolvimento e os testes unitários rodam localmente.

---

## Funcionalidades

- **Detectar** objetos quadro a quadro com RF-DETR (`nano`, `small`, `medium`, `large`, ou checkpoint próprio), com filtro de classes validado contra o vocabulário do modelo.
- **Estimar** profundidade métrica com Depth Anything 3 (`da3metric-large` com focal informada ou estimada; modelos NESTED já em metros).
- **Calcular** a distância de cada caixa com estatísticas robustas: encolhimento da caixa, exclusão de céu, rejeição de outliers (`iqr`, `mad`, `sigma`) e estatística configurável (`median`, `mean`, `trimmed`, `mode`, `pXX`), em profundidade `z` ou distância euclidiana.
- **Renderizar** vídeo anotado (rótulo `classe conf · D m ± σ`, HUD com tempos) e, opcionalmente, o mapa de profundidade lado a lado com legenda em metros.
- **Registrar** uma linha de CSV por caixa com todas as estatísticas, `metrics.json` com tempos, VRAM, focal e versões, e opcionalmente os mapas em `depth.npz`.
- **Validar** argumentos antes de carregar modelos e reportar erros de configuração sem traceback (código de saída 2).

---

## Arquitetura e Stack

| Componente | Tecnologia | Versão / origem |
|---|---|---|
| Linguagem | Python | 3.11 |
| Ambiente e dependências | Astral uv | `pyproject.toml` + `uv.lock` |
| Framework de inferência | PyTorch + torchvision | 2.8.0 / 0.23.0 (índice cu128) |
| Detecção | `rfdetr` (Roboflow RF-DETR) | git `1f2af2ea` |
| Profundidade | `depth-anything-3` (ByteDance Seed) | git `3d835ec` |
| Atenção eficiente (DA3) | xformers | 0.0.32.post2 |
| Vídeo | ffmpeg (sistema), OpenCV, Pillow | — |

As dependências pesadas ficam no extra `infer`, instalado apenas no host de
inferência; localmente basta o ambiente leve (numpy, OpenCV, Pillow, pytest,
ruff).

---

## Estrutura do Projeto

```
det-depth-distance/
├── pyproject.toml           # manifesto uv; extra "infer" com a stack GPU
├── uv.lock                  # lock versionado
├── src/det_depth_distance/
│   ├── cli.py               # argparse, laço por quadro, CSV/metrics.json/vídeo
│   ├── detector.py          # wrapper RF-DETR (Detections em numpy, filtro de classes)
│   ├── depth.py             # wrapper DA3: geometria do mapa, focal, conversão para metros
│   ├── distance.py          # numpy puro: caixa -> pixels do mapa -> estatísticas robustas
│   ├── render.py            # caixas, rótulos, HUD, painel de profundidade + legenda
│   ├── video_io.py          # ffprobe/ffmpeg por pipe
│   └── __main__.py          # python -m det_depth_distance
├── tests/                   # pytest (sem GPU), um arquivo por módulo
├── utils/
│   └── sync_gpu_host.sh      # rsync push/pull com o servidor GPU
├── docs/                    # tasks (docs/task_DDDD.md) e template
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
| Cache de pesos | servidor | DA3 via `--hf-cache` (ex.: `~/dev/depth-anything-3/_testbed/weights/hf`); RF-DETR em `~/.roboflow/models` |

> **Nota:** nunca instale o extra `infer` no laptop; a stack GPU ocupa ~9 GB.

---

## Início Rápido

Local (desenvolvimento, testes unitários e lint):

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Servidor GPU (inferência):

```bash
utils/sync_gpu_host.sh push
ssh gpu-host 'cd ~/dev/det-depth-distance && ~/.local/bin/uv sync --extra infer'
```

---

## Uso da CLI

No servidor GPU (a partir da raiz do repo; vídeos em `_testbed/data/`):

```bash
uv run --extra infer python -m det_depth_distance _testbed/data/IMG_4021.MOV \
    --layout side-by-side --draw-inner-box --show-stats \
    --hf-cache ~/dev/depth-anything-3/_testbed/weights/hf
```

Use `--extra infer` também no `uv sync`: sem ele o sync (exato por padrão)
remove a stack GPU. `--help` lista todos os argumentos com padrões.

| Grupo | Argumentos (padrão) |
|---|---|
| Entrada/saída | `video`; `--out-root` (`_testbed/outputs`); `--start` (0 s); `--duration` (0 = até o fim); `--max-frames` (0 = todos); `--stride` (1); `--tag` |
| Detector | `--det-model {nano,small,medium,large}` (`large`); `--det-weights` (checkpoint próprio); `--det-threshold` (0.5); `--classes` (`driving`, `all` ou lista); `--det-optimize` (TorchScript + FP16) |
| Profundidade | `--depth-model` (`da3metric-large`; alias, repo HF ou diretório); `--process-res` (504); `--process-res-method` (`upper_bound_resize`); `--focal-px` (0 = estimar; px do vídeo); `--focal-model` (`da3-large-1.1`); `--focal-frames` (16); `--hf-cache` |
| Distância | `--box-shrink` (0.1 por lado); `--outlier {none,iqr,mad,sigma}` (`iqr`); `--outlier-k` (padrão do método); `--distance-stat` (`median`, `mean`, `trimmed`, `mode`, `pXX`); `--distance-type {z,euclidean}` (`z`); `--min-valid-px` (20); `--max-range` (0 = sem limite); `--exclude-sky` / `--no-exclude-sky` (ligado) |
| Renderização | `--layout {overlay,side-by-side}` (`overlay`); `--show-stats` (± desvio-padrão); `--draw-inner-box`; `--max-depth` (30 m); `--cmap` (`turbo`); `--crf` (20); `--no-hud`; `--font` |
| Execução | `--device` (`cuda:0`); `--save-depth` |

---

## Saídas

Cada execução cria `<out-root>/<AAAAmmdd_HHMMSS>_<vídeo>_<detector>_<profundidade>[_tag]/`:

| Arquivo | Conteúdo |
|---|---|
| `video.mp4` | Vídeo anotado (H.264); áudio copiado quando o vídeo inteiro é processado sem `--stride` |
| `detections.csv` | Uma linha por caixa: quadro, tempo, classe, confiança, caixa, região amostrada no mapa, `n_valid`/`n_inliers`/`inlier_frac`, média, mediana, desvio, variância, p10–p90, mín/máx, `distance_m` (vazio = n/a) |
| `metrics.json` | Tempos por etapa (det, depth, dist, render), VRAM, focal (px do quadro e intrínsecos do mapa) e origem, parâmetros de distância, contagens, versões, commit, argumentos |
| `depth.npz` | Só com `--save-depth`: mapas em metros (`float16`), índices dos quadros, intrínsecos por quadro (`fx, fy, cx, cy` na resolução do mapa) e geometria (`geometry` + `geometry_fields`) |

Prévia leve para trazer ao laptop:
`ffmpeg -i video.mp4 -vf scale=-2:720 -crf 26 preview_720p.mp4`.

---

## Como a distância é calculada

1. O DA3 gera um mapa de profundidade em resolução própria (ex.: 504×280 para
   1920×1080); cada caixa do RF-DETR é levada ao mapa com escala separada em
   x/y (e deslocamento nos métodos `*_crop`).
2. A caixa é encolhida (`--box-shrink`), e pixels inválidos, além do limite
   (`--max-range`) ou de céu (`--exclude-sky`), são descartados.
3. Outliers (fundo dentro da caixa) são rejeitados (`--outlier`).
4. A distância é a estatística escolhida (`--distance-stat`) dos pixels
   restantes; com menos de `--min-valid-px` pixels, o rótulo mostra `n/a`.

O `DA3METRIC-LARGE` só fornece metros com a focal da câmera: informe
`--focal-px` (calibração) ou deixe que um modelo any-view a estime (mediana
de `--focal-frames` quadros). `--distance-type euclidean` converte a
profundidade ao longo do eixo óptico em distância em linha reta.

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
- [ ] **Pipeline modular** (`task_0002`, planejada) — componentes abstratos, configuração YAML, detector YOLO26 e vários detectores no mesmo passo.
- [ ] **Fusão entre detectores** — harmonização de rótulos e NMS/WBF entre modelos.
- [ ] **Segmentação** — máscaras de instância (YOLO26-seg, RF-DETR-Seg, SAM2) no lugar das caixas.
- [ ] **Suavização temporal** — rastreamento e filtragem das distâncias ao longo do tempo.
- [ ] **Calibração da câmera** — focal medida em vez de estimada; validação com distâncias de referência.
