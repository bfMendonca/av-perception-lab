# CLAUDE.md — det-depth-distance

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
- **`README.md`** — documentação humana completa em pt-BR.
- **`CHANGELOG.md`** — changelog homogêneo da casa; toda entrega relevante
  ganha entrada.
- **`docs/task_DDDD.md`** — toda implementação exige um arquivo de tarefa
  (ver AGENTS.md § "Regras de Ciclo de Vida de Tarefas").

## Comandos comuns

```bash
uv sync                                   # ambiente leve local
uv run pytest                             # testes unitários (sem GPU; pula -m gpu)
uv run ruff check --fix . && uv run ruff format .
utils/sync_gpu_host.sh push                # código -> gpu-host:~/dev/det-depth-distance
ssh gpu-host 'cd ~/dev/det-depth-distance && ~/.local/bin/uv sync --extra infer'
ssh gpu-host 'cd ~/dev/det-depth-distance && ~/.local/bin/uv run --extra infer pytest -m gpu -v'
ssh gpu-host 'cd ~/dev/det-depth-distance && ~/.local/bin/uv run --extra infer python -m det_depth_distance --config configs/rfdetr_da3.yaml'
utils/sync_gpu_host.sh pull [RUN_DIR]      # saídas <- gpu-host
```

## Arquitetura (big picture)

Dois ambientes, um manifesto: o laptop roda só o ambiente leve (numpy,
OpenCV, Pillow) para desenvolver e testar a lógica de distância; o servidor
GPU `gpu-host` (RTX PRO 6000 Blackwell, sm_120) instala o extra `infer`
(torch 2.8.0+cu128, xformers, `rfdetr` e `depth-anything-3` fixados por
commit git, `ultralytics==8.4.175`, `transformers` e `timm`). Não há git remote: o código vai por rsync.

Código em camadas: `core/` (contratos `Instances`/`DepthResult`/`SemanticMap`,
ABCs `Detector`/`DepthEstimator`/`MaskRefiner`/`SemanticSegmenter`, registro,
`LabelMap`, agregadores `concat`/`nms`/`wbf`, geometria de caixas,
concordância, `Pipeline`) não importa frameworks; `adapters/` envolve cada
framework (refinadores: `MaskRefiner` → `BoxPromptRefiner` → `HFSamRefiner` →
famílias finas, + `UltralyticsSamRefiner`); `io/` (vídeo, fonte, sinks);
`cli.py` é só a raiz de composição.

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
- No servidor, `uv sync` sem `--extra infer` remove a stack GPU (o sync é
  exato por padrão); use `--extra infer` no `uv sync` e no `uv run`.
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
