# Protocolo de calibração da câmera e de coleta de distâncias de referência

Roteiro de campo da task_0007: como calibrar a câmera das gravações e como
medir distâncias de referência (verdade de campo) para avaliar o pipeline
com `det-depth-distance-evaluate`. Ferramentas: `det-depth-distance-calibrate`
(`calibrate`, `board`, `inspect`) e `det-depth-distance-evaluate`.

## 1. Por que calibrar

- Com `DA3METRIC-LARGE`, `metros = saída · (fx + fy) / 2 / 300` (fx/fy na
  resolução do mapa): **a distância é proporcional à focal**. Um erro de
  10 % na focal vira 10 % na distância.
- Sem calibração a focal é estimada por um modelo any-view do DA3, e a
  estimativa variou **~8 %** entre dois vídeos da mesma câmera (IMG_4021:
  1390,4 px; IMG_4022: 1502,9 px), embora os metadados QuickTime das duas
  gravações sejam idênticos (iPhone 14 Pro Max, câmera principal 6,86 mm,
  24 mm equivalente).
- O "35 mm equivalente" **não** dá a focal do vídeo: 24 mm-eq. em
  1920×1080 dá 1222 px pela diagonal (1280 px pela largura de 36 mm), e o
  recorte da estabilização eletrônica, que aumenta a focal efetiva, não é
  registrado no arquivo. `inspect` mostra esse valor só como diagnóstico.

## 2. Gravação do alvo de calibração

1. Gerar o alvo e imprimir em **100 %** (sem "ajustar à página"):

   ```bash
   uv run det-depth-distance-calibrate board --out _testbed/board_9x6_25mm.png --dpi 300
   # ChArUco (aceita o alvo parcialmente visível, ajuda a cobrir os cantos do quadro):
   uv run det-depth-distance-calibrate board --target charuco --cols 7 --rows 5 \
       --square-mm 30 --marker-mm 22 --out _testbed/charuco_7x5.png
   ```

   O tabuleiro padrão tem 9×6 cantos internos (10×7 quadrados) de 25 mm.
2. Colar a impressão numa placa **plana e rígida** (MDF, acrílico, papelão
   grosso) e **medir** o lado do quadrado com paquímetro (média de vários
   quadrados). Usar o valor medido em `--square-mm`.
3. Gravar com o **mesmo aparelho e o mesmo modo** das gravações de campo:
   câmera principal "1×", mesma resolução e taxa (ex.: 1080p30), mesmos
   ajustes de estabilização (padrão/cinematográfico/ação — registrar qual),
   zoom e modos especiais (macro, cinematográfico, retrato) desligados, foco
   travado se possível (toque longo na tela: AE/AF travado).
4. 30–60 s movendo o alvo (ou a câmera) **devagar** — borrão de movimento
   estraga os cantos — cobrindo o centro, as bordas e **os quatro cantos**
   do quadro, com inclinações de ±30–45° em torno dos dois eixos e
   distâncias variadas. Boa iluminação, sem reflexo no alvo.
5. Repetir a gravação no início e no fim de cada campanha de campo (detecta
   mudanças de modo/atualização de software).

## 3. Calibrar

```bash
# Metadados da câmera (lente, 35 mm-eq.) do vídeo de calibração e dos vídeos de campo
uv run det-depth-distance-calibrate inspect _testbed/data/CALIB_01.MOV
uv run det-depth-distance-calibrate inspect _testbed/data/IMG_4021.MOV

# Calibração (um quadro a cada 0,5 s, até 60 vistas; --debug-dir salva as vistas detectadas)
uv run det-depth-distance-calibrate calibrate _testbed/data/CALIB_01.MOV \
    --square-mm 24.9 --note "estabilizacao padrao, 1080p30, camera 1x" \
    --out configs/calibration/iphone14promax_main_1080p30.yaml \
    --debug-dir _testbed/calib_debug

# Conferir a calibração contra um vídeo de campo (código 1 se a câmera não confere)
uv run det-depth-distance-calibrate inspect _testbed/data/IMG_4021.MOV \
    --calibration configs/calibration/iphone14promax_main_1080p30.yaml
```

O vídeo é decodificado exatamente como nas execuções do pipeline (quadro
"em pé", 1920×1080), então a calibração vale para os mesmos pixels. Também
aceita imagens: `calibrate --images "_testbed/calib/*.png" --out ...`.

**Critérios de aceitação** (conferidos e impressos pelo `calibrate`: `[accept] ok: …`
ou, em stderr, `[warn] not accepted: …`; o arquivo é gravado mesmo assim):

| Critério | Limite |
|---|---|
| RMS de reprojeção (`rms_px`) | < 0,5 px |
| Cobertura do quadro (`coverage`, grade 8×6) | ≥ 0,6 (aviso abaixo disso) |
| Desvio-padrão de fx (`std_intrinsics.fx`) | < 0,5 % de fx |
| Vistas usadas | ≥ 12 (mínimo da ferramenta); 30–60 recomendado |

Salvar como `configs/calibration/<aparelho>_<lente>_<modo>.yaml` (arquivo
pequeno, versionado). O arquivo é um `FileStorage` do OpenCV (YAML com
`%YAML:1.0`, matrizes `!!opencv-matrix`) — legível por qualquer ferramenta
OpenCV, mas **não** por `yaml.safe_load`. A matriz está na convenção do
OpenCV (centro do pixel `i` em `i`); o projeto soma 0,5 a cx/cy na leitura.

Usar na execução (`configs/rfdetr_da3_calibrated.yaml`):

```yaml
camera:
  calibration:
    path: configs/calibration/iphone14promax_main_1080p30.yaml
    undistort: false    # true: corrige a distorção de cada quadro
    alpha: 0.0
    check_lens: error   # compara modelo/lente/35 mm-eq. da calibração com os do vídeo
    cross_check: true   # também estima a focal e registra as duas no metrics.json
```

## 4. Verdade de campo (distâncias de referência)

- **Instrumentos**, sempre com incerteza declarada no arquivo de
  referências: trena laser (ex.: ±3 mm ± 0,1 %), marcos no solo medidos com
  trena (±5 cm), GNSS RTK base + rover (±3 cm; também para objetos em
  movimento).
- **O que medir**: do **centro da lente** até a **face do objeto voltada
  para a câmera** (registrar em `target`, ex.: `rear_face`) — é o que o
  pipeline vê. Linha reta = `quantity: range`; se medir ao longo do eixo
  óptico (ex.: marcos alinhados com a direção da câmera), `quantity: z`.
  A avaliação converte `range` ↔ `z` com a K da execução.
- **Câmera parada** (tripé ou veículo parado) para referências estáticas.
- **Variedade**: objetos de 3 a 40 m, de classes diferentes (carro,
  caminhão, pessoa…), no centro **e** nas bordas do quadro. Mínimo
  sugerido para a primeira avaliação: 10 referências estáticas.

## 5. Sincronismo

- Usar o **índice de quadro** identificado no próprio vídeo, nunca o relógio
  do celular (`creationdate` tem resolução de 1 s).
- Marcar cada referência com um **evento visual**: placa com o id da
  referência (`r01`, `r02`…) mostrada à câmera, palma, lanterna.
- RTK: evento visual no início e no fim da passagem; converter a série para
  o tempo do vídeo (`series: [[t_s, d_m], …]`, `t` como `Frame.t`, segundos
  desde o início do vídeo).

## 6. Anotação

Copiar `configs/references/example_references.yaml` (exemplo **fictício**)
e preencher um arquivo por vídeo:

- `video`: nome do arquivo do vídeo (conferido contra o `metrics.json`).
- `frames: [primeiro, último]` (inclusivo) **ou** `t: [início_s, fim_s]`.
- `select`: `box: [x0, y0, x1, y1]` (recomendado), `point: [x, y]` (a menor
  caixa que contém o ponto) ou `track_id` (só vale para uma execução;
  usar apenas com objetos em movimento). `label` filtra pela classe.
- Coordenadas em pixels do **vídeo original** (1920×1080, origem no canto
  superior esquerdo), mesmo quando a execução corrige a distorção — a
  avaliação transforma as coordenadas. Para obtê-las, pausar o vídeo num
  quadro do intervalo (ex.: `ffmpeg -ss <t> -i VIDEO -frames:v 1 q.png` e
  um visualizador de imagens que mostre a posição do cursor).
- `distance_m` (constante) **ou** `series` (RTK); `instrument` ou
  `uncertainty_m`.

## 7. Avaliar

```bash
# Uma execução: relatório em <execução>/eval_<nome das referências>/
uv run det-depth-distance-evaluate --refs refs/IMG_4100.yaml _testbed/outputs/<execução>

# Comparar configurações (focal estimada × calibrada × undistort × region both × suavização)
uv run det-depth-distance-evaluate --refs refs/IMG_4100.yaml RUN_A RUN_B RUN_C --out _testbed/eval_4100

# Gráficos (precisa do matplotlib: extra eval no laptop)
uv run --extra eval det-depth-distance-evaluate --refs refs/IMG_4100.yaml RUN_A --plot
# Com o ambiente de inferência, NUNCA --extra eval sozinho (o sync exato removeria a stack GPU);
# o --extra infer já traz o matplotlib:
#   uv run --extra infer det-depth-distance-evaluate --refs ... RUN_A --plot
```

Leitura do `eval.md`:

- **Nível por referência** (padrão): cada referência conta uma vez (mediana
  do erro nos seus quadros) — quadros vizinhos são correlacionados;
  `--level frame` pondera por quadro. `n_refs` é sempre reportado.
- Erro = estimativa − referência (positivo = estimou longe demais);
  `median_abs_rel`/`p90_abs_rel` são os números principais; `within_2u` é a
  fração dos erros dentro de 2× a incerteza da medida; `scale_median`
  (mediana de estimativa/referência) mostra erro de escala.
- **Cobertura**: quadros detectados com distância válida, detectados "n/a"
  e não detectados, e o nº de `track_id` distintos por referência (> 1 =
  troca de ID).
- **Focal implícita** (modelos `metric_focal`): a focal que zeraria o erro
  de escala de `distance_m` — comparar com a calibrada e a estimada.
- Estimadores: `distance_m` (estatística configurada), as estatísticas
  das *inliers* (`median`, `mean`, `p10`…`p90`, `min`), `box_distance_m`
  (caixa, com `distance.region: both`) e `distance_smooth_m` (suavizada).
