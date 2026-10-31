# Interface web do det-depth-distance — design

Data: 2026-10-31. Estado: rascunho para aprovação (nada foi construído).

## Intenção e porquê

O projeto é um estudo de como encadear detectores, profundidade, segmentadores,
rastreamento e previsão de trajetória numa pilha de percepção para veículos
autônomos, só com algoritmos e modelos abertos. A execução hoje é um YAML
(`configs/`) passado à CLI. Novos módulos entrarão em tasks futuras,
especializando as ABCs de `core/` e registrando-se em `adapters/`.

A interface web existe para duas coisas, as duas pedidas pelo usuário:

1. **Experimentar sem editar YAML à mão:** montar a pipeline, trocar ou
   adicionar módulos, mexer em estratégia de agregação, modelo de trajetória,
   parâmetros do Kalman, e reprocessar um vídeo enviado.
2. **Servir de peça de portfólio:** interface moderna, intuitiva, com
   aparência cuidada.

## Objetivos

- Editor visual da pipeline como uma **linha de etapas fixas** (as seções do
  YAML: fonte, detectores, agregação, refinadores, profundidade,
  rastreamento/suavização, semântica, movimento). Cada vaga aceita um ou mais
  módulos quando o YAML aceita lista (mais um detector, mais um refinador).
- Clicar em qualquer módulo abre o painel de configuração dele. Os campos são
  **gerados das dataclasses de opções** (`core/options.py`, `core/registry.py`),
  de modo que um adaptador novo aparece na interface sem mexer no front-end.
- Seções sem cartão na linha (`distance`, `camera`, `render`, `output`) ficam
  num painel de "Configurações gerais" (proposta do construtor do design).
- Abrir e salvar YAML; o YAML continua a fonte de verdade e o formato de troca.
- Carregar vídeo soltando-o na tela (ou colando um caminho do servidor), com
  reprodução imediata e metadados automáticos.
- Escolher um trecho (início e duração) e clicar em **Processar** para rodar
  só o trecho; **Processar vídeo inteiro** como segunda ação. Resultado do
  trecho (~3 s) com suavização e trajetória, chegando quadro a quadro, mesmo
  que leve alguns segundos.
- **Histórico** persistente: cada execução guarda o YAML usado, o vídeo de
  entrada, o vídeo anotado e as métricas; reabrir no editor, reexecutar,
  comparar duas execuções (diff de YAML, indicadores, gráfico de distância
  sobreposto).

## Não-objetivos

- Não é um editor de grafo livre: ligações arbitrárias entre nós e ordem de
  etapas diferente da do código ficam fora (o código não as executa).
- Não reprocessa ao mudar parâmetro; só ao clicar em Processar (decisão do
  usuário).
- Sem contas, autenticação ou multiusuário: uso pessoal, acessado de outra
  máquina da rede local (decisão do construtor).
- Sem edição de pesos, treino ou download de modelos pela interface.
- Sem avaliação contra referências de campo/CARLA nesta entrega (a bancada é a
  `task_0012`); a interface não deve impedir ligá-la depois.

## Anti-objetivos (seria fracasso mesmo "funcionando")

- Parecer um painel genérico (Gradio/Streamlit) ou um editor de nós que
  promete uma flexibilidade que o código não tem.
- Quebrar o contrato do projeto: `core/` e `adapters/` não podem passar a
  importar a camada web; a CLI continua funcionando sem o extra web.
- Um YAML salvo pela interface que a CLI recuse, ou o contrário.

## Restrições

- Uma GPU (RTX 3070, 8 GB); um job por vez. O processo web não pode disputar
  a GPU com um lote em andamento sem aviso (hoje há um lote de 16 vídeos
  rodando na mesma máquina).
- Imports de framework de ML só dentro dos adaptadores (regra do `CLAUDE.md`):
  o servidor deve subir, listar módulos e validar YAML sem torch.
- Todo parâmetro temporal é em segundos; `Frame.t` define o `dt`.
- Esta máquina não tem `node`; o front-end exige instalá-lo sem sudo.
- Processo do repositório: toda implementação exige `docs/task_DDDD.md`
  antes do código; `DESCRIPTION.md`, `README.md` e `CHANGELOG.md` atualizados
  no mesmo commit; nunca commitar em `main`.

## Abordagens consideradas

- **A. FastAPI + React/TypeScript (React Flow para a pipeline).** Escolhida
  pelo usuário. Backend importa o `Pipeline`; o esquema das opções vem das
  dataclasses; o progresso e os quadros chegam por SSE. Mais código e um
  build de front-end; é a que entrega o editor, a linha do tempo e a
  comparação sob medida e o melhor valor de portfólio.
- **B. FastAPI + HTMX/Alpine, sem build.** Mais simples; editor visual e
  linha do tempo ficam pobres. Descartada.
- **C. Gradio/Streamlit.** Mais rápida; visual genérico, não comporta o
  editor por etapas, não serve de vitrine. Descartada.

## Decisões tomadas pelo usuário

- Opção A.
- Editor por etapas fixas com módulos adicionáveis, não grafo livre.
- Painel de configuração por módulo ao clicar.
- Preview = trecho curto processado ao clicar **Processar**, não ao mudar
  parâmetro; a espera de alguns segundos é aceita.
- Histórico guardando YAML, vídeo de entrada, vídeo de saída (e comparação).
- Mockups aprovados em direção (editor e execuções); acesso remoto pela rede
  local.
- **À prova de futuro:** novos modelos e também novas etapas de pipeline
  entrarão no futuro e devem aparecer na interface sem editar a camada web
  (substitui a decisão anterior de etapas fixas numa tabela da web). A fonte
  única das etapas, das opções e das dicas de interface fica junto do código
  da pipeline (registro de etapas em `core/registry.py` e metadados nos campos
  das dataclasses); a web só lê. A execução de uma etapa nova continua sendo
  trabalho da `Pipeline`/`cli.run`. O front-end tem alternativas genéricas
  para etapa, campo e artefato desconhecidos e um registro de extensão para
  telas próprias; o catálogo é versionado.
- Interface em **inglês** e **somente tema escuro** (sem alternância de tema
  nem tradução nesta entrega). O restante do repositório segue em pt-BR.

## Decisões do construtor (o usuário pode revisar)

- Pacote novo `src/det_depth_distance/web/` (API) e `web/` ou `frontend/` na
  raiz (UI), com extra `web` no `pyproject.toml`. Sem tocar em `core/` e
  `adapters/`, exceto o registro de etapas e os metadados de UI (aditivos, sem importar a
  web), necessários para o requisito de futuro.
- (Revisada) Dicas de interface ficam como metadados dos campos das
  dataclasses, não numa tabela da web; as etapas vêm de um registro em
  `core/registry.py`.
- Esquema: um endpoint devolve, por tipo de módulo registrado, um JSON
  Schema derivado das dataclasses (tipos, `Literal`, limites, padrões); os
  erros de `ConfigError` são devolvidos com o caminho pontilhado do campo.
- Execuções reaproveitam o formato atual de pasta de `_testbed/outputs/`
  (`detections.csv`, `video.mp4`, `trajectories.jsonl`, `metrics.json`) mais o
  YAML efetivo; o histórico é um índice leve sobre essas pastas.
- Um único trabalhador de GPU, fila de jobs, cancelamento entre quadros.
- Biblioteca de vídeos com deduplicação por hash; prévia em MP4 gerada em
  segundo plano (o navegador não toca `.mkv` de taxa variável). Os `.h264`
  da Raspberry Pi com `meta.json` são convertidos automaticamente pelo mesmo
  procedimento do `remux_vfr.py`.
- Servidor escuta em `127.0.0.1` por padrão; `--host 0.0.0.0` explícito para
  acesso pela rede local.
- Parâmetros do Kalman e afins ficam visíveis no painel; o que for raro
  fica em "Avançado" recolhido.

### Decisões do construtor após a revisão do documento (o usuário pode revisar)

- **Onde roda:** o servidor roda nesta máquina (RTX 3070, 8 GB), onde ficam
  `data/`, `weights/` e `_testbed/outputs/`; o usuário só a acessa pela rede.
  Ambiente único: `uv run --extra infer --extra web` (com
  `UV_PROJECT_ENVIRONMENT=_testbed/venv-infer`, como no `CLAUDE.md`). Sem o
  extra `infer`, o servidor sobe e só lista módulos/valida YAML (sem torch).
- **GPU ocupada:** a API mede a memória livre com `torch.cuda.mem_get_info`
  (no worker; `nvidia-smi` pode falhar) e, se houver pouca, **avisa e deixa
  o usuário confirmar**; não bloqueia e não há trava entre processos na v1.
- **Execução:** cada "Processar" roda num **subprocesso trabalhador** (fácil de
  cancelar com `kill`, isola falhas e memória da GPU), que grava a pasta da
  execução e emite progresso por linhas JSON. Modelos residentes entre
  execuções ficam para depois; a v1 carrega a cada execução (aceitável).
- **Trecho:** `source.start` e `source.duration` já existem no YAML; o
  "Processar" preenche esses campos (e `stride`). O YAML efetivo, com o trecho
  incluído, é o que o histórico guarda.
- **Preview quadro a quadro:** a v1 mostra o quadro anotado (JPEG por SSE,
  desenhado no servidor pelo mesmo `render`) e, ao fim, o `video.mp4`
  completo; gráficos leem o `detections.csv`. Sobreposições desenhadas pelo
  front-end ficam para depois.
- **Esquema das opções:** dicas de interface (grupos, "Avançado", rótulos,
  unidades) ficam numa tabela do pacote `web/`, não em `core/`; se uma
  dataclass precisar de metadados novos, isso vira uma task própria.
- **YAML:** o salvo pela interface é regenerado (`yaml.safe_dump`); comentários
  e ordem original de chaves não são preservados, e a interface avisa. Chaves
  desconhecidas ao esquema são mantidas e mostradas como "extras".
- **Histórico:** inclui as pastas antigas de `_testbed/outputs/` (somente
  leitura, tolerando YAML efetivo ou trajetórias ausentes) e as novas
  execuções gravam na mesma pasta, com `output.tag` único para evitar a
  colisão do `make_run_dir`.
- **Front-end:** `node` instalado sem sudo via gerenciador de versões; o
  build (`web/dist`) não é commitado e é servido pelo FastAPI; o repositório
  documenta o passo de build.
- **Mockups** estão em `docs/superpowers/specs/mockups/web-interface-mockup.html`
  como referência de direção, não como especificação de pixels.

## Deixado ao construtor

- Escolha de bibliotecas (React Flow, gerenciador de estado, estilo), do
  índice do histórico (arquivos ou SQLite) e do transporte exato dos quadros.
- **Modelos residentes na GPU entre execuções:** hoje `cli.run()` compõe e
  carrega tudo a cada chamada. O construtor decide como reaproveitar
  componentes entre execuções com a mesma configuração de modelos (e como
  descarregar), sem violar a regra de imports lazy. Se isso se mostrar
  grande, entregar primeiro com carga a cada execução e otimizar depois.
- Cache de detecção e profundidade por quadro para reexecuções que só mudam
  rastreamento/suavização: otimização futura, fora desta entrega.
- Divisão em tasks (`docs/task_DDDD.md`): sugestão de ordem — (1) API de
  esquema e validação, (2) execução e histórico, (3) editor, (4) vídeo e
  trecho, (5) resultados e comparação.

## Riscos conhecidos

- Tempo do trecho: ~2 quadros/s medidos com a config recomendada; 3 s a
  25 fps são ~75 quadros (30–40 s); com `stride` 3–4 cai para ~10 s.
- `Pipeline` pode não separar etapas o bastante para o cache futuro.
- Compartilhar a GPU com lotes longos: a interface deve mostrar quando a GPU
  está ocupada.
