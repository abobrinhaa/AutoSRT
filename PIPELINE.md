# Pipeline de dublagem com IA — notas de desenvolvimento (2026-11-01)

Documento de progresso do projeto de dublagem automática. Objetivo final: ferramenta
dentro do autosrt (como "atualizar legendas", "atualizar gêneros" etc.) onde o usuário
aponta um filme e o processo roda sozinho.

Status atual: **validando o pipeline manualmente, passo a passo, antes de automatizar.**
Nada disso ainda está empacotado como ferramenta — são scripts soltos em `/opt/autosrt-dub/piloto2/`.

## Onde tudo roda
- Container CT104 (mesmo do autosrt), clone separado em `/opt/autosrt-dub`, branch `claude/dublagem-ia`
- Venv próprio: `/opt/autosrt-dub/venv`
- GPU compartilhada (RTX 3060, 12GB) com autosrt e jellyfin — rodar só uma etapa pesada por vez

## Etapas do pipeline (ordem)

### 1. Transcrição + diarização (áudio original, inglês)
- Ferramenta: Faster-Whisper-XXL (já existe em `/opt/faster-whisper-xxl`, reusado do autosrt)
- Comando base: `faster-whisper-xxl <audio> --model large-v3 --device cuda --diarize pyannote_v3.1 --word_timestamps True --hallucination_silence_threshold 2.0 --condition_on_previous_text False --standard`
- **Importante**: diarizar o filme inteiro de uma vez tende a juntar personagens diferentes num só SPEAKER_XX em cenas com diálogo rápido. Diarizar só a cena/trecho específico (não o filme todo) deu resultado bem melhor — ex: um trecho que saiu "1 speaker só" no filme inteiro saiu "3 speakers corretos" quando diarizado sozinho, com `--max_speakers 3` como dica.
- **Limitação conhecida, não resolvida**: mesmo diarizando só a cena, ainda ocorrem erros de classificação pontuais (personagem B não detectado dentro de um bloco do personagem A). Não escala corrigir ouvindo linha por linha pro filme inteiro — precisa de UI de revisão eventualmente.

### 2. Escolher o personagem/trecho pra dublar
- Analisar `duração total de fala por SPEAKER_XX` pra achar quem tem mais material (bom pra referência de voz)
- Script usado: soma `cue.duration` por `cue.speaker` num dicionário, ordena decrescente

### 3. Tradução
- **Não usar OpenRouter** (opção que existe no código do autosrt mas não é usada na prática)
- Fonte de tradução usada nos testes: a legenda PT já existente embutida no arquivo do filme (`ffmpeg -map 0:<indice da legenda> arquivo.srt`), recortada pro trecho de interesse
- Alternativa pra quando não existe legenda PT pronta: Ollama rodando no **host Proxmox** (`192.168.1.100:11434`, não no CT104), modelo `qwen2.5:14b-instruct-q4_K_M` (já estava instalado lá antes do projeto). Reusar `autosrt.llm_translate.translate_cues_llm` + `autosrt.llm.LLMClient` apontando `base_url` pro Ollama do host.

### 4. Referência de voz limpa (Demucs)
- Extrair os trechos de áudio do(s) locutor(es) alvo (pelos timestamps da diarização), concatenar num arquivo só
- **Nunca rodar o Demucs no áudio do filme inteiro de uma vez** — estoura os 16GB de RAM do CT104 (confirmado, OOM kill). Rodar só no áudio já recortado do(s) locutor(es)/cena.
- Comando: `python3 -m demucs --two-stems=vocals -n htdemucs -d cuda -o <pasta> <audio.wav>`
- Saída: `vocals.wav` (voz limpa, pra referência do XTTS) e `no_vocals.wav` (música/ambiente, pra manter no mix final)
- Isso resolveu o problema de "voz com chiado/ruído" que a clonagem direto do áudio original causava

### 5. Síntese de voz (XTTS-v2)
- Pacote `TTS` 0.22.0 (Coqui, projeto não é mais mantido — atenção a isso em futuras atualizações de dependências)
- **Dependências frágeis, cuidado ao reinstalar**:
  - `torch==2.5.1+cu121` e `torchaudio==2.5.1+cu121` — se o `pip install TTS` puxar uma versão mais nova de torch automaticamente, quebra o cuDNN (erro `CUDNN_STATUS_NOT_INITIALIZED`). Se acontecer, desinstalar TODOS os pacotes `torch`/`nvidia-*` e reinstalar do zero (não dá pra só reinstalar o torch por cima)
  - `transformers==4.41.2` — versão mais nova (5.x) quebra o import `BeamSearchScorer`
  - Ao carregar o modelo, preciso liberar classes pro `torch.load` (PyTorch 2.5+ mudou o padrão `weights_only`):
    ```python
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
    ```
- **Parâmetros validados que evitam artefatos** (confirmado ouvindo):
  - `split_sentences=False`
  - Tirar o ponto final do texto antes de mandar (`re.sub(r"[.]+$", "", texto)`) — sem isso o modelo às vezes fala a palavra "ponto" em voz alta
  - `temperature=0.65` (padrão é 0.75) — mais estável
- **Problema não totalmente resolvido**: frases muito curtas (1-4 palavras) ainda geram áudio desproporcionalmente longo (hallucination/repetição), mesmo com os parâmetros acima. Mitigação atual: gerar de novo (até 4 tentativas) e aceitar a menos ruim se nenhuma ficar boa. Sem solução definitiva ainda.

### 6. Ajuste de sincronismo (velocidade)
- XTTS não tem controle de duração alvo — o áudio gerado raramente bate com a duração original da fala
- No teste real (74 falas), **62% precisaram de ajuste de velocidade** pra caber no tempo original
- Aplicar `ffmpeg -af atempo=<fator>`, limitado a `MAX_SPEEDUP = 1.6x` (acima disso a voz soa "esquilo")
- Mesmo com o limite de 1.6x, algumas falas muito comprimidas (pouco tempo original, texto longo) continuam estourando o espaço — não tem solução só acelerando áudio pronto
- **Limitação fundamental identificada**: isso resolve só duração, não resolve "ritmo"/cadência natural da fala. Dublagem profissional resolve isso **reescrevendo a tradução** pra caber no tempo certo, não só comprimindo o áudio depois. Ainda não implementado.

### 7. Preencher brechas com áudio original (fallback)
- Pedido explícito do usuário: nunca deixar silêncio onde não há fala dublada — tocar o áudio original (inglês) nesses intervalos
- Implementado via remontagem segmento a segmento (não dá pra usar o demuxer `-f concat` do ffmpeg direto quando os segmentos têm parâmetros de áudio diferentes — gera erro de decodificação tipo `channel element N.M is not allocated`; resolvido convertendo tudo pra WAV puro antes de concatenar)
- **Problema em aberto, reportado pelo usuário mas não diagnosticado**: em algum ponto a fala em inglês parece "vazar" sob a fala em português. Testei o nível do fundo separado pelo Demucs em 9 pontos da cena (todos entre -72dB e -84dB, praticamente silêncio) — não é vazamento do Demucs. Suspeita não confirmada: a alternância rápida entre inglês (brecha) e português (fala) pode estar criando sensação de "eco"/sobreposição mesmo sendo sequencial, não simultâneo. **Precisa investigar mais ou repensar a abordagem** (ex: só usar fallback em inglês nos trechos realmente sem diálogo, nunca logo antes/depois de uma fala dublada).

### 8. Volume final
- A dublagem clássica embutida nos filmes é bem mais "comprimida"/alta que o áudio original e que a trilha gerada
- Aplicar `loudnorm=I=-18:TP=-1.5:LRA=11` no mix final ajuda, mas não testei se bate exatamente com o nível que o usuário espera

### 9. Montagem do arquivo final (mux)
- **Sempre** `-map_chapters -1` — capítulos herdados do filme original (com timestamps fora do range do trecho cortado) causam "erro fatal do player" no Jellyfin
- **Nunca** usar `-c:v copy` num recorte feito com `-ss` antes do `-i` — gera timestamps DTS não-monotônicos que também quebram o player. Sempre reencodar o vídeo (`-c:v libx264`) ao cortar um trecho.
- Testar sempre com `ffmpeg -v warning -xerror -i arquivo.mkv -f null -` antes de publicar — pega erro de decodificação que não aparece só olhando os metadados com ffprobe

### 10. Testar no Jellyfin
- Criar uma pasta própria por item em `/mnt/filmes_storage/<nome da pasta>/<arquivo>.mkv` — **o Jellyfin usa o nome da PASTA como título**, não o nome do arquivo
- Monitoramento em tempo real da biblioteca pode não pegar mudança de arquivo sozinho depois de um crash do serviço — às vezes precisa forçar scan manual

### 7b. Correção da causa raiz do dessincronismo (resolvido)
A montagem por concatenação sequencial (pedaço atrás de pedaço) deixava a duração
total **flutuar** em vez de bater exatamente com a duração do trecho original (ex:
240s virou 241.2s). Esse erro pequeno se acumula ao longo do vídeo e explica o
sintoma de "ora atrasa, ora adianta" relatado em testes diferentes.

**Correção**: cada fala agora ocupa um slot de duração **fixa e exata**, igual ao
`end - start` original daquela fala — nunca mais, nunca menos:
- Se a fala dublada (já com ajuste de velocidade) for mais curta que o slot, completa com fundo (`apad`)
- Se ainda for mais longa, corta no limite do slot (`atrim`)
- O cursor sempre avança exatamente `end` da fala original, nunca o fim real do áudio gerado

Resultado validado: trilha de 150s de entrada saiu com exatamente `150.000000s` de saída (era `241.2s` para um alvo de `240s` antes da correção). Confirmado pelo usuário: sincronismo ficou bom depois dessa correção.

### 7c. Resto de palavra em inglês no fim da fala dublada (resolvido)
Mesmo com o sincronismo corrigido, sobrava um pedacinho de palavra em inglês logo
depois de cada fala dublada. Causa: o timestamp de **fim** da legenda/transcrição
quase sempre vem um pouco antes do áudio original realmente silenciar (a pessoa
ainda está terminando de falar quando a legenda já "acabou"). Como a brecha seguinte
começa exatamente nesse ponto e usa o áudio original como fallback, ela pegava esse
restinho.

**Correção**: as primeiras `250ms` de cada brecha agora tocam o fundo (sem voz,
`no_vocals.wav`) em vez do áudio original completo — um "colchão" curto que mascara
a cauda da palavra em inglês sem comprometer a duração total (que continua exata).

## Problemas ainda em aberto (para quando for sistematizar)
1. Erro de diarização pontual mesmo diarizando só a cena (personagem não detectado)
2. Frases curtas ainda geram áudio desproporcional às vezes, mesmo com retry
3. Ritmo/cadência da fala não combina com a cena (zero-shot não tem esse controle — fine-tuning de peso é o caminho mais provável, ainda não feito)
4. Vazamento percebido de inglês sob a dublagem em português — causa não confirmada (pode ter sido só sintoma do dessincronismo, já corrigido no passo 7b — reavaliar depois dessa correção)
5. Nada disso está automatizado ainda — é tudo script manual, rodado passo a passo

### 7d. Falas cortando no meio da palavra (resolvido parcialmente)
A correção 7b (slot de duração fixa) trouxe de volta um problema antigo: quando a
fala dublada ainda é mais longa que o slot mesmo depois de acelerada (`MAX_SPEEDUP`),
o `atrim` corta no meio da palavra. Em conteúdo com diálogo muito rápido (testado
com episódio de série, falas de 0.4-0.8s no original) isso acontecia na maioria das
falas, não só em casos raros.

**Duas causas identificadas e atacadas**:
1. Tradução livre do Ollama não considerava o tempo disponível → pedido explícito
   de tradução **concisa com orçamento de palavras** (`~2.8 palavras/segundo em
   português`, calculado a partir da duração original de cada fala). Ajuda bastante
   nas falas médias/longas.
2. **Limite técnico do XTTS**: mesmo com 1-2 palavras, o modelo não gera áudio abaixo
   de ~1.2-2 segundos. Encurtar o texto não resolve esses casos — é piso do modelo,
   não da tradução.

**Tentativa de mitigação (rejeitada pelo usuário)**: tradução concisa (orçamento de
palavras) + fala podendo usar o espaço livre até a próxima fala começar antes de
cortar. Tecnicamente reduziu de ~30 falas cortando pra 8 de 47, mas o usuário ouviu
o resultado e achou pior ("ficou uma merda") — a tradução mais curta deve ter
sacrificado naturalidade/sentido demais, ou o efeito do espaço "emprestado" da
próxima fala soou estranho. **Revertido** pra versão anterior (7c: slot fixo exato,
aceita cortar no meio de palavra em casos raros) como melhor estado conhecido.
Lição: nem toda correção tecnicamente "melhor" nos números soa melhor ouvindo —
sempre validar ouvindo antes de assumir que resolveu.

### 7e. Pesquisa de projetos abertos existentes + técnica de desacelerar vídeo
Antes de continuar remendando, pesquisamos se já existe ferramenta open-source pronta
pra isso (Linly-Dubbing, pyvideotrans, open-dubbing, Python-Autodub). Conclusão: usam
a mesma técnica de fundo (acelerar áudio até ~1.5-2x, tradução pensando no tempo) —
não existe solução mágica pronta, é limite real do problema. Mas o **pyvideotrans**
documenta uma técnica que não tínhamos usado: **desacelerar o vídeo** (não só acelerar
o áudio) para absorver o excesso, dividindo o ajuste entre os dois lados.

**Implementado**: quando a fala dublada não cabe mesmo depois de acelerada (até 1.6x),
em vez de cortar, o vídeo daquele trecho específico é desacelerado (`setpts`) até a
fala caber. Limite de desaceleração do vídeo elevado pra `4.0x` a pedido do usuário
("prefiro demorar e ficar perfeito do que rápido e cortado" — ver
[[feedback-qualidade-sobre-velocidade]]). Resultado: **0 de 47 falas cortando**
(eram 14 mesmo depois da primeira tentativa com limite de vídeo em 1.6x). O vídeo
final fica mais longo que o original (162s vs 150s no teste) — isso é esperado e
aceito, é a troca consciente: vídeo mais longo/com variação de ritmo, mas sem perder
nenhuma palavra.

**Bug de tradução encontrado e corrigido à parte**: uma fala (índice 9) estava com
6.8x de estouro — não era problema de voz nem de tempo, era a tradução duplicando o
conteúdo: o Whisper cortou uma frase em inglês em dois cues ("I can't take that
chance..." + "...of their cases"), e a tradução por LLM, sem contexto de que eram a
mesma frase, traduziu os DOIS cues como a frase inteira. Corrigido mesclando os dois
cues num só antes de sintetizar, em vez de forçar a dublagem a caber artificialmente
num espaço de 0.6s. Vale ficar de olho nesse tipo de bug ao automatizar: cues muito
curtos (<1s) com texto desproporcional ao original valem uma checagem antes de ir
pra síntese.

## Feedback do usuário sobre a UI (mockups comparados)
Dois mockups de interface foram gerados pra comparar abordagens de unificar "Legendas"
e "Dublagem IA" na mesma ferramenta. O usuário preferiu a versão que reaproveita o
visual real do autosrt (extraído do CSS/componentes reais do app, não um tema
genérico): https://claude.ai/artifact/XcxHYPqQsYCBYLuS2XZDcA

Decisões de produto confirmadas pelo usuário:
1. **Legenda e dublagem são operações independentes** — precisa dar pra gerar só a
   legenda, só a dublagem, ou os dois juntos. Não forçar o fluxo completo sempre.
2. **Banco de vozes por ATOR, não por filme** (feature importante pra quando
   escalar pro acervo inteiro, ~85 mil itens): o Jellyfin já cataloga o elenco de
   cada filme (confirmado no banco: tabela de pessoas com nomes reais, ex.
   "J.D. Cannon", "Dyan Cannon"). Em vez de clonar/treinar a voz do zero a cada
   filme, a voz de um ator deveria ser reaproveitada entre todos os filmes em que
   ele aparece — muitos atores se repetem num acervo desse tamanho. Isso economiza
   processamento (não repete Demucs+treino por filme) e melhora a qualidade com o
   tempo (mais material acumulado = referência melhor). Ainda não implementado,
   mas é a próxima peça importante de arquitetura a desenhar antes de automatizar
   de verdade.

## Segundo teste de validação (série, não filme)
Testado também com um episódio de série ("Cannon", 1976, sem legenda PT embutida —
validou a tradução via Ollama pela primeira vez de verdade, não só a legenda já pronta
usada nos testes com filme). Confirma que o pipeline generaliza além do caso de teste
original. Descoberta à parte: a biblioteca de séries real do Jellyfin fica em
`/mnt/jellyfin_storage/SERIADOS/` — storage **diferente** de `/mnt/filmes_storage`
onde todo teste deste documento rodou. Pra dublar conteúdo de série de verdade no
futuro, o arquivo final precisa ir pra esse outro storage.
