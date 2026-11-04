import sys, json
sys.path.insert(0, "/opt/autosrt-dub")
from autosrt.llm import LLMClient

PILOTO = "/opt/autosrt-dub/piloto3"

with open(f"{PILOTO}/cannon_traduzido.json", encoding="utf-8") as f:
    cues = json.load(f)

client = LLMClient(
    api_key="ollama-local",
    base_url="http://192.168.1.100:11434/v1",
    model="qwen2.5:14b-instruct-q4_K_M",
    timeout=60,
)

SYSTEM = """Voce traduz falas de dublagem do ingles para portugues do Brasil.

REGRA MAIS IMPORTANTE: a traducao precisa caber no tempo de fala original.
Voce recebe o numero maximo de palavras permitido para cada fala -- NUNCA
ultrapasse esse limite, mesmo que precise simplificar ou cortar detalhes
menos essenciais da frase. Priorize o sentido principal da fala sobre
detalhes secundarios.

Responda APENAS com a traducao, sem aspas, sem explicacao, sem comentario."""

from autosrt.transcribe import load_transcript
en_cues = load_transcript(f"{PILOTO}/clip_en.srt")
en_por_index = {c.index: c.source_text for c in en_cues}

for c in cues:
    texto_en = en_por_index.get(c["index"], "")
    slot_s = c["original_duration"]
    max_palavras = max(1, round(slot_s * 2.8))  # ~2.8 palavras/seg em portugues falado
    prompt = f'Fala em ingles: "{texto_en}"\nTempo disponivel: {slot_s:.1f} segundos (maximo {max_palavras} palavras em portugues).\nTraducao:'
    try:
        resposta = client.complete(SYSTEM, prompt, temperature=0.3)
        resposta = resposta.strip().strip('"').strip()
    except Exception as e:
        print(f"erro em {c['index']}: {e}")
        resposta = c["text"]
    print(f"{c['index']} (max {max_palavras}p, {slot_s:.1f}s): {resposta}")
    c["text_curto"] = resposta

with open(f"{PILOTO}/cannon_traduzido.json", "w", encoding="utf-8") as f:
    json.dump(cues, f, ensure_ascii=False, indent=2)
print("OK")
