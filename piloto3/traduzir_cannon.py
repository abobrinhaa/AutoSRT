import sys, json
sys.path.insert(0, "/opt/autosrt-dub")

from autosrt.transcribe import load_transcript
from autosrt.llm import LLMClient
from autosrt.llm_translate import translate_cues_llm

PILOTO = "/opt/autosrt-dub/piloto3"
cues = load_transcript(f"{PILOTO}/clip_en.srt")
print(f"{len(cues)} cues carregadas")

client = LLMClient(
    api_key="ollama-local",
    base_url="http://192.168.1.100:11434/v1",
    model="qwen2.5:14b-instruct-q4_K_M",
    timeout=120,
)

falhas = translate_cues_llm(cues, source_lang="inglês", client=client)
print("falhas:", falhas)

for c in cues:
    print(c.index, c.speaker, "|", c.text)

out = [{"index": c.index, "start": c.start, "end": c.end, "speaker": c.speaker, "text": c.text} for c in cues]
with open(f"{PILOTO}/cannon_traduzido.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("salvo")
