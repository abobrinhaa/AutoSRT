import sys
sys.path.insert(0, "/opt/autosrt-dub")
from autosrt.transcribe import load_transcript
from collections import defaultdict

cues = load_transcript("/opt/autosrt-dub/piloto2/filme_completo_en.srt")
print(f"{len(cues)} cues no total")

totals = defaultdict(float)
counts = defaultdict(int)
for c in cues:
    spk = c.speaker or "SEM_LOCUTOR"
    totals[spk] += c.duration / 1000
    counts[spk] += 1

for spk, dur in sorted(totals.items(), key=lambda x: -x[1]):
    print(f"{spk}: {dur:.1f}s de fala em {counts[spk]} falas ({dur/60:.1f} min)")

import json
out = [{"index": c.index, "start": c.start, "end": c.end, "speaker": c.speaker, "text": c.source_text} for c in cues]
with open("/opt/autosrt-dub/piloto2/filme_completo_cues.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("salvo em filme_completo_cues.json")
