import sys
sys.path.insert(0, "/opt/autosrt-dub")
from autosrt.transcribe import load_transcript
from collections import defaultdict

cues = load_transcript("/opt/autosrt-dub/piloto2/clip_en_real_rediarizado/clip_en_real.srt")
totals = defaultdict(float)
counts = defaultdict(int)
for c in cues:
    spk = c.speaker or "SEM_LOCUTOR"
    totals[spk] += c.duration / 1000
    counts[spk] += 1

for spk, dur in sorted(totals.items(), key=lambda x: -x[1]):
    print(f"{spk}: {dur:.1f}s em {counts[spk]} falas")

print()
for c in cues[:15]:
    print(c.index, c.speaker, c.start, c.end, "|", c.source_text)

import json
out = [{"index": c.index, "start": c.start, "end": c.end, "speaker": c.speaker, "text": c.source_text} for c in cues]
with open("/opt/autosrt-dub/piloto2/clip_en_real_cues_v2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
