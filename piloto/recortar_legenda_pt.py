import sys
sys.path.insert(0, "/opt/autosrt-dub")

from autosrt import srt_io

CLIP_START_MS = 600_000
CLIP_END_MS = 780_000

cues = srt_io.load_cues("/opt/autosrt-dub/piloto/legenda_pt_completa.srt")
recortadas = []
for c in cues:
    if c.end <= CLIP_START_MS or c.start >= CLIP_END_MS:
        continue
    c.start = max(0, c.start - CLIP_START_MS)
    c.end = min(CLIP_END_MS - CLIP_START_MS, c.end - CLIP_START_MS)
    recortadas.append(c)

for c in recortadas:
    print(c.index, c.start, c.end, "|", c.source_text.replace("\n", " / "))

print(f"\n{len(recortadas)} falas no trecho")

import json
out = [
    {"index": c.index, "start": c.start, "end": c.end, "text": c.source_text}
    for c in recortadas
]
with open("/opt/autosrt-dub/piloto/legenda_pt_recortada.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("salvo em legenda_pt_recortada.json")
