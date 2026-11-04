import json
from collections import defaultdict

with open("/opt/autosrt-dub/piloto3/cannon_traduzido.json", encoding="utf-8") as f:
    cues = json.load(f)

totals = defaultdict(float)
counts = defaultdict(int)
for c in cues:
    totals[c["speaker"]] += (c["end"] - c["start"]) / 1000
    counts[c["speaker"]] += 1

for spk, dur in sorted(totals.items(), key=lambda x: -x[1]):
    print(f"{spk}: {dur:.1f}s em {counts[spk]} falas")
