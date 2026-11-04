import json

with open("/opt/autosrt-dub/piloto2/legenda_pt_real_v2_locutores.json", encoding="utf-8") as f:
    linhas = json.load(f)

original_dur = {}
# recomputa original_duration a partir de start/end
for l in linhas:
    l["original_duration"] = (l["end"] - l["start"]) / 1000

outliers = []
for l in linhas:
    ratio = l["synth_duration"] / max(l["original_duration"], 0.3)
    if ratio > 2.2:
        outliers.append(l["index"])
        print(f"{l['index']} ({l['speaker']}): gerado {l['synth_duration']:.1f}s original {l['original_duration']:.1f}s ratio {ratio:.1f}x | {l['text'][:40]}")

print(f"\n{len(outliers)} de {len(linhas)} com duracao desproporcional")
print("indices:", outliers)
