import json

with open("/opt/autosrt-dub/piloto2/legenda_pt_real_recortada.json", encoding="utf-8") as f:
    linhas = json.load(f)

outliers = 0
for l in linhas:
    ratio = l["synth_duration"] / max(l["original_duration"], 0.3)
    if ratio > 2.2:
        outliers += 1
        print(f"{l['index']}: gerado {l['synth_duration']:.1f}s original {l['original_duration']:.1f}s ratio {ratio:.1f}x | {l['text'][:50]}")

print(f"\n{outliers} de {len(linhas)} linhas com duracao desproporcional (>2.2x)")
