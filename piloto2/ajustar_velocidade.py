import json, subprocess, os

PILOTO = "/opt/autosrt-dub/piloto2"
os.makedirs(f"{PILOTO}/synth_sync", exist_ok=True)

with open(f"{PILOTO}/legenda_pt_real_v2_locutores.json", encoding="utf-8") as f:
    linhas = json.load(f)

MAX_SPEEDUP = 1.6  # acima disso a voz comeca a soar estranha (efeito "esquilo")

ajustadas = 0
for linha in linhas:
    src = f"{PILOTO}/synth_multi/{linha['index']}.wav"
    dst = f"{PILOTO}/synth_sync/{linha['index']}.wav"
    original_dur = linha["original_duration"]
    synth_dur = linha["synth_duration"]

    if synth_dur <= original_dur * 1.05:
        # ja cabe, so copia
        subprocess.run(["cp", src, dst], check=True)
        linha["final_duration"] = synth_dur
        continue

    tempo = min(synth_dur / original_dur, MAX_SPEEDUP)
    # atempo so aceita 0.5-2.0 por filtro, dentro desse range estamos ok
    subprocess.run([
        "ffmpeg", "-y", "-i", src, "-af", f"atempo={tempo:.4f}",
        dst, "-loglevel", "error"
    ], check=True)
    nova_dur = synth_dur / tempo
    linha["final_duration"] = nova_dur
    linha["tempo_aplicado"] = tempo
    ajustadas += 1
    print(f"{linha['index']}: {synth_dur:.2f}s -> {nova_dur:.2f}s (tempo={tempo:.2f}x, original={original_dur:.2f}s)")

print(f"\n{ajustadas} de {len(linhas)} falas tiveram velocidade ajustada")

with open(f"{PILOTO}/legenda_pt_real_v3_sync.json", "w", encoding="utf-8") as f:
    json.dump(linhas, f, ensure_ascii=False, indent=2)
print("OK")
