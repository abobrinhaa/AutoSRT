import json, subprocess

PILOTO = "/opt/autosrt-dub/piloto2"
CLIP_DURATION_S = 240.0

with open(f"{PILOTO}/legenda_pt_real_v2_locutores.json", encoding="utf-8") as f:
    linhas = json.load(f)

inputs = []
filters = []
for i, linha in enumerate(linhas):
    path = f"{PILOTO}/synth_multi/{linha['index']}.wav"
    inputs += ["-i", path]
    delay_ms = int(linha["start"])
    filters.append(f"[{i}:a]adelay={delay_ms}:all=1[a{i}]")

mix_inputs = "".join(f"[a{i}]" for i in range(len(linhas)))
filters.append(f"{mix_inputs}amix=inputs={len(linhas)}:duration=longest:dropout_transition=0[vozes]")
filters.append(f"[vozes]apad=whole_dur={CLIP_DURATION_S}[out]")

cmd = ["ffmpeg", "-y"] + inputs + [
    "-filter_complex", ";".join(filters),
    "-map", "[out]", "-ar", "48000", "-ac", "2",
    f"{PILOTO}/vozes_multi.wav", "-loglevel", "error",
]
subprocess.run(cmd, check=True)
print("vozes_multi.wav OK")

# mix com o fundo (ja separado antes) e normaliza o volume
subprocess.run([
    "ffmpeg", "-y",
    "-i", f"{PILOTO}/separado_cena/htdemucs/clip_en_real_audio/no_vocals.wav",
    "-i", f"{PILOTO}/vozes_multi.wav",
    "-filter_complex",
    "[0:a]volume=1.0[bg];[1:a]volume=1.0[voz];[bg][voz]amix=inputs=2:duration=longest:dropout_transition=0[mix];"
    "[mix]loudnorm=I=-18:TP=-1.5:LRA=11[out]",
    "-map", "[out]", "-ar", "48000", "-ac", "2",
    f"{PILOTO}/trilha_final_v3.wav", "-loglevel", "error",
], check=True)
print("trilha_final_v3.wav OK")
