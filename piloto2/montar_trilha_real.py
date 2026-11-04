import json, subprocess

PILOTO = "/opt/autosrt-dub/piloto2"
CLIP_DURATION_S = 240.0

with open(f"{PILOTO}/legenda_pt_real_recortada.json", encoding="utf-8") as f:
    linhas = json.load(f)

inputs = []
filters = []
for i, linha in enumerate(linhas):
    path = f"{PILOTO}/synth_real/{linha['index']}.wav"
    inputs += ["-i", path]
    delay_ms = int(linha["start"])
    filters.append(f"[{i}:a]adelay={delay_ms}:all=1[a{i}]")

mix_inputs = "".join(f"[a{i}]" for i in range(len(linhas)))
filters.append(f"{mix_inputs}amix=inputs={len(linhas)}:duration=longest:dropout_transition=0[mixed]")
filters.append(f"[mixed]apad=whole_dur={CLIP_DURATION_S}[out]")

filter_complex = ";".join(filters)

cmd = ["ffmpeg", "-y"] + inputs + [
    "-filter_complex", filter_complex,
    "-map", "[out]",
    "-ar", "48000", "-ac", "2",
    f"{PILOTO}/trilha_pt_ia_real.wav",
    "-loglevel", "error",
]
subprocess.run(cmd, check=True)
print("OK - trilha_pt_ia_real.wav gerada")
