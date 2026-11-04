import json, subprocess

PILOTO = "/opt/autosrt-dub/piloto"
CLIP_DURATION_S = 181.848

with open(f"{PILOTO}/pt_lines_com_locutor.json", encoding="utf-8") as f:
    pt_lines = json.load(f)

inputs = []
filters = []
for i, line in enumerate(pt_lines):
    path = f"{PILOTO}/synth_trim/{line['index']}.wav"
    inputs += ["-i", path]
    delay_ms = int(line["start"])
    filters.append(f"[{i}:a]adelay={delay_ms}:all=1[a{i}]")

mix_inputs = "".join(f"[a{i}]" for i in range(len(pt_lines)))
filters.append(f"{mix_inputs}amix=inputs={len(pt_lines)}:duration=longest:dropout_transition=0[mixed]")
filters.append(f"[mixed]apad=whole_dur={CLIP_DURATION_S}[out]")

filter_complex = ";".join(filters)

cmd = ["ffmpeg", "-y"] + inputs + [
    "-filter_complex", filter_complex,
    "-map", "[out]",
    "-ar", "48000", "-ac", "2",
    f"{PILOTO}/trilha_pt_ia.wav",
    "-loglevel", "error",
]
print(" ".join(cmd[:10]), "...")
subprocess.run(cmd, check=True)
print("OK - trilha_pt_ia.wav gerada")
