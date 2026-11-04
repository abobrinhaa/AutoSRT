import json, subprocess, os

PILOTO = "/opt/autosrt-dub/piloto2"
AUDIO = f"{PILOTO}/filme_completo_en.wav"

with open(f"{PILOTO}/filme_completo_cues.json", encoding="utf-8") as f:
    cues = json.load(f)

speaker_cues = [c for c in cues if c["speaker"] == "SPEAKER_10"]
print(f"{len(speaker_cues)} falas do SPEAKER_10")

os.makedirs(f"{PILOTO}/speaker10_bruto", exist_ok=True)
concat_list = f"{PILOTO}/speaker10_concat.txt"
parts = []
for i, c in enumerate(speaker_cues):
    part_path = f"{PILOTO}/speaker10_bruto/{i:04d}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", AUDIO,
        "-ss", str(c["start"] / 1000), "-to", str(c["end"] / 1000),
        part_path, "-loglevel", "error"
    ], check=True)
    parts.append(part_path)

with open(concat_list, "w") as f:
    for p in parts:
        f.write(f"file '{os.path.abspath(p)}'\n")

out_path = f"{PILOTO}/speaker10_bruto_concat.wav"
subprocess.run([
    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
    out_path, "-loglevel", "error"
], check=True)

dur = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "default=noprint_wrappers=1:nokey=1", out_path],
    capture_output=True, text=True).stdout.strip()
print(f"concatenado: {out_path} ({dur}s)")
