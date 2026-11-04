import json, subprocess, os

PILOTO = "/opt/autosrt-dub/piloto3"
AUDIO = f"{PILOTO}/clip_en_audio.wav"

with open(f"{PILOTO}/cannon_traduzido.json", encoding="utf-8") as f:
    cues = json.load(f)

# linha sem locutor (None) vira SPEAKER_00 por padrao
for c in cues:
    if not c["speaker"]:
        c["speaker"] = "SPEAKER_00"

speakers = sorted(set(c["speaker"] for c in cues))
os.makedirs(f"{PILOTO}/refs_bruto", exist_ok=True)

for spk in speakers:
    segs = [c for c in cues if c["speaker"] == spk]
    concat_list = f"{PILOTO}/refs_bruto/{spk}_concat.txt"
    parts = []
    for i, c in enumerate(segs):
        part_path = f"{PILOTO}/refs_bruto/{spk}_{i}.wav"
        subprocess.run(["ffmpeg", "-y", "-i", AUDIO,
                         "-ss", str(c["start"] / 1000), "-to", str(c["end"] / 1000),
                         part_path, "-loglevel", "error"], check=True)
        parts.append(part_path)
    with open(concat_list, "w") as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p)}'\n")
    ref_path = f"{PILOTO}/refs_bruto/{spk}_ref.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                     ref_path, "-loglevel", "error"], check=True)
    print(f"{spk}: {ref_path}")

with open(f"{PILOTO}/cannon_traduzido.json", "w", encoding="utf-8") as f:
    json.dump(cues, f, ensure_ascii=False, indent=2)
print("OK")
