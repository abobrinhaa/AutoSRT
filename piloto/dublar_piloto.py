import sys, os, json, subprocess
sys.path.insert(0, "/opt/autosrt-dub")

from autosrt.transcribe import load_transcript

PILOTO = "/opt/autosrt-dub/piloto"
CLIP_WAV = f"{PILOTO}/clip_en.wav"
CLIP_DURATION_MS = 181848

en_cues = load_transcript(f"{PILOTO}/clip_en.srt")
with open(f"{PILOTO}/legenda_pt_recortada.json", encoding="utf-8") as f:
    pt_lines = json.load(f)

def overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))

for line in pt_lines:
    best_speaker, best_overlap = None, 0
    for c in en_cues:
        ov = overlap(line["start"], line["end"], c.start, c.end)
        if ov > best_overlap:
            best_overlap, best_speaker = ov, c.speaker
    line["speaker"] = best_speaker or "SPEAKER_00"

print("--- linhas com locutor atribuido ---")
for l in pt_lines:
    print(l["index"], l["speaker"], l["start"], l["end"], "|", l["text"])

# referencia de voz por locutor: concatena os trechos do audio original (ingles)
speakers = sorted(set(l["speaker"] for l in pt_lines))
ref_paths = {}
os.makedirs(f"{PILOTO}/refs", exist_ok=True)
for spk in speakers:
    segs = [c for c in en_cues if c.speaker == spk]
    if not segs:
        continue
    concat_list = f"{PILOTO}/refs/{spk}_concat.txt"
    parts = []
    for i, c in enumerate(segs):
        part_path = f"{PILOTO}/refs/{spk}_{i}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", CLIP_WAV,
            "-ss", str(c.start / 1000), "-to", str(c.end / 1000),
            part_path, "-loglevel", "error"
        ], check=True)
        parts.append(part_path)
    with open(concat_list, "w") as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p)}'\n")
    ref_path = f"{PILOTO}/refs/{spk}_ref.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
        ref_path, "-loglevel", "error"
    ], check=True)
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", ref_path],
        capture_output=True, text=True).stdout.strip()
    print(f"referencia {spk}: {ref_path} ({dur}s)")
    ref_paths[spk] = ref_path

with open(f"{PILOTO}/pt_lines_com_locutor.json", "w", encoding="utf-8") as f:
    json.dump(pt_lines, f, ensure_ascii=False, indent=2)

print("\nreferencias:", ref_paths)
print("OK - pronto para sintese XTTS")
