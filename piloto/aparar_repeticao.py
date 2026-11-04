import json, re, subprocess, os

PILOTO = "/opt/autosrt-dub/piloto"

with open(f"{PILOTO}/pt_lines_com_locutor.json", encoding="utf-8") as f:
    pt_lines = json.load(f)

SIL_RE = re.compile(r"silence_start: ([\d.]+)")

os.makedirs(f"{PILOTO}/synth_trim", exist_ok=True)

for line in pt_lines:
    idx = line["index"]
    src = f"{PILOTO}/synth/{idx}.wav"
    dst = f"{PILOTO}/synth_trim/{idx}.wav"

    proc = subprocess.run(
        ["ffmpeg", "-i", src, "-af", "silencedetect=noise=-30dB:d=0.3", "-f", "null", "-"],
        capture_output=True, text=True)
    starts = [float(m) for m in SIL_RE.findall(proc.stderr)]

    total_dur = line["synth_duration"]
    original_dur = line["original_duration"]
    # corta no primeiro silencio que vem depois de pelo menos 0.3s de fala
    # (evita cortar um silencio inicial antes da fala comecar)
    cut_at = None
    for s in starts:
        if s > 0.3:
            cut_at = s
            break

    if cut_at and cut_at < total_dur * 0.95:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-t", str(cut_at), dst, "-loglevel", "error"],
            check=True)
        new_dur = cut_at
    else:
        subprocess.run(["cp", src, dst], check=True)
        new_dur = total_dur

    line["trim_duration"] = new_dur
    print(f"{idx}: {total_dur:.2f}s -> {new_dur:.2f}s (original {original_dur:.2f}s) | {line['text'][:40]}")

with open(f"{PILOTO}/pt_lines_com_locutor.json", "w", encoding="utf-8") as f:
    json.dump(pt_lines, f, ensure_ascii=False, indent=2)

print("OK - aparado")
