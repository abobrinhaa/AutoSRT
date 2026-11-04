import json, subprocess, os

PILOTO = "/opt/autosrt-dub/piloto2"
CLIP_DURATION_MS = 240000
ORIGINAL_FULL = f"{PILOTO}/clip_en_real_audio.wav"  # ingles original, cena inteira
BACKGROUND = f"{PILOTO}/separado_cena/htdemucs/clip_en_real_audio/no_vocals.wav"

with open(f"{PILOTO}/legenda_pt_real_v3_sync.json", encoding="utf-8") as f:
    linhas = sorted(json.load(f), key=lambda l: l["start"])

os.makedirs(f"{PILOTO}/segmentos", exist_ok=True)

def extrair(src, start_ms, end_ms, dst):
    subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-ss", str(start_ms / 1000), "-to", str(end_ms / 1000),
        "-ar", "48000", "-ac", "2",
        dst, "-loglevel", "error"
    ], check=True)

def mixar_voz_fundo(voz_path, bg_start_ms, bg_end_ms, dst):
    bg_tmp = f"{dst}.bg.wav"
    extrair(BACKGROUND, bg_start_ms, bg_end_ms, bg_tmp)
    subprocess.run([
        "ffmpeg", "-y", "-i", bg_tmp, "-i", voz_path,
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[out]",
        "-map", "[out]",
        "-ar", "48000", "-ac", "2", dst, "-loglevel", "error"
    ], check=True)
    os.remove(bg_tmp)

segmentos = []
cursor = 0
idx_seg = 0

for linha in linhas:
    start, end = linha["start"], linha["end"]
    if start > cursor:
        # brecha: usa o audio original (ingles) completo
        gap_path = f"{PILOTO}/segmentos/{idx_seg:03d}_gap.wav"
        extrair(ORIGINAL_FULL, cursor, start, gap_path)
        segmentos.append(gap_path)
        idx_seg += 1

    voz_path = f"{PILOTO}/synth_sync/{linha['index']}.wav"
    final_dur_ms = int(linha.get("final_duration", linha["original_duration"]) * 1000)
    fala_end = start + max(final_dur_ms, end - start)
    seg_path = f"{PILOTO}/segmentos/{idx_seg:03d}_fala{linha['index']}.wav"
    mixar_voz_fundo(voz_path, start, fala_end, seg_path)
    segmentos.append(seg_path)
    idx_seg += 1
    cursor = fala_end

if cursor < CLIP_DURATION_MS:
    tail_path = f"{PILOTO}/segmentos/{idx_seg:03d}_tail.wav"
    extrair(ORIGINAL_FULL, cursor, CLIP_DURATION_MS, tail_path)
    segmentos.append(tail_path)

concat_list = f"{PILOTO}/concat_fallback.txt"
with open(concat_list, "w") as f:
    for s in segmentos:
        f.write(f"file '{s}'\n")

out = f"{PILOTO}/trilha_com_fallback.wav"
subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                 "-af", "loudnorm=I=-18:TP=-1.5:LRA=11",
                 "-ar", "48000", "-ac", "2", out, "-loglevel", "error"], check=True)

dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1", out],
                      capture_output=True, text=True).stdout.strip()
print(f"OK: {out} ({dur}s, {len(segmentos)} segmentos)")
