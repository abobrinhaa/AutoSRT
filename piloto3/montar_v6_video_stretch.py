import json, subprocess, os

PILOTO = "/opt/autosrt-dub/piloto3"
CLIP_DURATION_MS = 150000
VIDEO_SRC = f"{PILOTO}/clip_en.mkv"
ORIGINAL_FULL_AUDIO = f"{PILOTO}/clip_en_audio.wav"
BACKGROUND = f"{PILOTO}/separado/htdemucs/clip_en_audio/no_vocals.wav"
BUFFER_MS = 250
MAX_VIDEO_SLOWDOWN = 1.6  # video nao fica mais que 1.6x mais lento

with open(f"{PILOTO}/cannon_traduzido.json", encoding="utf-8") as f:
    cues = json.load(f)
cues = [c for c in cues if os.path.exists(f"{PILOTO}/synth_sync/{c['index']}.wav")]
cues.sort(key=lambda c: c["start"])

os.makedirs(f"{PILOTO}/segmentos_v6", exist_ok=True)

def run(cmd):
    subprocess.run(cmd, check=True)

def extrair_audio(src, start_ms, end_ms, dst):
    run(["ffmpeg", "-y", "-i", src, "-ss", str(start_ms / 1000), "-to", str(end_ms / 1000),
         "-ar", "48000", "-ac", "2", dst, "-loglevel", "error"])

def clipe_normal(start_ms, end_ms, dst):
    """Video + audio original, sem nenhuma mudanca de velocidade (usado nas brechas)."""
    gap_ms = end_ms - start_ms
    if gap_ms <= 0:
        return None
    audio_tmp = f"{dst}.audio.wav"
    if gap_ms <= BUFFER_MS:
        extrair_audio(BACKGROUND, start_ms, end_ms, audio_tmp)
    else:
        head = f"{dst}.head.wav"; tail = f"{dst}.tail.wav"
        extrair_audio(BACKGROUND, start_ms, start_ms + BUFFER_MS, head)
        extrair_audio(ORIGINAL_FULL_AUDIO, start_ms + BUFFER_MS, end_ms, tail)
        lst = f"{dst}.concat.txt"
        with open(lst, "w") as f:
            f.write(f"file '{head}'\nfile '{tail}'\n")
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-ar", "48000", "-ac", "2", audio_tmp, "-loglevel", "error"])
        os.remove(head); os.remove(tail); os.remove(lst)
    run(["ffmpeg", "-y", "-ss", str(start_ms / 1000), "-to", str(end_ms / 1000), "-i", VIDEO_SRC,
         "-i", audio_tmp, "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac",
         dst, "-loglevel", "error"])
    os.remove(audio_tmp)
    return dst

def clipe_de_fala(c, dst):
    start, end = c["start"], c["end"]
    slot_ms = end - start
    voz_path = f"{PILOTO}/synth_sync/{c['index']}.wav"
    audio_dur_ms = int(c["final_duration"] * 1000)

    target_ms = audio_dur_ms if audio_dur_ms > slot_ms else slot_ms
    # limita o quanto o video pode esticar
    max_target_ms = int(slot_ms * MAX_VIDEO_SLOWDOWN)
    target_ms = min(target_ms, max_target_ms)
    cortou = audio_dur_ms > target_ms

    bg_tmp = f"{dst}.bg.wav"
    extrair_audio(BACKGROUND, start, start + target_ms, bg_tmp)
    mix_tmp = f"{dst}.mix.wav"
    run(["ffmpeg", "-y", "-i", bg_tmp, "-i", voz_path,
         "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[out]",
         "-map", "[out]", "-ar", "48000", "-ac", "2", mix_tmp, "-loglevel", "error"])
    audio_final = f"{dst}.audiofinal.wav"
    run(["ffmpeg", "-y", "-i", mix_tmp, "-af", f"apad,atrim=0:{target_ms/1000}",
         "-ar", "48000", "-ac", "2", audio_final, "-loglevel", "error"])
    os.remove(bg_tmp); os.remove(mix_tmp)

    speed_factor = slot_ms / target_ms  # <1 = video mais lento
    video_tmp = f"{dst}.video.mkv"
    if abs(speed_factor - 1.0) < 0.01:
        run(["ffmpeg", "-y", "-ss", str(start / 1000), "-to", str(end / 1000), "-i", VIDEO_SRC,
             "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an",
             video_tmp, "-loglevel", "error"])
    else:
        run(["ffmpeg", "-y", "-ss", str(start / 1000), "-to", str(end / 1000), "-i", VIDEO_SRC,
             "-vf", f"setpts=PTS/{speed_factor}",
             "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an",
             video_tmp, "-loglevel", "error"])

    run(["ffmpeg", "-y", "-i", video_tmp, "-i", audio_final, "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-shortest", dst, "-loglevel", "error"])
    os.remove(video_tmp); os.remove(audio_final)
    return cortou, target_ms

segmentos = []
cursor = 0
idx = 0
cortes = 0
esticou = 0
for c in cues:
    start, end = c["start"], c["end"]
    if start > cursor:
        gap_path = f"{PILOTO}/segmentos_v6/{idx:03d}_gap.mkv"
        if clipe_normal(cursor, start, gap_path):
            segmentos.append(gap_path)
            idx += 1
    fala_path = f"{PILOTO}/segmentos_v6/{idx:03d}_fala{c['index']}.mkv"
    cortou, target_ms = clipe_de_fala(c, fala_path)
    if target_ms > (end - start) * 1.02:
        esticou += 1
    if cortou:
        cortes += 1
        print(f"{c['index']}: ainda cortou mesmo esticando video ate o limite")
    segmentos.append(fala_path)
    idx += 1
    cursor = start + target_ms

if cursor < CLIP_DURATION_MS:
    tail_path = f"{PILOTO}/segmentos_v6/{idx:03d}_tail.mkv"
    if clipe_normal(cursor, CLIP_DURATION_MS, tail_path):
        segmentos.append(tail_path)

print(f"\n{esticou} falas tiveram o video esticado, {cortes} ainda cortaram mesmo assim")

concat_list = f"{PILOTO}/concat_v6.txt"
with open(concat_list, "w") as f:
    for s in segmentos:
        f.write(f"file '{s}'\n")

out = f"{PILOTO}/cannon_v6_final.mkv"
run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
     "-c:v", "libx264", "-preset", "fast", "-crf", "20",
     "-af", "loudnorm=I=-18:TP=-1.5:LRA=11", "-c:a", "aac",
     "-map_chapters", "-1",
     out, "-loglevel", "error"])

dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1", out],
                      capture_output=True, text=True).stdout.strip()
print(f"OK: {out} ({dur}s, original era {CLIP_DURATION_MS/1000}s)")
