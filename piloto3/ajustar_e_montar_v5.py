import json, subprocess, os

PILOTO = "/opt/autosrt-dub/piloto3"
CLIP_DURATION_MS = 150000
ORIGINAL_FULL = f"{PILOTO}/clip_en_audio.wav"
BACKGROUND = f"{PILOTO}/separado/htdemucs/clip_en_audio/no_vocals.wav"
BUFFER_MS = 250
MAX_SPEEDUP = 1.6

with open(f"{PILOTO}/cannon_traduzido.json", encoding="utf-8") as f:
    cues = json.load(f)
cues = [c for c in cues if os.path.exists(f"{PILOTO}/synth_curto/{c['index']}.wav")]
cues.sort(key=lambda c: c["start"])

# ajuste de velocidade (ate 1.6x) em cima do synth_curto
os.makedirs(f"{PILOTO}/synth_curto_sync", exist_ok=True)
for c in cues:
    src = f"{PILOTO}/synth_curto/{c['index']}.wav"
    dst = f"{PILOTO}/synth_curto_sync/{c['index']}.wav"
    synth_dur = c["synth_duration_curto"]
    original_dur = c["original_duration"]
    if synth_dur <= original_dur * 1.05:
        subprocess.run(["cp", src, dst], check=True)
        c["final_duration_curto"] = synth_dur
        continue
    tempo = min(synth_dur / original_dur, MAX_SPEEDUP)
    subprocess.run(["ffmpeg", "-y", "-i", src, "-af", f"atempo={tempo:.4f}", dst, "-loglevel", "error"], check=True)
    c["final_duration_curto"] = synth_dur / tempo

with open(f"{PILOTO}/cannon_traduzido.json", "w", encoding="utf-8") as f:
    json.dump(cues, f, ensure_ascii=False, indent=2)
print("velocidade ajustada")

# montagem com overflow controlado (usa o espaco livre ate a proxima fala comecar)
os.makedirs(f"{PILOTO}/segmentos_v5", exist_ok=True)

def extrair(src, start_ms, end_ms, dst):
    subprocess.run(["ffmpeg", "-y", "-i", src, "-ss", str(start_ms / 1000), "-to", str(end_ms / 1000),
                     "-ar", "48000", "-ac", "2", dst, "-loglevel", "error"], check=True)

def segmento_de_fala(voz_path, start_ms, slot_ms, dst):
    bg_tmp = f"{dst}.bg.wav"
    extrair(BACKGROUND, start_ms, start_ms + slot_ms, bg_tmp)
    mix_tmp = f"{dst}.mix.wav"
    subprocess.run(["ffmpeg", "-y", "-i", bg_tmp, "-i", voz_path,
                     "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[out]",
                     "-map", "[out]", "-ar", "48000", "-ac", "2", mix_tmp, "-loglevel", "error"], check=True)
    subprocess.run(["ffmpeg", "-y", "-i", mix_tmp, "-af", f"apad,atrim=0:{slot_ms/1000}",
                     "-ar", "48000", "-ac", "2", dst, "-loglevel", "error"], check=True)
    os.remove(bg_tmp); os.remove(mix_tmp)

def segmento_de_brecha(start_ms, end_ms, dst):
    gap_ms = end_ms - start_ms
    if gap_ms <= 0:
        return None
    if gap_ms <= BUFFER_MS:
        extrair(BACKGROUND, start_ms, end_ms, dst)
        return dst
    head_path = f"{dst}.head.wav"
    tail_path = f"{dst}.tail.wav"
    extrair(BACKGROUND, start_ms, start_ms + BUFFER_MS, head_path)
    extrair(ORIGINAL_FULL, start_ms + BUFFER_MS, end_ms, tail_path)
    concat_list = f"{dst}.concat.txt"
    with open(concat_list, "w") as f:
        f.write(f"file '{head_path}'\n")
        f.write(f"file '{tail_path}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                     "-ar", "48000", "-ac", "2", dst, "-loglevel", "error"], check=True)
    os.remove(head_path); os.remove(tail_path); os.remove(concat_list)
    return dst

segmentos = []
cursor = 0
idx_seg = 0
cortes_forcados = 0
for i, c in enumerate(cues):
    start, end = c["start"], c["end"]
    voz_path = f"{PILOTO}/synth_curto_sync/{c['index']}.wav"
    final_dur_ms = int(c["final_duration_curto"] * 1000)

    if start > cursor:
        gap_path = f"{PILOTO}/segmentos_v5/{idx_seg:03d}_gap.wav"
        if segmento_de_brecha(cursor, start, gap_path):
            segmentos.append(gap_path)
            idx_seg += 1

    proxima_start = cues[i + 1]["start"] if i + 1 < len(cues) else CLIP_DURATION_MS
    espaco_disponivel = proxima_start - start
    slot_usado = min(max(final_dur_ms, end - start), espaco_disponivel)
    if final_dur_ms > slot_usado:
        cortes_forcados += 1

    seg_path = f"{PILOTO}/segmentos_v5/{idx_seg:03d}_fala{c['index']}.wav"
    segmento_de_fala(voz_path, start, slot_usado, seg_path)
    segmentos.append(seg_path)
    idx_seg += 1
    cursor = start + slot_usado

if cursor < CLIP_DURATION_MS:
    tail_path = f"{PILOTO}/segmentos_v5/{idx_seg:03d}_tail.wav"
    if segmento_de_brecha(cursor, CLIP_DURATION_MS, tail_path):
        segmentos.append(tail_path)

print(f"{cortes_forcados} de {len(cues)} ainda cortaram mesmo usando o espaco todo disponivel")

concat_list = f"{PILOTO}/concat_v5.txt"
with open(concat_list, "w") as f:
    for s in segmentos:
        f.write(f"file '{s}'\n")

out = f"{PILOTO}/trilha_v5.wav"
subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                 "-af", "loudnorm=I=-18:TP=-1.5:LRA=11",
                 "-ar", "48000", "-ac", "2", out, "-loglevel", "error"], check=True)
dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1", out],
                      capture_output=True, text=True).stdout.strip()
print(f"OK: {out} ({dur}s)")
