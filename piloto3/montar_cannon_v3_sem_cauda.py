import json, subprocess, os

PILOTO = "/opt/autosrt-dub/piloto3"
CLIP_DURATION_MS = 150000
ORIGINAL_FULL = f"{PILOTO}/clip_en_audio.wav"
BACKGROUND = f"{PILOTO}/separado/htdemucs/clip_en_audio/no_vocals.wav"
BUFFER_MS = 250  # mascara o resto da palavra em ingles que ainda toca logo apos o fim da legenda

with open(f"{PILOTO}/cannon_traduzido.json", encoding="utf-8") as f:
    cues = sorted(json.load(f), key=lambda c: c["start"])

os.makedirs(f"{PILOTO}/segmentos_v3", exist_ok=True)

def extrair(src, start_ms, end_ms, dst):
    subprocess.run(["ffmpeg", "-y", "-i", src, "-ss", str(start_ms / 1000), "-to", str(end_ms / 1000),
                     "-ar", "48000", "-ac", "2", dst, "-loglevel", "error"], check=True)

def segmento_de_fala_tamanho_fixo(voz_path, start_ms, end_ms, dst):
    slot_ms = end_ms - start_ms
    bg_tmp = f"{dst}.bg.wav"
    extrair(BACKGROUND, start_ms, end_ms, bg_tmp)
    mix_tmp = f"{dst}.mix.wav"
    subprocess.run(["ffmpeg", "-y", "-i", bg_tmp, "-i", voz_path,
                     "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[out]",
                     "-map", "[out]", "-ar", "48000", "-ac", "2", mix_tmp, "-loglevel", "error"], check=True)
    subprocess.run(["ffmpeg", "-y", "-i", mix_tmp,
                     "-af", f"apad,atrim=0:{slot_ms/1000}",
                     "-ar", "48000", "-ac", "2", dst, "-loglevel", "error"], check=True)
    os.remove(bg_tmp)
    os.remove(mix_tmp)

def segmento_de_brecha(start_ms, end_ms, dst):
    """Brecha em ingles, mas com um colchao de fundo (sem voz) logo no inicio
    pra nao pegar o resto de uma palavra que ainda tava tocando."""
    gap_ms = end_ms - start_ms
    if gap_ms <= BUFFER_MS:
        # brecha curta demais pro colchao, so toca fundo (sem voz) inteiro, mais seguro
        extrair(BACKGROUND, start_ms, end_ms, dst)
        return
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
    os.remove(head_path)
    os.remove(tail_path)
    os.remove(concat_list)

segmentos = []
cursor = 0
idx_seg = 0
for c in cues:
    start, end = c["start"], c["end"]
    voz_path = f"{PILOTO}/synth_sync/{c['index']}.wav"
    if not os.path.exists(voz_path):
        continue
    if start > cursor:
        gap_path = f"{PILOTO}/segmentos_v3/{idx_seg:03d}_gap.wav"
        segmento_de_brecha(cursor, start, gap_path)
        segmentos.append(gap_path)
        idx_seg += 1
    seg_path = f"{PILOTO}/segmentos_v3/{idx_seg:03d}_fala{c['index']}.wav"
    segmento_de_fala_tamanho_fixo(voz_path, start, end, seg_path)
    segmentos.append(seg_path)
    idx_seg += 1
    cursor = end

if cursor < CLIP_DURATION_MS:
    tail_path = f"{PILOTO}/segmentos_v3/{idx_seg:03d}_tail.wav"
    segmento_de_brecha(cursor, CLIP_DURATION_MS, tail_path)
    segmentos.append(tail_path)

concat_list = f"{PILOTO}/concat_v3.txt"
with open(concat_list, "w") as f:
    for s in segmentos:
        f.write(f"file '{s}'\n")

out = f"{PILOTO}/trilha_v3.wav"
subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                 "-af", "loudnorm=I=-18:TP=-1.5:LRA=11",
                 "-ar", "48000", "-ac", "2", out, "-loglevel", "error"], check=True)

dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1", out],
                      capture_output=True, text=True).stdout.strip()
print(f"OK: {out} ({dur}s, esperado {CLIP_DURATION_MS/1000}s)")
