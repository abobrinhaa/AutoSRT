import json, subprocess, os, csv

PILOTO = "/opt/autosrt-dub/piloto2"
BRUTO_DIR = f"{PILOTO}/speaker10_bruto"
VOCALS_LIMPO = f"{PILOTO}/separado/htdemucs/speaker10_bruto_concat/vocals.wav"
DATASET_DIR = "/opt/autosrt-dub/dataset_henry_fonda"
WAVS_DIR = f"{DATASET_DIR}/wavs"

os.makedirs(WAVS_DIR, exist_ok=True)

with open(f"{PILOTO}/filme_completo_cues.json", encoding="utf-8") as f:
    todas_cues = json.load(f)
speaker_cues = [c for c in todas_cues if c["speaker"] == "SPEAKER_10"]
print(f"{len(speaker_cues)} cues do SPEAKER_10")

def duracao(caminho):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", caminho],
        capture_output=True, text=True).stdout.strip())

linhas_metadata = []
cursor_s = 0.0
descartadas_longas = 0
descartadas_curtas = 0

for i, cue in enumerate(speaker_cues):
    bruto_path = f"{BRUTO_DIR}/{i:04d}.wav"
    if not os.path.exists(bruto_path):
        continue
    dur = duracao(bruto_path)
    inicio, fim = cursor_s, cursor_s + dur
    cursor_s = fim

    if dur > 11.0:
        descartadas_longas += 1
        continue
    if dur < 0.6:
        descartadas_curtas += 1
        continue

    texto = cue["text"].replace("\n", " ").strip()
    if not texto:
        continue

    nome_arquivo = f"fonda_{i:04d}"
    destino = f"{WAVS_DIR}/{nome_arquivo}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", VOCALS_LIMPO,
        "-ss", str(inicio), "-to", str(fim),
        "-ar", "22050", "-ac", "1",
        destino, "-loglevel", "error"
    ], check=True)
    linhas_metadata.append((nome_arquivo, texto, texto))

print(f"\n{len(linhas_metadata)} clipes de treino gerados")
print(f"descartadas (longas >11s): {descartadas_longas}, (curtas <0.6s): {descartadas_curtas}")

metadata_path = f"{DATASET_DIR}/metadata.csv"
with open(metadata_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="|")
    for linha in linhas_metadata:
        writer.writerow(linha)

print(f"salvo em {metadata_path}")
