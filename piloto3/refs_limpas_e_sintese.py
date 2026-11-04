import os, json, subprocess, re
os.environ["COQUI_TOS_AGREED"] = "1"

PILOTO = "/opt/autosrt-dub/piloto3"
VOCALS = f"{PILOTO}/separado/htdemucs/clip_en_audio/vocals.wav"

with open(f"{PILOTO}/cannon_traduzido.json", encoding="utf-8") as f:
    cues = json.load(f)

speakers = sorted(set(c["speaker"] for c in cues))
os.makedirs(f"{PILOTO}/refs_limpas", exist_ok=True)
refs = {}
for spk in speakers:
    segs = [c for c in cues if c["speaker"] == spk]
    concat_list = f"{PILOTO}/refs_limpas/{spk}_concat.txt"
    parts = []
    for i, c in enumerate(segs):
        part_path = f"{PILOTO}/refs_limpas/{spk}_{i}.wav"
        subprocess.run(["ffmpeg", "-y", "-i", VOCALS,
                         "-ss", str(c["start"] / 1000), "-to", str(c["end"] / 1000),
                         part_path, "-loglevel", "error"], check=True)
        parts.append(part_path)
    with open(concat_list, "w") as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p)}'\n")
    ref_path = f"{PILOTO}/refs_limpas/{spk}_ref.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                     ref_path, "-loglevel", "error"], check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                           "-of", "default=noprint_wrappers=1:nokey=1", ref_path],
                          capture_output=True, text=True).stdout.strip()
    print(f"referencia {spk}: {dur}s")
    refs[spk] = ref_path

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
from TTS.api import TTS

print("carregando XTTS-v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("modelo carregado")

os.makedirs(f"{PILOTO}/synth", exist_ok=True)
for c in cues:
    texto = c["text"].replace("\n", " ").strip()
    texto = re.sub(r"[.]+$", "", texto)
    if not texto:
        continue
    out = f"{PILOTO}/synth/{c['index']}.wav"
    tts.tts_to_file(text=texto, speaker_wav=refs[c["speaker"]], language="pt",
                     file_path=out, split_sentences=False, temperature=0.65)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "default=noprint_wrappers=1:nokey=1", out],
                                capture_output=True, text=True).stdout.strip())
    c["synth_duration"] = dur
    c["original_duration"] = (c["end"] - c["start"]) / 1000
    print(f"{c['index']} ({c['speaker']}): {dur:.2f}s (orig {c['original_duration']:.2f}s) | {texto[:40]}")

with open(f"{PILOTO}/cannon_traduzido.json", "w", encoding="utf-8") as f:
    json.dump(cues, f, ensure_ascii=False, indent=2)
print("OK")
