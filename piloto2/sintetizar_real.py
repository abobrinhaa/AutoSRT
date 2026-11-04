import os, json, subprocess, re
os.environ["COQUI_TOS_AGREED"] = "1"

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

from TTS.api import TTS

PILOTO = "/opt/autosrt-dub/piloto2"
ref = f"{PILOTO}/ref_speaker10_limpo.wav"
os.makedirs(f"{PILOTO}/synth_real", exist_ok=True)

with open(f"{PILOTO}/legenda_pt_real_recortada.json", encoding="utf-8") as f:
    linhas = json.load(f)

print("carregando XTTS-v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("modelo carregado")

for linha in linhas:
    texto = linha["text"].replace("\n", " ").strip()
    texto = re.sub(r"[.]+$", "", texto)  # tira ponto final
    out = f"{PILOTO}/synth_real/{linha['index']}.wav"
    tts.tts_to_file(
        text=texto,
        speaker_wav=ref,
        language="pt",
        file_path=out,
        split_sentences=False,
        temperature=0.65,
    )
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                           "-of", "default=noprint_wrappers=1:nokey=1", out],
                          capture_output=True, text=True).stdout.strip()
    original_dur = (linha["end"] - linha["start"]) / 1000
    linha["synth_duration"] = float(dur)
    linha["original_duration"] = original_dur
    print(f"{linha['index']}: gerado {dur}s (original {original_dur:.2f}s) | {texto[:40]}")

with open(f"{PILOTO}/legenda_pt_real_recortada.json", "w", encoding="utf-8") as f:
    json.dump(linhas, f, ensure_ascii=False, indent=2)

print("OK")
