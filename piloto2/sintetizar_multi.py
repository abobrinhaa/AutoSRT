import os, json, subprocess, re
os.environ["COQUI_TOS_AGREED"] = "1"

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

from TTS.api import TTS

PILOTO = "/opt/autosrt-dub/piloto2"
refs = {
    "SPEAKER_00": f"{PILOTO}/refs_cena/SPEAKER_00_ref.wav",
    "SPEAKER_01": f"{PILOTO}/refs_cena/SPEAKER_01_ref.wav",
    "SPEAKER_02": f"{PILOTO}/refs_cena/SPEAKER_02_ref.wav",
}
os.makedirs(f"{PILOTO}/synth_multi", exist_ok=True)

with open(f"{PILOTO}/legenda_pt_real_v2_locutores.json", encoding="utf-8") as f:
    linhas = json.load(f)

print("carregando XTTS-v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("modelo carregado")

for linha in linhas:
    ref = refs[linha["speaker"]]
    texto = linha["text"].replace("\n", " ").strip()
    texto = re.sub(r"[.]+$", "", texto)
    out = f"{PILOTO}/synth_multi/{linha['index']}.wav"
    tts.tts_to_file(
        text=texto, speaker_wav=ref, language="pt", file_path=out,
        split_sentences=False, temperature=0.65,
    )
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                           "-of", "default=noprint_wrappers=1:nokey=1", out],
                          capture_output=True, text=True).stdout.strip()
    linha["synth_duration"] = float(dur)
    print(f"{linha['index']} ({linha['speaker']}): {dur}s | {texto[:40]}")

with open(f"{PILOTO}/legenda_pt_real_v2_locutores.json", "w", encoding="utf-8") as f:
    json.dump(linhas, f, ensure_ascii=False, indent=2)
print("OK")
