import os
os.environ["COQUI_TOS_AGREED"] = "1"

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

from TTS.api import TTS

PILOTO = "/opt/autosrt-dub/piloto2"
ref = f"{PILOTO}/ref_speaker10_limpo.wav"

print("carregando XTTS-v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("modelo carregado")

frases = [
    "Senhoras e senhores, eu tenho uma dúvida razoável sobre esse caso.",
    "Não podemos condenar um homem só porque todos concordam rápido demais.",
    "Vamos conversar sobre isso com calma antes de decidir.",
]

for i, texto in enumerate(frases):
    out = f"{PILOTO}/teste_voz_{i}.wav"
    tts.tts_to_file(text=texto, speaker_wav=ref, language="pt", file_path=out)
    print(f"gerado: {out}")

print("OK")
