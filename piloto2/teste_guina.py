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

texto = "o tio Guina não para de comer o dia todo"
out = f"{PILOTO}/teste_guina.wav"
tts.tts_to_file(
    text=texto,
    speaker_wav=ref,
    language="pt",
    file_path=out,
    split_sentences=False,
    temperature=0.65,
)
print(f"gerado: {out}")

import subprocess
dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1", out],
                      capture_output=True, text=True).stdout.strip()
video_out = f"{PILOTO}/teste_guina_video.mkv"
subprocess.run([
    "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=640x360:d={dur}",
    "-i", out,
    "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
    "-shortest", video_out, "-loglevel", "error"
], check=True)
print(f"video: {video_out} ({dur}s)")
