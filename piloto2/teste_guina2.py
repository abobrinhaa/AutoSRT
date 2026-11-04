import os
os.environ["COQUI_TOS_AGREED"] = "1"

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

from TTS.api import TTS
import subprocess

PILOTO = "/opt/autosrt-dub/piloto2"
ref = f"{PILOTO}/ref_speaker10_limpo.wav"

print("carregando XTTS-v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("modelo carregado")

frases = [
    "o tio Guina não para de comer o dia todo",
    "ontem o tio Guina comeu uma feijoada inteira sozinho",
    "o tio Guina jura que vai fazer dieta segunda-feira",
    "ninguém nunca viu o tio Guina recusar uma sobremesa",
]

arquivos = []
for i, texto in enumerate(frases):
    out = f"{PILOTO}/guina_{i}.wav"
    tts.tts_to_file(
        text=texto,
        speaker_wav=ref,
        language="pt",
        file_path=out,
        split_sentences=False,
        temperature=0.65,
    )
    print(f"gerado: {out}")
    arquivos.append(out)

subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                 "-t", "0.7", f"{PILOTO}/silence3.wav", "-loglevel", "error"], check=True)

concat_list = f"{PILOTO}/concat_guina.txt"
with open(concat_list, "w") as f:
    for fn in arquivos:
        f.write(f"file '{fn}'\n")
        f.write(f"file '{PILOTO}/silence3.wav'\n")

out_concat = f"{PILOTO}/guina_concat.wav"
subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, out_concat, "-loglevel", "error"], check=True)

dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1", out_concat],
                      capture_output=True, text=True).stdout.strip()
print(f"concatenado: {out_concat} ({dur}s)")

video_out = f"{PILOTO}/guina_video.mkv"
subprocess.run([
    "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=640x360:d={dur}",
    "-i", out_concat,
    "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
    "-shortest", video_out, "-loglevel", "error"
], check=True)
print(f"video: {video_out}")
