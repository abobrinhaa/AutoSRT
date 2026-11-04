import os
os.environ["COQUI_TOS_AGREED"] = "1"
import json, subprocess, sys

PILOTO = "/opt/autosrt-dub/piloto"
os.makedirs(f"{PILOTO}/synth", exist_ok=True)

with open(f"{PILOTO}/pt_lines_com_locutor.json", encoding="utf-8") as f:
    pt_lines = json.load(f)

ref_paths = {
    "SPEAKER_00": f"{PILOTO}/refs/SPEAKER_00_ref.wav",
    "SPEAKER_01": f"{PILOTO}/refs/SPEAKER_01_ref.wav",
    "SPEAKER_02": f"{PILOTO}/refs/SPEAKER_02_ref.wav",
}

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

from TTS.api import TTS
print("carregando XTTS-v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("modelo carregado")

for line in pt_lines:
    spk = line["speaker"]
    ref = ref_paths.get(spk)
    out_path = f"{PILOTO}/synth/{line['index']}.wav"
    print(f"sintetizando {line['index']} ({spk}): {line['text'][:50]!r}")
    tts.tts_to_file(
        text=line["text"],
        speaker_wav=ref,
        language="pt",
        file_path=out_path,
    )
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", out_path],
        capture_output=True, text=True).stdout.strip()
    original_dur = (line["end"] - line["start"]) / 1000
    line["synth_duration"] = float(dur)
    line["original_duration"] = original_dur
    print(f"  -> gerado {dur}s (original {original_dur}s)")

with open(f"{PILOTO}/pt_lines_com_locutor.json", "w", encoding="utf-8") as f:
    json.dump(pt_lines, f, ensure_ascii=False, indent=2)

print("OK - sintese concluida")
