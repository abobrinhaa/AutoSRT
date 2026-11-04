import os, json, subprocess, re
os.environ["COQUI_TOS_AGREED"] = "1"

PILOTO = "/opt/autosrt-dub/piloto3"
with open(f"{PILOTO}/cannon_traduzido.json", encoding="utf-8") as f:
    cues = json.load(f)

refs = {spk: f"{PILOTO}/refs_limpas/{spk}_ref.wav" for spk in set(c["speaker"] for c in cues)}

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
from TTS.api import TTS

print("carregando XTTS-v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("modelo carregado")

os.makedirs(f"{PILOTO}/synth_curto", exist_ok=True)
for c in cues:
    texto = (c.get("text_curto") or c["text"]).replace("\n", " ").strip()
    texto = re.sub(r"[.]+$", "", texto)
    if not texto:
        continue
    out = f"{PILOTO}/synth_curto/{c['index']}.wav"
    tts.tts_to_file(text=texto, speaker_wav=refs[c["speaker"]], language="pt",
                     file_path=out, split_sentences=False, temperature=0.65)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "default=noprint_wrappers=1:nokey=1", out],
                                capture_output=True, text=True).stdout.strip())
    c["synth_duration_curto"] = dur
    ratio = dur / max(c["original_duration"], 0.3)
    print(f"{c['index']}: {dur:.2f}s (orig {c['original_duration']:.2f}s, ratio {ratio:.2f}) | {texto}")

with open(f"{PILOTO}/cannon_traduzido.json", "w", encoding="utf-8") as f:
    json.dump(cues, f, ensure_ascii=False, indent=2)
print("OK")
