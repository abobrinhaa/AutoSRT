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

with open(f"{PILOTO}/legenda_pt_real_recortada.json", encoding="utf-8") as f:
    linhas = json.load(f)
por_index = {l["index"]: l for l in linhas}

alvo = [671, 701, 704, 744]

print("carregando XTTS-v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("modelo carregado")

for idx in alvo:
    linha = por_index[idx]
    texto = linha["text"].replace("\n", " ").strip()
    texto = re.sub(r"[.]+$", "", texto)
    out = f"{PILOTO}/synth_real/{idx}.wav"
    melhor_dur = None
    for tentativa in range(4):
        tmp = f"{PILOTO}/synth_real/{idx}_try{tentativa}.wav"
        tts.tts_to_file(
            text=texto, speaker_wav=ref, language="pt", file_path=tmp,
            split_sentences=False, temperature=0.6,
        )
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "default=noprint_wrappers=1:nokey=1", tmp],
                                    capture_output=True, text=True).stdout.strip())
        ratio = dur / max(linha["original_duration"], 0.3)
        print(f"{idx} tentativa {tentativa}: {dur:.1f}s (ratio {ratio:.1f}x)")
        if ratio <= 2.0:
            melhor_dur = dur
            subprocess.run(["cp", tmp, out], check=True)
            break
    if melhor_dur is None:
        print(f"{idx}: nenhuma tentativa ficou boa, mantendo a original mesmo assim")
    else:
        linha["synth_duration"] = melhor_dur

with open(f"{PILOTO}/legenda_pt_real_recortada.json", "w", encoding="utf-8") as f:
    json.dump(linhas, f, ensure_ascii=False, indent=2)
print("OK")
