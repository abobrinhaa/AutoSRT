import os, json, subprocess, re
os.environ["COQUI_TOS_AGREED"] = "1"

PILOTO = "/opt/autosrt-dub/piloto3"
with open(f"{PILOTO}/cannon_traduzido.json", encoding="utf-8") as f:
    cues = json.load(f)

outliers = [c for c in cues if c.get("synth_duration", 0) / max(c.get("original_duration", 0.3), 0.3) > 2.2]
print(f"{len(outliers)} outliers de {len(cues)}")
for c in outliers:
    print(c["index"], c["speaker"], c["synth_duration"], c["original_duration"], c["text"][:30])

if outliers:
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
    from TTS.api import TTS
    print("carregando XTTS pra regenerar outliers...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    refs = {spk: f"{PILOTO}/refs_limpas/{spk}_ref.wav" for spk in set(c["speaker"] for c in cues)}
    for c in outliers:
        texto = re.sub(r"[.]+$", "", c["text"].replace("\n", " ").strip())
        out = f"{PILOTO}/synth/{c['index']}.wav"
        for tentativa in range(4):
            tmp = f"{PILOTO}/synth/{c['index']}_try{tentativa}.wav"
            tts.tts_to_file(text=texto, speaker_wav=refs[c["speaker"]], language="pt", file_path=tmp,
                             split_sentences=False, temperature=0.6)
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "default=noprint_wrappers=1:nokey=1", tmp],
                                        capture_output=True, text=True).stdout.strip())
            ratio = dur / max(c["original_duration"], 0.3)
            print(f"  {c['index']} tentativa {tentativa}: {dur:.2f}s (ratio {ratio:.2f})")
            if ratio <= 2.0:
                subprocess.run(["cp", tmp, out], check=True)
                c["synth_duration"] = dur
                break

# ajuste de velocidade pra todos
MAX_SPEEDUP = 1.6
os.makedirs(f"{PILOTO}/synth_sync", exist_ok=True)
ajustadas = 0
for c in cues:
    src = f"{PILOTO}/synth/{c['index']}.wav"
    dst = f"{PILOTO}/synth_sync/{c['index']}.wav"
    if not os.path.exists(src):
        continue
    synth_dur = c.get("synth_duration", 0)
    original_dur = c.get("original_duration", 0.3)
    if synth_dur <= original_dur * 1.05:
        subprocess.run(["cp", src, dst], check=True)
        c["final_duration"] = synth_dur
        continue
    tempo = min(synth_dur / original_dur, MAX_SPEEDUP)
    subprocess.run(["ffmpeg", "-y", "-i", src, "-af", f"atempo={tempo:.4f}", dst, "-loglevel", "error"], check=True)
    nova_dur = synth_dur / tempo
    c["final_duration"] = nova_dur
    ajustadas += 1

print(f"{ajustadas} falas ajustadas de velocidade")
with open(f"{PILOTO}/cannon_traduzido.json", "w", encoding="utf-8") as f:
    json.dump(cues, f, ensure_ascii=False, indent=2)
print("OK")
