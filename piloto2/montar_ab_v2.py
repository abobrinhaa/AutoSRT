import subprocess

PILOTO = "/opt/autosrt-dub/piloto2"

def run(cmd):
    subprocess.run(cmd, check=True)

# audio de cada clipe em wav puro
run(["ffmpeg", "-y", "-i", f"{PILOTO}/clipA_original.mkv", "-ar", "48000", "-ac", "2",
     f"{PILOTO}/clipA_audio.wav", "-loglevel", "error"])
run(["ffmpeg", "-y", "-i", f"{PILOTO}/clipB_dublado.mkv", "-ar", "48000", "-ac", "2",
     f"{PILOTO}/clipB_audio.wav", "-loglevel", "error"])

# concatena o audio em wav (sem problema de codec)
concat_list = f"{PILOTO}/concat_audio_ab.txt"
with open(concat_list, "w") as f:
    f.write(f"file '{PILOTO}/clipA_audio.wav'\n")
    f.write(f"file '{PILOTO}/clipB_audio.wav'\n")
run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
     f"{PILOTO}/ab_audio_full.wav", "-loglevel", "error"])

# concatena o video via filtro (reencoda os dois juntos, video puro sem audio)
run(["ffmpeg", "-y", "-i", f"{PILOTO}/clipA_original.mkv", "-i", f"{PILOTO}/clipB_dublado.mkv",
     "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
     "-map", "[v]", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
     f"{PILOTO}/ab_video_full.mkv", "-loglevel", "error"])

# junta video + audio final
run(["ffmpeg", "-y", "-i", f"{PILOTO}/ab_video_full.mkv", "-i", f"{PILOTO}/ab_audio_full.wav",
     "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
     "-shortest", f"{PILOTO}/comparacao_ab_v3.mkv", "-loglevel", "error"])

print("OK - comparacao_ab_v3.mkv gerado")
