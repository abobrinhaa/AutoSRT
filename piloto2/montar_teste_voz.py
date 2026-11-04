import subprocess

PILOTO = "/opt/autosrt-dub/piloto2"

subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                 "-t", "0.6", f"{PILOTO}/silence.wav", "-loglevel", "error"], check=True)

concat_list = f"{PILOTO}/concat_teste.txt"
files = ["teste_voz_0.wav", "silence.wav", "teste_voz_1.wav", "silence.wav", "teste_voz_2.wav"]
with open(concat_list, "w") as f:
    for fn in files:
        f.write(f"file '{PILOTO}/{fn}'\n")

out = f"{PILOTO}/teste_voz_concat.wav"
subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, out, "-loglevel", "error"], check=True)

dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1", out],
                      capture_output=True, text=True).stdout.strip()
print(f"concatenado: {out} ({dur}s)")

# video preto simples do mesmo tamanho
video_out = f"{PILOTO}/teste_voz_video.mkv"
subprocess.run([
    "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=640x360:d={dur}",
    "-i", out,
    "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
    "-shortest", video_out, "-loglevel", "error"
], check=True)
print(f"video: {video_out}")
