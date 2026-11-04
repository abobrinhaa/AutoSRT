import subprocess

PILOTO = "/opt/autosrt-dub/piloto2"

subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                 "-t", "0.8", f"{PILOTO}/silence2.wav", "-loglevel", "error"], check=True)

files = []
for i in range(3):
    for t in range(2):
        files.append(f"teste_v2_{i}_{t}.wav")
        files.append("silence2.wav")

concat_list = f"{PILOTO}/concat_v2.txt"
with open(concat_list, "w") as f:
    for fn in files:
        f.write(f"file '{PILOTO}/{fn}'\n")

out = f"{PILOTO}/teste_v2_concat.wav"
subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, out, "-loglevel", "error"], check=True)

dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1", out],
                      capture_output=True, text=True).stdout.strip()
print(f"concatenado: {out} ({dur}s)")

video_out = f"{PILOTO}/teste_v2_video.mkv"
subprocess.run([
    "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=640x360:d={dur}",
    "-i", out,
    "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
    "-shortest", video_out, "-loglevel", "error"
], check=True)
print(f"video: {video_out}")
