import subprocess

PILOTO = "/opt/autosrt-dub/piloto2"
concat_list = f"{PILOTO}/concat_ab.txt"
with open(concat_list, "w") as f:
    f.write(f"file '{PILOTO}/clipA_original.mkv'\n")
    f.write(f"file '{PILOTO}/clipB_dublado.mkv'\n")

out = f"{PILOTO}/comparacao_ab.mkv"
subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                 "-c", "copy", out, "-loglevel", "error"], check=True)
print(f"gerado: {out}")
