import json

with open("/opt/autosrt-dub/piloto2/filme_completo_cues.json", encoding="utf-8") as f:
    cues = json.load(f)

WINDOW_MS = 4 * 60 * 1000  # 4 minutos
STEP_MS = 30 * 1000

max_start = max(c["end"] for c in cues)
melhor = None
for win_start in range(0, int(max_start - WINDOW_MS), STEP_MS):
    win_end = win_start + WINDOW_MS
    falas = [c for c in cues if c["start"] >= win_start and c["end"] <= win_end]
    if not falas:
        continue
    speakers_no_trecho = set(c["speaker"] for c in falas)
    dur_speaker10 = sum(c["end"] - c["start"] for c in falas if c["speaker"] == "SPEAKER_10") / 1000
    total_falas = len(falas)
    score = dur_speaker10
    if melhor is None or score > melhor[0]:
        melhor = (score, win_start, win_end, total_falas, speakers_no_trecho, dur_speaker10)

score, win_start, win_end, total_falas, speakers, dur10 = melhor
print(f"melhor janela: {win_start/1000:.0f}s - {win_end/1000:.0f}s ({win_start/60000:.1f}min - {win_end/60000:.1f}min)")
print(f"total de falas: {total_falas}, speakers presentes: {speakers}")
print(f"segundos de SPEAKER_10 nessa janela: {dur10:.1f}s")
