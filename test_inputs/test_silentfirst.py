"""Silent-first merge test (v2): reuse joined silent video if present, concat audio
properly (FLAC stream-copy breaks timestamps), trim to video length, lip-sync once."""
import sys, subprocess
from pathlib import Path
ROOT = "/workspace/alluvi-pipeline"; sys.path.insert(0, ROOT)
D = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
FM = f"{ROOT}/test_inputs/merge_tests/framematch.py"
M = Path(f"{ROOT}/test_inputs/merge_tests/silentfirst"); M.mkdir(parents=True, exist_ok=True)
def run(c): subprocess.run(c, check=True)
def dur(p): return float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(p)]).decode())
silents = [D/f"shot_{i}"/"silent.mp4" for i in range(1, N+1)]
for s in silents:
    if not s.exists(): print("MISSING:", s); sys.exit(1)
joined = M/"silent_matched.mp4"
if joined.exists():
    print(f"[silent-first] reusing existing joined silent video")
else:
    run(["ffmpeg","-y","-i",str(silents[0]),"-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-an",str(joined),"-loglevel","error"])
    for i in range(1, N):
        tmp = M/"_acc.mp4"
        run(["python", FM, str(joined), str(silents[i]), str(tmp), "1.0"])
        run(["mv","-f",str(tmp),str(joined)])
vdur = dur(joined); print(f"[silent-first] frame-matched silent video = {vdur:.2f}s")
ain = []
for i in range(1, N+1): ain += ["-i", f"{D.resolve()}/shot_{i}/audio.flac"]
afilt = "".join(f"[{k}:a]" for k in range(N)) + f"concat=n={N}:v=0:a=1[a]"
combined = M/"combined_audio.wav"
run(["ffmpeg","-y"] + ain + ["-filter_complex", afilt, "-map","[a]","-ar","16000","-ac","1",str(combined),"-loglevel","error"])
print(f"[silent-first] combined audio = {dur(combined):.2f}s")
atrim = M/"audio_trim.wav"
run(["ffmpeg","-y","-i",str(combined),"-t",f"{vdur:.3f}","-c","copy",str(atrim),"-loglevel","error"])
print(f"[silent-first] lip-syncing against first {vdur:.2f}s of audio")
from video_pipeline import step_6_lipsync as lip
res = lip.generate(lip.load_pipeline(), str(joined), str(atrim), M/"final_lipsynced", scene_id="silentfirst")
print(f"\n[silent-first] DONE -> {res.get('local_path')}")
