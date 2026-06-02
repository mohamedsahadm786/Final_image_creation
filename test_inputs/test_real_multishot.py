"""Scratch: 2-shot (5s each) multishot from ANY photo, then stitch -> ~10s.
Usage: python test_inputs/test_real_multishot.py <image> [m|f] [num_shots] [shot_seconds]
Reuses the multishot rulebook + step modules; no pipeline code touched, no DB needed."""
import sys, random
from pathlib import Path
sys.path.insert(0, "/workspace/alluvi-pipeline")
from PIL import Image
from video_pipeline.multishot import script_builder_multi, stitch as stitcher
from video_pipeline import step_4_tts_f5 as tts, step_5_video_wan as wan, step_6_lipsync as lip

img = sys.argv[1]
gender = (sys.argv[2].lower() if len(sys.argv) > 2 else "f")
num_shots = int(sys.argv[3]) if len(sys.argv) > 3 else 2
shot_seconds = int(sys.argv[4]) if len(sys.argv) > 4 else 5

voice = "voices_examples/male/male_01.wav" if gender.startswith("m") else "voices_examples/female/female_02.wav"
account = {"name": "Subject", "gender": "male" if gender.startswith("m") else "female",
           "country": "United States", "language": "English", "age": 30}
scenario = {"id": "real_photo_test", "category": "lifestyle",
            "location": "the setting shown in the photo", "mood": "calm confidence",
            "activity": "holding and presenting the product"}

out = Path("/workspace/alluvi-pipeline/test_inputs/out_multishot"); out.mkdir(parents=True, exist_ok=True)
resized = out / "_frame.jpg"
im = Image.open(img).convert("RGB"); w, h = im.size
sc = max(768 / w, 1344 / h); nw, nh = int(w * sc), int(h * sc)
im = im.resize((nw, nh), Image.LANCZOS).crop(((nw - 768) // 2, (nh - 1344) // 2,
                                               (nw - 768) // 2 + 768, (nh - 1344) // 2 + 1344))
im.save(resized, quality=95)

print(f"\n=== real-photo multishot: {num_shots} x {shot_seconds}s, voice={voice} ===")
script = script_builder_multi.build_multishot_script(account, scenario, num_shots=num_shots, target_seconds=shot_seconds)
shots = script["shots"]

th, wh, lh = tts.load_pipeline(), wan.load_pipeline(), lip.load_pipeline()
base_seed = random.randint(1, 2_000_000_000)
finals = []
for i, sh in enumerate(shots):
    tag = f"real#shot{i+1}"
    sd = out / f"shot_{i+1}"; sd.mkdir(parents=True, exist_ok=True)
    a = tts.generate(th, sh["dialogue"], sd / "audio", narrator_voice=voice, seed=base_seed, scene_id=tag)
    cap = int((shot_seconds + 3) * 16)
    nf = wan.frames_for_duration(a["audio_duration_seconds"], cap=cap)
    v = wan.generate(wh, str(resized), sd / "silent", motion_prompt=sh["wan_motion_prompt"],
                     negative_prompt=sh.get("wan_negative_prompt"), num_frames=nf,
                     seed=base_seed + 1 + i, scene_id=tag)
    f = lip.generate(lh, v["local_path"], a["local_path"], sd / "final", scene_id=tag)
    finals.append(f["local_path"])

final = out / "final.mp4"
stitcher.stitch(finals, final)
print(f"\nFINAL JOINED -> {final}")
