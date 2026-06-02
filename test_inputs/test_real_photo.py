"""Scratch isolation test: run one shot (TTS -> Wan -> LatentSync) from ANY photo.
Usage: python test_inputs/test_real_photo.py <image_path> [m|f]
Pipeline code is untouched; this just reuses the step modules on your chosen image."""
import sys
from pathlib import Path
sys.path.insert(0, "/workspace/alluvi-pipeline")
from PIL import Image
from video_pipeline import step_4_tts_f5 as tts, step_5_video_wan as wan, step_6_lipsync as lip

img = sys.argv[1]
g = (sys.argv[2].lower() if len(sys.argv) > 2 else "f")
voice = "voices_examples/male/male_01.wav" if g.startswith("m") else "voices_examples/female/female_02.wav"

# cover-resize to the pipeline's 768x1344 so Wan gets a clean portrait frame
out = Path("/workspace/alluvi-pipeline/test_inputs/out"); out.mkdir(parents=True, exist_ok=True)
resized = out / "_frame.jpg"
im = Image.open(img).convert("RGB"); w, h = im.size
sc = max(768 / w, 1344 / h); nw, nh = int(w * sc), int(h * sc)
im = im.resize((nw, nh), Image.LANCZOS).crop(((nw - 768) // 2, (nh - 1344) // 2,
                                               (nw - 768) // 2 + 768, (nh - 1344) // 2 + 1344))
im.save(resized, quality=95)

line = "Honestly, this has just become part of my everyday routine now."
motion = ("The person gives one small calm nod, then settles into a natural hold, breathing "
          "easily with a steady gaze toward the camera. The rest of the body stays relaxed and "
          "still, expression softening into a calm half-smile, with a single natural blink. "
          "Camera performs one gentle steady push-in at a normal pace. Calm natural human "
          "motion, the rest of the body still and relaxed, mouth closed, face stable, identity "
          "preserved, product label stable, cinematic realism.")

print(f"\n=== real-photo test ===\nimage: {img}\nvoice: {voice}\n")
th, wh, lh = tts.load_pipeline(), wan.load_pipeline(), lip.load_pipeline()
a = tts.generate(th, line, out / "audio", narrator_voice=voice, scene_id="realtest")
v = wan.generate(wh, str(resized), out / "silent", motion_prompt=motion,
                 audio_duration_seconds=a["audio_duration_seconds"], scene_id="realtest")
f = lip.generate(lh, v["local_path"], a["local_path"], out / "final", scene_id="realtest")
print(f"\nSILENT (motion only): {v['local_path']}\nFINAL (lip-synced):  {f['local_path']}")
