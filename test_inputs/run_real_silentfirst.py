import sys, datetime
from pathlib import Path
sys.path.insert(0, "/workspace/alluvi-pipeline")
from PIL import Image
from video_pipeline import step_4_tts_f5 as tts, step_5_video_wan as wan, step_6_lipsync as lip
from video_pipeline.silentfirst import build_silentfirst as bsf

num_beats   = int(sys.argv[1]) if len(sys.argv) > 1 else 8
shot_seconds= int(sys.argv[2]) if len(sys.argv) > 2 else 5
gender      = (sys.argv[3].lower() if len(sys.argv) > 3 else "f")
run_dir = Path("/workspace/alluvi-pipeline/outputs")/datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_silentfirst_real")
run_dir.mkdir(parents=True, exist_ok=True)

anchor = run_dir/"_anchor.jpg"
im = Image.open("/workspace/alluvi-pipeline/test_inputs/real.jpg").convert("RGB"); w,h = im.size
sc = max(768/w, 1344/h); nw,nh = int(w*sc), int(h*sc); im = im.resize((nw,nh), Image.LANCZOS)
l=(nw-768)//2; t=(nh-1344)//2; im.crop((l,t,l+768,t+1344)).save(anchor, quality=95)

account = {"id":0,"tiktok_id":"real_test","name":"Subject",
           "gender":"male" if gender.startswith("m") else "female",
           "country":"United States","language":"English","age":27}
output  = {"id":0,"persona_id":0,"scenario_id":"gym_post_workout_mirror_01",
           "final_image_local_path":str(anchor)}
handles = (tts.load_pipeline(), wan.load_pipeline(), lip.load_pipeline())
path = bsf.build_one(account, output, num_beats, 12345, run_dir, handles, target_seconds=shot_seconds)
print("\nDONE silentfirst ->", path)
tts.unload_pipeline(handles[0]); wan.unload_pipeline(handles[1]); lip.unload_pipeline(handles[2])
