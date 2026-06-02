import sys, glob, json, copy, subprocess
sys.path.insert(0, "/workspace/alluvi-pipeline")
from pathlib import Path
from video_pipeline import step_6_lipsync as lip
_NULL = subprocess.DEVNULL
ORIG_WF = Path("/workspace/alluvi-pipeline/video_pipeline/workflows/latentsync_api.json")

scene = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else Path(sorted(glob.glob(
    "/workspace/alluvi-pipeline/outputs/*_silentfirst_real/*/*"))[-1])
combos = sys.argv[2] if len(sys.argv) > 2 else "2.0:20 2.0:30 2.0:40 2.5:40"
combos = [c for c in combos.replace(",", " ").split() if c]
start = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
dur   = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
print(f"scene : {scene}\ncombos: {combos}\nwindow: {start}s +{dur}s")

td = scene / "_lipsync_sweep"; td.mkdir(parents=True, exist_ok=True)
vslice = td / "slice.mp4"; aslice = td / "slice.wav"
if not (vslice.exists() and aslice.exists()):          # reuse if already sliced
    subprocess.run(["ffmpeg","-y","-ss",f"{start}","-i",str(scene/"silent_joined.mp4"),"-t",f"{dur}",
        "-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-an",str(vslice)],check=True,stdout=_NULL,stderr=_NULL)
    subprocess.run(["ffmpeg","-y","-ss",f"{start}","-i",str(scene/"full_audio.wav"),"-t",f"{dur}",
        "-ar","16000","-ac","1",str(aslice)],check=True,stdout=_NULL,stderr=_NULL)

base = json.loads(ORIG_WF.read_text())   # read-only
h = lip.load_pipeline()
for c in combos:
    expr, steps = c.split(":"); expr = float(expr); steps = int(steps)
    g = copy.deepcopy(base)
    for n in g.values():
        if n.get("class_type") == "LatentSyncNode":
            n["inputs"]["lips_expression"] = expr
            n["inputs"]["inference_steps"] = steps
    tag = f"e{str(expr).replace('.','p')}_s{steps}"     # dot-free -> .stem won't truncate
    tmp = td / f"wf_{tag}.json"; tmp.write_text(json.dumps(g))
    lip.WORKFLOW_PATH = tmp               # in-memory only; original untouched
    r = lip.generate(h, str(vslice), str(aslice), td / tag, seed=1247, scene_id=tag)
    print(f"  expr={expr} steps={steps} -> {r['local_path']}")
lip.WORKFLOW_PATH = ORIG_WF              # restore
lip.unload_pipeline(h)
print(f"\ncompare in: {td}")
