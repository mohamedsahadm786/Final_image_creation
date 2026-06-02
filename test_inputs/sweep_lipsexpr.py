import sys, glob, json, copy, subprocess
sys.path.insert(0, "/workspace/alluvi-pipeline")
from pathlib import Path
from video_pipeline import step_6_lipsync as lip

_NULL = subprocess.DEVNULL
ORIG_WF = Path("/workspace/alluvi-pipeline/video_pipeline/workflows/latentsync_api.json")

def _dur(p):
    return float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1",str(p)]).decode())

scene = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else Path(sorted(glob.glob(
    "/workspace/alluvi-pipeline/outputs/*_silentfirst_real/*/*"))[-1])
values = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1.5, 2.0, 2.5, 3.0]
start = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0      # skip the 2s intro silence
dur   = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
print(f"scene : {scene}\nvalues: {values}\nwindow: {start}s +{dur}s")

td = scene / "_lipexpr_test"; td.mkdir(parents=True, exist_ok=True)
vslice = td / "slice.mp4"; aslice = td / "slice.wav"
subprocess.run(["ffmpeg","-y","-ss",f"{start}","-i",str(scene/"silent_joined.mp4"),"-t",f"{dur}",
    "-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-an",str(vslice)],check=True,stdout=_NULL,stderr=_NULL)
subprocess.run(["ffmpeg","-y","-ss",f"{start}","-i",str(scene/"full_audio.wav"),"-t",f"{dur}",
    "-ar","16000","-ac","1",str(aslice)],check=True,stdout=_NULL,stderr=_NULL)

base = json.loads(ORIG_WF.read_text())   # read-only; never written back
h = lip.load_pipeline()
for v in values:
    g = copy.deepcopy(base)
    for n in g.values():
        if n.get("class_type") == "LatentSyncNode":
            n["inputs"]["lips_expression"] = v
    tmp_wf = td / f"wf_{v}.json"; tmp_wf.write_text(json.dumps(g))
    lip.WORKFLOW_PATH = tmp_wf            # in-memory override only (original file untouched)
    r = lip.generate(h, str(vslice), str(aslice), td / f"lipexpr_{v}", seed=1247, scene_id=f"expr{v}")
    print(f"  lips_expression={v} -> {r['local_path']}")
lip.WORKFLOW_PATH = ORIG_WF              # restore
lip.unload_pipeline(h)
print(f"\nCompare the clips in: {td}")
