
import sys, glob, json, copy, subprocess

sys.path.insert(0, "/workspace/alluvi-pipeline")

from pathlib import Path

from video_pipeline import step_6_lipsync as lip

_NULL = subprocess.DEVNULL

ORIG_WF = Path("/workspace/alluvi-pipeline/video_pipeline/workflows/latentsync_api.json")

def _dur(p):

    return float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration",

        "-of","default=noprint_wrappers=1:nokey=1",str(p)]).decode())

def lipsync_segmented(h, video, audio, out_path, chunk, tag):

    out_path = Path(out_path); A = _dur(audio)

    seg = out_path.parent/f"_seg_{tag}"; seg.mkdir(parents=True, exist_ok=True)

    conf = seg/"conf.mp4"

    subprocess.run(["ffmpeg","-y","-i",str(video),"-t",f"{A:.3f}","-c:v","libx264","-crf","18",

        "-pix_fmt","yuv420p","-an",str(conf)],check=True,stdout=_NULL,stderr=_NULL)

    n=max(1,int(A/chunk+0.999)); b=[i*A/n for i in range(n+1)]; b[-1]=A; parts=[]

    for i in range(n):

        s,d=b[i],b[i+1]-b[i]; vc=seg/f"v{i}.mp4"; ac=seg/f"a{i}.wav"

        subprocess.run(["ffmpeg","-y","-ss",f"{s:.3f}","-i",str(conf),"-t",f"{d:.3f}","-c:v","libx264",

            "-crf","18","-pix_fmt","yuv420p","-an",str(vc)],check=True,stdout=_NULL,stderr=_NULL)

        subprocess.run(["ffmpeg","-y","-ss",f"{s:.3f}","-i",str(audio),"-t",f"{d:.3f}",

            "-ar","16000","-ac","1",str(ac)],check=True,stdout=_NULL,stderr=_NULL)

        r=lip.generate(h,str(vc),str(ac),seg/f"f{i}",seed=1247,scene_id=f"{tag}#{i+1}")

        parts.append(r["local_path"]); print(f"    seg {i+1}/{n} ok")

    lst=seg/"list.txt"; lst.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in parts))

    rc=subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(lst),"-c","copy",

        str(out_path)],stdout=_NULL,stderr=_NULL).returncode

    if rc!=0:

        subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(lst),"-c:v","libx264",

            "-crf","18","-pix_fmt","yuv420p","-c:a","aac","-b:a","192k",str(out_path)],

            check=True,stdout=_NULL,stderr=_NULL)

    return str(out_path)

scene = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else Path(sorted(glob.glob(

    "/workspace/alluvi-pipeline/outputs/*_silentfirst_real/*/*"))[-1])

combos = sys.argv[2] if len(sys.argv) > 2 else "2.0:40 2.5:40"

combos = [c for c in combos.replace(",", " ").split() if c]

chunk  = float(sys.argv[3]) if len(sys.argv) > 3 else 7.0

print(f"scene : {scene}\ncombos: {combos}")

outdir = scene/"_full_sweep"; outdir.mkdir(parents=True, exist_ok=True)

base = json.loads(ORIG_WF.read_text())   # read-only

h = lip.load_pipeline()

for c in combos:

    expr, steps = c.split(":"); expr=float(expr); steps=int(steps)

    g = copy.deepcopy(base)

    for nn in g.values():

        if nn.get("class_type") == "LatentSyncNode":

            nn["inputs"]["lips_expression"]=expr; nn["inputs"]["inference_steps"]=steps

    tag = f"e{str(expr).replace('.','p')}_s{steps}"

    tmp = outdir/f"wf_{tag}.json"; tmp.write_text(json.dumps(g)); lip.WORKFLOW_PATH = tmp

    print(f"\n=== FULL video: lips_expression={expr}, inference_steps={steps} ===")

    out = lipsync_segmented(h, scene/"silent_joined.mp4", scene/"full_audio.wav",

                            outdir/f"full_{tag}.mp4", chunk, tag)

    print(f"  -> {out}  ({_dur(out):.2f}s)")

lip.WORKFLOW_PATH = ORIG_WF

lip.unload_pipeline(h)

print(f"\ncompare full videos in: {outdir}")

