import sys, glob, subprocess
sys.path.insert(0, "/workspace/alluvi-pipeline")
from pathlib import Path
from video_pipeline import step_6_lipsync as lip
_NULL = subprocess.DEVNULL

def _dur(p):
    return float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1",str(p)]).decode())

def lipsync_segmented(h, video, audio, out_path, chunk=7.0):
    out_path = Path(out_path); A = _dur(audio)
    seg = out_path.parent/"_lipseg"; seg.mkdir(parents=True, exist_ok=True)
    conf = seg/"conf.mp4"
    subprocess.run(["ffmpeg","-y","-i",str(video),"-t",f"{A:.3f}","-c:v","libx264","-crf","18",
        "-pix_fmt","yuv420p","-an",str(conf)],check=True,stdout=_NULL,stderr=_NULL)
    n = max(1, int(A/chunk + 0.999)); b=[i*A/n for i in range(n+1)]; b[-1]=A; parts=[]
    for i in range(n):
        s,d = b[i], b[i+1]-b[i]; vc=seg/f"v{i}.mp4"; ac=seg/f"a{i}.wav"
        subprocess.run(["ffmpeg","-y","-ss",f"{s:.3f}","-i",str(conf),"-t",f"{d:.3f}","-c:v","libx264",
            "-crf","18","-pix_fmt","yuv420p","-an",str(vc)],check=True,stdout=_NULL,stderr=_NULL)
        subprocess.run(["ffmpeg","-y","-ss",f"{s:.3f}","-i",str(audio),"-t",f"{d:.3f}",
            "-ar","16000","-ac","1",str(ac)],check=True,stdout=_NULL,stderr=_NULL)
        r=lip.generate(h,str(vc),str(ac),seg/f"f{i}",scene_id=f"reseg{i+1}")
        parts.append(r["local_path"]); print(f"  seg {i+1}/{n} ({d:.1f}s) ok")
    lst=seg/"list.txt"; lst.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in parts))
    rc=subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(lst),"-c","copy",
        str(out_path)],stdout=_NULL,stderr=_NULL).returncode
    if rc!=0:
        subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(lst),"-c:v","libx264",
            "-crf","18","-pix_fmt","yuv420p","-c:a","aac","-b:a","192k",str(out_path)],
            check=True,stdout=_NULL,stderr=_NULL)
    return str(out_path)

scene = Path(sys.argv[1]) if len(sys.argv)>1 else Path(sorted(glob.glob(
    "/workspace/alluvi-pipeline/outputs/*_silentfirst_real/*/*"))[-1])
chunk = float(sys.argv[2]) if len(sys.argv)>2 else 7.0
print("scene:", scene)
h = lip.load_pipeline()
out = lipsync_segmented(h, scene/"silent_joined.mp4", scene/"full_audio.wav",
                        scene/"final_segmented.mp4", chunk=chunk)
print("DONE ->", out, f"({_dur(out):.2f}s)")
lip.unload_pipeline(h)
