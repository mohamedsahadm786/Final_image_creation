"""
video_pipeline/silentfirst/frame_join.py

Seamless join of SILENT clips via dHash frame-matching, with a quality threshold
and a punch-in fallback when no good match exists. CPU-only, no audio (silent clips);
the audio is added by ONE lip-sync pass later in the silent-first pipeline.

    join(clips, out_path, threshold=70, win=1.0, punch_in=1.2) -> (out_path, report)

report = list of per-seam dicts {seam, method, hamming, cutA, cutB}.
Method is 'framematch' (cut at most-similar frame pair) or 'punchin_fallback'
(no good match -> plain cut with the next clip punched-in).
"""
from __future__ import annotations
import subprocess
from pathlib import Path
from PIL import Image
import imagehash

FPS = 25
_NULL = subprocess.DEVNULL


def _dur(p):
    return float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(p)]).decode())


def _reencode(src, dst, vf):
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-c:v", "libx264",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(dst)],
                   check=True, stdout=_NULL, stderr=_NULL)


def _trim(src, dst, *, start=None, end=None):
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if start is not None: cmd += ["-ss", f"{start:.3f}"]
    if end is not None:   cmd += ["-t", f"{end:.3f}"]
    cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(dst)]
    subprocess.run(cmd, check=True, stdout=_NULL, stderr=_NULL)


def _hashes(path, start, length, workdir, tag):
    d = workdir / tag; d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(path),
                    "-t", f"{length:.3f}", "-vf", f"fps={FPS}", f"{d}/%04d.png"],
                   check=True, stdout=_NULL, stderr=_NULL)
    fs = sorted(d.glob("*.png"))
    return [(start + i / FPS, imagehash.dhash(Image.open(f), hash_size=16))
            for i, f in enumerate(fs)]


def _concat(a, b, dst):
    lst = Path(dst).parent / "_fj_list.txt"
    lst.write_text(f"file '{Path(a).resolve()}'\nfile '{Path(b).resolve()}'\n")
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(dst)], stdout=_NULL, stderr=_NULL)
    if r.returncode != 0:
        subprocess.run(["ffmpeg", "-y", "-i", str(a), "-i", str(b), "-filter_complex",
                        "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-c:v", "libx264",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(dst)],
                       check=True, stdout=_NULL, stderr=_NULL)


def join(clips, out_path, *, threshold=70, win=1.0, punch_in=1.2, w=768, h=1344, workdir=None):
    clips = [str(c) for c in clips]
    if len(clips) < 2:
        raise ValueError("need at least 2 clips")
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(workdir) if workdir else out_path.parent / "_fj_work"
    work.mkdir(parents=True, exist_ok=True)

    norm = []
    for i, c in enumerate(clips):
        n = work / f"norm_{i}.mp4"; _reencode(c, n, f"scale={w}:{h}"); norm.append(n)

    acc = work / "acc.mp4"; _reencode(norm[0], acc, f"scale={w}:{h}")
    report = []
    for i in range(1, len(norm)):
        nxt = norm[i]; ad = _dur(acc)
        tail = _hashes(acc, max(0, ad - win), min(win, ad), work, f"tA{i}")
        head = _hashes(nxt, 0, min(win, _dur(nxt)), work, f"hB{i}")
        cutA, cutB, dist = min(((ta, tb, ha - hb) for ta, ha in tail for tb, hb in head),
                               key=lambda x: x[2])
        new_acc = work / f"acc{i}.mp4"
        if dist <= threshold:
            aT = work / f"aT{i}.mp4"; bT = work / f"bT{i}.mp4"
            _trim(acc, aT, end=cutA); _trim(nxt, bT, start=cutB)
            _concat(aT, bT, new_acc); method = "framematch"
        else:
            bP = work / f"bP{i}.mp4"
            vf = (f"crop=iw/{punch_in}:ih/{punch_in},scale={w}:{h}" if punch_in > 1
                  else f"scale={w}:{h}")
            _reencode(nxt, bP, vf); _concat(acc, bP, new_acc); method = "punchin_fallback"
        acc = new_acc
        report.append({"seam": i, "method": method, "hamming": int(dist),
                       "cutA": round(cutA, 3), "cutB": round(cutB, 3)})
        print(f"  seam {i}: {method:18s} hamming={int(dist)}")
    subprocess.run(["ffmpeg", "-y", "-i", str(acc), "-c", "copy", str(out_path)],
                   check=True, stdout=_NULL, stderr=_NULL)
    return str(out_path), report


if __name__ == "__main__":
    import sys, json
    threshold = float(sys.argv[1]); out = sys.argv[2]; clips = sys.argv[3:]
    op, rep = join(clips, out, threshold=threshold)
    print("\n" + json.dumps(rep, indent=2)); print("joined ->", op, f"({_dur(op):.2f}s)")
