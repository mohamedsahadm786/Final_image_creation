"""video_pipeline/multishot/stitch.py — join N clips.
Default: plain concat. With punch_in>1.0: zoom every other clip (2,4,6,8...) by that
factor before joining so same-anchor cuts read as intentional reframes. Audio untouched."""
from __future__ import annotations
import subprocess
from pathlib import Path
_NULL = subprocess.DEVNULL


def _punch(clips, out_dir, factor, w, h):
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    for idx, c in enumerate(clips):
        vf = (f"crop=iw/{factor}:ih/{factor},scale={w}:{h}" if (idx + 1) % 2 == 0
              else f"scale={w}:{h}")
        outp = out_dir / f"_p{idx+1}.mp4"
        subprocess.run(["ffmpeg","-y","-i",c,"-vf",vf,"-c:v","libx264","-crf","18",
                        "-pix_fmt","yuv420p","-c:a","aac","-b:a","160k",str(outp)],
                       check=True, stdout=_NULL, stderr=_NULL)
        res.append(str(outp))
    return res


def stitch(clip_paths, out_path, *, punch_in: float = 1.0,
           punch_w: int = 768, punch_h: int = 1344, **_ignored):
    clips = [str(Path(p).resolve()) for p in clip_paths]
    if not clips:
        raise ValueError("no clips to stitch")
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    if punch_in and punch_in > 1.0 and len(clips) > 1:
        print(f"[stitch] alternating punch-in x{punch_in} on shots 2,4,6,...")
        clips = _punch(clips, out_path.parent / "_punch", punch_in, punch_w, punch_h)
    listfile = out_path.parent / "_concat_list.txt"
    listfile.write_text("".join(f"file '{c}'\n" for c in clips))
    print(f"[stitch] plain concat of {len(clips)} clip(s) -> {out_path}")
    r = subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(listfile),
                        "-c","copy",str(out_path)])
    if r.returncode != 0:
        print("[stitch] copy concat failed; re-encode fallback")
        inp = []
        for c in clips: inp += ["-i", c]
        n = len(clips)
        filt = "".join(f"[{i}:v][{i}:a]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]"
        subprocess.run(["ffmpeg","-y"]+inp+["-filter_complex",filt,"-map","[v]","-map","[a]",
                        "-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-c:a","aac",
                        "-b:a","160k","-movflags","+faststart",str(out_path)], check=True)
    return str(out_path)


if __name__ == "__main__":
    import sys
    stitch(sys.argv[2:], sys.argv[1]); print(f"done -> {sys.argv[1]}")
