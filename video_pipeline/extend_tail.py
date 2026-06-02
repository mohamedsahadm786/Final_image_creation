"""Append a held-frame + silence tail to a finished video for a natural ending.
extend_tail(video, out=None, seconds=2.0) -> out_path
Clones the last frame for `seconds` and pads the audio with matching silence.
If out is None, overwrites the input in place (via temp). Generic: works on any
finished video (multishot stitch or silent-first lip-sync output)."""
import os, subprocess
from pathlib import Path
_NULL = subprocess.DEVNULL


def extend_tail(video, out=None, seconds=2.0):
    video = str(video)
    if not seconds or seconds <= 0:
        return video
    out = str(out) if out else video
    tmp = str(Path(out).with_suffix(".tail_tmp.mp4"))
    base = ["ffmpeg", "-y", "-i", video,
            "-vf", f"tpad=stop_mode=clone:stop_duration={seconds}"]
    try:
        subprocess.run(base + ["-af", f"apad=pad_dur={seconds}", "-c:v", "libx264",
                               "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
                               "-b:a", "192k", tmp], check=True, stdout=_NULL, stderr=_NULL)
    except subprocess.CalledProcessError:           # no audio stream -> video-only tail
        subprocess.run(base + ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                               "-an", tmp], check=True, stdout=_NULL, stderr=_NULL)
    os.replace(tmp, out)
    return out
