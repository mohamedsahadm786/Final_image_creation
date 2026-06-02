"""Trim a video to its audio track's duration — removes LatentSync's silent_padding
frozen/looped tail. General utility; called by step_6_lipsync so every pipeline benefits."""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path


def _duration(path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(out.decode().strip())


def trim_video_to_audio(video_path, audio_path, *, margin: float = 0.0) -> str:
    """Trim video_path IN PLACE to the audio duration (+ optional margin seconds).
    No-op if the video is already that short (never extends). Returns video_path."""
    video_path = str(video_path)
    target = _duration(audio_path) + float(margin)
    if _duration(video_path) <= target + 0.02:
        return video_path                       # already tight, nothing to remove
    tmp = video_path + ".trim.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-t", f"{target:.3f}",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", tmp],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.move(tmp, video_path)
    return video_path
