"""
src/vram_utils.py — VRAM management helpers for the local pipeline.

`unload_pipeline()` is the canonical way to free GPU memory between stages.
The 4-step incantation (del + gc + empty_cache + ipc_collect) is what
PyTorch actually requires to release VRAM — del alone leaves cached
allocations behind, and gc alone leaves CUDA's internal pool reserved.

Call this between every stage transition in process_scenario().
"""

import gc
import torch


def unload_pipeline(pipeline) -> None:
    """
    Free all VRAM held by a diffusers pipeline (or any GPU-bound object).

    Args:
        pipeline: the pipeline reference to drop. Caller's reference is also
                  stale after this call. Typical usage:
                      pipe = step_x.load_pipeline()
                      ... use pipe ...
                      vram_utils.unload_pipeline(pipe)
                      pipe = None   # optional but explicit
    """
    try:
        # Move to CPU first to help PyTorch release CUDA allocations cleanly.
        if hasattr(pipeline, "to"):
            try:
                pipeline.to("cpu")
            except Exception:
                pass
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def report_vram(label: str = "") -> None:
    """
    Print current VRAM usage. Useful for verifying load/unload pattern works.
    """
    if not torch.cuda.is_available():
        print(f"[vram] {label} — no CUDA")
        return
    allocated_gb = torch.cuda.memory_allocated() / 1024**3
    reserved_gb = torch.cuda.memory_reserved() / 1024**3
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"[vram] {label} — allocated={allocated_gb:.2f}GB "
        f"reserved={reserved_gb:.2f}GB peak={peak_gb:.2f}GB"
    )


def reset_vram_peak() -> None:
    """Reset peak-allocation counter. Call at start of each stage to track per-stage peak."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()