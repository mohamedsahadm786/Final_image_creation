"""
src/trace_html.py — per-scenario 4-panel chain viewer.

Renders chain.html with:
  Panel 1: Persona reference photo (assets/persona.jpg)
  Panel 2: Step 1 — PuLID output (03_step1_persona.jpg)
  Panel 3: Step 2 — Qwen composite (05_step2_final.jpg)
  Panel 4: Step 3 — Kontext realism pass (07_step3_realism.jpg)

Plus:
  - Compare strip showing endpoint / elapsed / cost / params per stage
  - Step 1 / Step 2 / Step 3 prompts in collapsible blocks
  - Click-to-zoom lightbox on every image
  - Visual indicator of which panel is the canonical final image
    (Stage 3 if it ran, otherwise Stage 2)

Stage 3 states handled:
  - Ran successfully    → panel shows image, highlighted as final
  - Failed              → panel shows "Stage 3 failed: <error>" placeholder
  - Skipped (QC failed) → panel shows "Stage 3 skipped (QC did not pass)"
  - Disabled            → panel shows "Stage 3 disabled (STEP_3_ENABLED=false)"

Path math:
  Single-scenario:  outputs/<ts>_<sid>/chain.html  → 2 levels up to repo root → assets/persona.jpg = "../../assets/persona.jpg"
  Batch:            outputs/<ts>_batch/<sid>/chain.html → 3 levels up → "../../../assets/persona.jpg"

The caller passes `persona_rel_path` so this module doesn't need to know
whether it's being called from single-run or batch context.
"""

import html
from pathlib import Path


def _classify_step_3(record: dict) -> tuple[str, str | None]:
    """
    Determine the Stage 3 state from the record.

    Returns (state, message) where state is one of:
      - "ran"      : Stage 3 ran successfully → image exists at 07_step3_realism.jpg
      - "failed"   : Stage 3 was attempted but failed → message has error detail
      - "skipped"  : Stage 3 was skipped (QC failed) → message explains
      - "disabled" : Stage 3 was disabled via STEP_3_ENABLED=false
      - "unknown"  : couldn't determine — render an empty placeholder
    """
    step_3_meta = record.get("step_3_meta")
    qc_passed = bool((record.get("qc_result") or {}).get("passed"))
    final_status = record.get("final_status")

    # Stage 3 meta with no error and a local_path → it ran
    if step_3_meta and isinstance(step_3_meta, dict):
        if step_3_meta.get("error"):
            return ("failed", str(step_3_meta.get("error")))
        if step_3_meta.get("local_path"):
            return ("ran", None)
        # Meta dict exists but is malformed — treat as unknown
        return ("unknown", None)

    # No step_3_meta at all
    # If QC failed in Claude flow, Stage 3 was deliberately skipped
    if final_status == "qc_failed" or (record.get("qc_result") and not qc_passed):
        return ("skipped", "QC did not pass — Stage 3 not eligible")

    # Otherwise it was disabled or never reached
    return ("disabled", "Stage 3 was not run for this scenario")


def write_chain_html(
    out_dir: Path,
    record: dict,
    persona_rel_path: str = "../../assets/persona.jpg",
) -> None:
    """
    Render chain.html in out_dir.

    Args:
        out_dir: scenario output directory (where chain.html will be written)
        record: scenario record dict with keys:
            - scenario (dict with id, category, archetype, difficulty)
            - step_1_output (dict with step_1_image_prompt, word_count, fal_pulid_params)
            - step_2_output (dict with step_2_image_prompt, word_count)
            - step_1_meta (dict with elapsed_seconds, seed, cost_usd)
            - step_2_meta (dict with elapsed_seconds, seed, cost_usd)
            - step_3_meta (optional dict with elapsed_seconds, seed, cost_usd,
                           endpoint, safety_tolerance, prompt_used, OR error)
            - final_status ("success" / "failed" / "qc_failed")
            - error_message (optional)
            - error_stage (optional)
        persona_rel_path: relative path from chain.html to assets/persona.jpg
    """
    scenario = record.get("scenario", {}) or {}
    sc_id = scenario.get("id", "?")
    archetype = scenario.get("archetype", "")
    no_persona = archetype in ("flat_lay", "object_in_lineup")

    final_status = record.get("final_status", "?")
    error_message = record.get("error_message")
    error_stage = record.get("error_stage")

    # Classify Stage 3 so we know how to render its panel and where the
    # "final" highlight should go
    step_3_state, step_3_msg = _classify_step_3(record)
    step_3_is_final = step_3_state == "ran"

    # Build the 4 panel cards
    if no_persona:
        panel_persona = ("Persona reference", "(no persona — flat-lay scenario)", None, False, None)
    else:
        panel_persona = ("Persona reference", "assets/persona.jpg", persona_rel_path, False, None)

    # Each panel tuple is: (label, caption, src_or_None, is_final, fallback_message)
    panel_step_1 = (
        "Step 1 — PuLID",
        "Persona in scene (no product)",
        "03_step1_persona.jpg",
        False,
        None,
    )
    panel_step_2 = (
        "Step 2 — Qwen composite",
        "Product composited" if step_3_is_final else "Product composited (final)",
        "05_step2_final.jpg",
        not step_3_is_final,  # Stage 2 is final if Stage 3 didn't run
        None,
    )

    # Stage 3 panel — depends on state
    if step_3_state == "ran":
        panel_step_3 = (
            "Step 3 — Kontext realism",
            "Photoreal pass (final)",
            "07_step3_realism.jpg",
            True,
            None,
        )
    elif step_3_state == "failed":
        panel_step_3 = (
            "Step 3 — Kontext realism",
            "Stage 3 failed",
            None,
            False,
            f"Stage 3 failed: {step_3_msg[:120] if step_3_msg else 'unknown error'}",
        )
    elif step_3_state == "skipped":
        panel_step_3 = (
            "Step 3 — Kontext realism",
            "Skipped",
            None,
            False,
            step_3_msg or "Stage 3 skipped",
        )
    else:  # disabled / unknown
        panel_step_3 = (
            "Step 3 — Kontext realism",
            "Not run",
            None,
            False,
            step_3_msg or "Stage 3 not run",
        )

    panels = [panel_persona, panel_step_1, panel_step_2, panel_step_3]

    cards = []
    for label, caption, src, is_final, fallback_msg in panels:
        accent = "stage-final" if is_final else ""
        if src is None:
            placeholder = html.escape(fallback_msg or "not produced")
            cards.append(
                f"""<div class="stage {accent}">
  <div class="stage-label">{html.escape(label)}<br><span class="stage-cap">{html.escape(caption)}</span></div>
  <div class="stage-empty">{placeholder}</div>
</div>"""
            )
        else:
            cards.append(
                f"""<div class="stage {accent}">
  <div class="stage-label">{html.escape(label)}<br><span class="stage-cap">{html.escape(caption)}</span></div>
  <img src="{html.escape(src)}" alt="{html.escape(caption)}"
       onerror="this.outerHTML='<div class=stage-empty>not produced</div>'">
</div>"""
            )

    # Prompts
    step_1_prompt = (
        record.get("step_1_output", {}).get("step_1_image_prompt", "(not available)")
    )
    step_2_prompt = (
        record.get("step_2_output", {}).get("step_2_image_prompt", "(not available)")
    )
    step_1_wc = record.get("step_1_output", {}).get("word_count", "—")
    step_2_wc = record.get("step_2_output", {}).get("word_count", "—")

    step_1_meta = record.get("step_1_meta") or {}
    step_2_meta = record.get("step_2_meta") or {}
    step_3_meta = record.get("step_3_meta") or {}

    # Stage 3 instruction prompt (the realism instruction sent to Kontext)
    step_3_prompt = step_3_meta.get("prompt_used") if isinstance(step_3_meta, dict) else None
    step_3_wc = (
        len(step_3_prompt.split()) if isinstance(step_3_prompt, str) and step_3_prompt
        else "—"
    )

    def _fmt_seconds(meta):
        s = meta.get("elapsed_seconds") if isinstance(meta, dict) else None
        return f"{s:.1f}s" if isinstance(s, (int, float)) else "—"

    def _fmt_seed(meta):
        s = meta.get("seed") if isinstance(meta, dict) else None
        return str(s) if s is not None else "—"

    def _fmt_cost(meta):
        c = meta.get("cost_usd") if isinstance(meta, dict) else None
        return f"${c:.3f}" if isinstance(c, (int, float)) else "—"

    # Pull a couple of key PuLID params for display
    pulid_params = record.get("step_1_output", {}).get("fal_pulid_params", {}) or {}
    id_weight = pulid_params.get("id_weight", "—")
    true_cfg = pulid_params.get("true_cfg", "—")
    guidance = pulid_params.get("guidance_scale", "—")

    # Stage 3 endpoint + safety_tolerance for the compare-strip
    step_3_endpoint = (
        step_3_meta.get("endpoint", "fal-ai/flux-pro/kontext")
        if isinstance(step_3_meta, dict) else "fal-ai/flux-pro/kontext"
    )
    step_3_safety = (
        step_3_meta.get("safety_tolerance", "—")
        if isinstance(step_3_meta, dict) else "—"
    )

    # Status badge + error block
    if final_status == "success":
        badge = '<span class="badge badge-ok">SUCCESS</span>'
    elif final_status == "qc_failed":
        badge = '<span class="badge badge-warn">QC FAILED</span>'
    else:
        badge = '<span class="badge badge-fail">FAILED</span>'

    error_block = ""
    if final_status not in ("success",) and error_message:
        stage_str = (
            f' at stage <code>{html.escape(str(error_stage))}</code>'
            if error_stage else ""
        )
        error_block = f"""<div class="error-block">
  <strong>Run did not fully succeed{stage_str}:</strong> {html.escape(str(error_message))}
</div>"""

    # Step 3 column for compare-strip — varies by state
    if step_3_state == "ran":
        step_3_col = f"""<div class="col {'col-final' if step_3_is_final else ''}">
    <div class="col-title">Step 3 — Kontext realism</div>
    <div class="col-row">endpoint: <strong>{html.escape(step_3_endpoint)}</strong></div>
    <div class="col-row">instruction: <strong>{html.escape(str(step_3_wc))} words</strong></div>
    <div class="col-row">safety_tolerance: <strong>{html.escape(str(step_3_safety))}</strong></div>
    <div class="col-row">elapsed: <strong>{_fmt_seconds(step_3_meta)}</strong> · seed: <strong>{_fmt_seed(step_3_meta)}</strong> · cost: <strong>{_fmt_cost(step_3_meta)}</strong></div>
  </div>"""
    else:
        state_label = {
            "failed": "Failed",
            "skipped": "Skipped",
            "disabled": "Disabled",
            "unknown": "Not run",
        }.get(step_3_state, "Not run")
        step_3_col = f"""<div class="col col-step3-empty">
    <div class="col-title">Step 3 — Kontext realism</div>
    <div class="col-row" style="color:#9CA3AF">{html.escape(state_label)}</div>
    <div class="col-row" style="color:#9CA3AF;font-size:11px">{html.escape((step_3_msg or '')[:160])}</div>
  </div>"""

    # Step 3 prompt block — only render if Stage 3 actually ran
    step_3_prompt_block = ""
    if step_3_state == "ran" and step_3_prompt:
        step_3_prompt_block = f"""
  <h3 class="h3-final">Step 3 instruction → fal-ai/flux-pro/kontext</h3>
  <div class="prompt-meta">{html.escape(str(step_3_wc))} words (instruction-based edit, not full-image description)</div>
  <div class="prompt-block prompt-final">{html.escape(step_3_prompt)}</div>"""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html.escape(sc_id)} — Alluvi pipeline</title>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif;
       margin:0;padding:24px;background:#F4F6F8;color:#1F2937;max-width:1700px;margin:0 auto}}
  h1{{margin:0 0 6px 0;font-size:18px;padding:24px 24px 0}}
  .meta{{color:#6B7280;font-size:13px;margin-bottom:18px;padding:0 24px}}
  .meta-pill{{display:inline-block;background:#fff;padding:3px 10px;
             border-radius:4px;border:1px solid #E5E7EB;margin-right:6px;font-size:12px}}
  .stage-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;
             margin:0 24px 24px;padding:0}}
  .stage{{background:#fff;border:1px solid #E5E7EB;border-radius:8px;overflow:hidden}}
  .stage.stage-final{{border:2px solid #0EA5A4;box-shadow:0 0 0 2px #CCFBF1}}
  .stage-label{{padding:10px 12px;border-bottom:1px solid #E5E7EB;font-size:12px;
               font-weight:600;background:#F9FAFB;line-height:1.4;min-height:50px}}
  .stage-final .stage-label{{background:#ECFEFF;color:#0F766E}}
  .stage-cap{{font-weight:400;color:#6B7280;font-size:11px}}
  .stage img{{width:100%;display:block;aspect-ratio:9/16;object-fit:cover;background:#F3F4F6;
             cursor:zoom-in}}
  .stage-empty{{aspect-ratio:9/16;background:#F3F4F6;display:flex;
               align-items:center;justify-content:center;color:#9CA3AF;font-size:11px;
               text-align:center;padding:0 12px;line-height:1.5}}
  .compare-strip{{background:#fff;border:1px solid #E5E7EB;border-radius:8px;
                 padding:14px 18px;margin:0 24px 18px;font-size:13px;
                 display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
  .compare-strip .col-title{{font-weight:600;margin-bottom:6px;font-size:12px;
                            text-transform:uppercase;letter-spacing:.5px;color:#374151}}
  .compare-strip .col.col-final .col-title{{color:#0F766E}}
  .compare-strip .col-step3-empty .col-title{{color:#9CA3AF}}
  .compare-strip .col-row{{color:#6B7280;font-size:12px;line-height:1.7}}
  .compare-strip .col-row strong{{color:#1F2937;font-weight:600}}
  .prompts{{background:#fff;border:1px solid #E5E7EB;border-radius:8px;padding:18px;
           margin:0 24px}}
  .prompts h3{{margin:0 0 8px;font-size:13px;color:#374151;text-transform:uppercase;
              letter-spacing:.5px}}
  .prompts h3.h3-final{{color:#0F766E}}
  .prompt-block{{background:#F9FAFB;border:1px solid #E5E7EB;border-radius:6px;
                padding:12px;font-size:12.5px;line-height:1.55;
                font-family:ui-monospace,'SF Mono',Menlo,monospace;
                margin-bottom:18px;white-space:pre-wrap;word-wrap:break-word}}
  .prompt-block.prompt-final{{background:#ECFEFF;border-color:#A5F3FC}}
  .prompt-meta{{font-size:11px;color:#6B7280;margin-bottom:4px;
               font-family:ui-monospace,'SF Mono',Menlo,monospace}}
  .badge{{display:inline-block;padding:3px 9px;border-radius:12px;
         font-size:10.5px;font-weight:600;margin-left:6px}}
  .badge-ok{{background:#D1FAE5;color:#065F46}}
  .badge-warn{{background:#FEF3C7;color:#92400E}}
  .badge-fail{{background:#FEE2E2;color:#991B1B}}
  .error-block{{background:#FFF7ED;border-left:3px solid #F97316;padding:10px 14px;
                margin:0 24px 18px;border-radius:4px;font-size:13px;color:#7C2D12}}
  .error-block code{{background:#FED7AA;padding:1px 6px;border-radius:3px}}
  .back{{display:inline-block;margin:24px 24px 0;color:#6B7280;font-size:12px;
        text-decoration:none}}
  .back:hover{{color:#1F2937;text-decoration:underline}}
  .lightbox{{display:none;position:fixed;top:0;left:0;width:100vw;height:100vh;
            background:rgba(0,0,0,.85);z-index:1000;cursor:zoom-out;
            align-items:center;justify-content:center}}
  .lightbox.active{{display:flex}}
  .lightbox img{{max-width:95vw;max-height:95vh;object-fit:contain}}
  @media (max-width: 1100px) {{
    .stage-row {{ grid-template-columns: repeat(2, 1fr); }}
    .compare-strip {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<a class="back" href="../overview.html">← Back to overview</a>
<h1>{html.escape(sc_id)} {badge}</h1>
<div class="meta">
  <span class="meta-pill"><strong>{html.escape(scenario.get('category', '?'))}</strong></span>
  <span class="meta-pill">{html.escape(scenario.get('archetype', '?'))}</span>
  <span class="meta-pill">{html.escape(scenario.get('difficulty', '?'))}</span>
</div>

{error_block}

<div class="stage-row">
{''.join(cards)}
</div>

<div class="compare-strip">
  <div class="col">
    <div class="col-title">Step 1 — PuLID</div>
    <div class="col-row">endpoint: <strong>fal-ai/flux-pulid</strong></div>
    <div class="col-row">prompt: <strong>{html.escape(str(step_1_wc))} words</strong></div>
    <div class="col-row">id_weight: <strong>{html.escape(str(id_weight))}</strong> · true_cfg: <strong>{html.escape(str(true_cfg))}</strong> · guidance: <strong>{html.escape(str(guidance))}</strong></div>
    <div class="col-row">elapsed: <strong>{_fmt_seconds(step_1_meta)}</strong> · seed: <strong>{_fmt_seed(step_1_meta)}</strong> · cost: <strong>{_fmt_cost(step_1_meta)}</strong></div>
  </div>
  <div class="col {'col-final' if not step_3_is_final else ''}">
    <div class="col-title">Step 2 — Qwen composite</div>
    <div class="col-row">endpoint: <strong>fal-ai/qwen-image-edit-2511</strong></div>
    <div class="col-row">prompt: <strong>{html.escape(str(step_2_wc))} words (qwen-tuned)</strong></div>
    <div class="col-row">elapsed: <strong>{_fmt_seconds(step_2_meta)}</strong> · seed: <strong>{_fmt_seed(step_2_meta)}</strong> · cost: <strong>{_fmt_cost(step_2_meta)}</strong></div>
  </div>
  {step_3_col}
</div>

<div class="prompts">
  <h3>Step 1 prompt → fal-ai/flux-pulid</h3>
  <div class="prompt-meta">{html.escape(str(step_1_wc))} words</div>
  <div class="prompt-block">{html.escape(step_1_prompt)}</div>

  <h3{'' if step_3_is_final else ' class="h3-final"'}>Step 2 prompt → fal-ai/qwen-image-edit-2511 (qwen-tuned)</h3>
  <div class="prompt-meta">{html.escape(str(step_2_wc))} words</div>
  <div class="prompt-block{'' if step_3_is_final else ' prompt-final'}">{html.escape(step_2_prompt)}</div>
{step_3_prompt_block}
</div>

<div class="lightbox" id="lightbox" onclick="this.classList.remove('active')">
  <img id="lightbox-img" src="" alt="">
</div>

<script>
  document.querySelectorAll('.stage img').forEach(img => {{
    img.addEventListener('click', () => {{
      const lb = document.getElementById('lightbox');
      const lbImg = document.getElementById('lightbox-img');
      lbImg.src = img.src;
      lb.classList.add('active');
    }});
  }});
</script>
</body>
</html>
"""
    (out_dir / "chain.html").write_text(html_doc, encoding="utf-8")