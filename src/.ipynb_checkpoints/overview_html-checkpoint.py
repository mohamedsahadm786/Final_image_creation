"""
src/overview_html.py — batch overview viewer.

Renders overview.html at the batch root with a grid of cards. Each card:
  - Scenario id + success/fail/qc_failed badge
  - Canonical final image thumbnail
      (07_step3_realism.jpg if Stage 3 ran, otherwise 05_step2_final.jpg)
  - Meta pills: category, archetype, difficulty, qwen elapsed, prompt size,
    Stage 3 status (✓ ran / "—" skipped / "fail" errored)
  - Failure stage pill if applicable
  - Click anywhere on card → opens <scenario_id>/chain.html (4-panel view)
"""

import html
from pathlib import Path


def _classify_step_3(record: dict) -> str:
    """
    Compact Stage 3 classification used for the overview card pill.
    Returns one of: "ran", "failed", "skipped", "disabled", "unknown".
    """
    step_3_meta = record.get("step_3_meta")
    qc_passed = bool((record.get("qc_result") or {}).get("passed"))
    final_status = record.get("final_status")

    if step_3_meta and isinstance(step_3_meta, dict):
        if step_3_meta.get("error"):
            return "failed"
        if step_3_meta.get("local_path"):
            return "ran"
        return "unknown"

    if final_status == "qc_failed" or (record.get("qc_result") and not qc_passed):
        return "skipped"
    return "disabled"


def write_overview_html(
    batch_dir: Path,
    records: list,
    summary: dict,
) -> None:
    """
    Render overview.html at the batch root.

    Args:
        batch_dir: top-level batch output directory
        records: list of per-scenario record dicts
        summary: dict with keys:
            - succeeded (int), failed (int)
            - actual_cost_usd (float)
            - elapsed_seconds (float)
            - timestamp (str)
            - model_label (str)
            - interrupted (bool)
    """
    n = len(records)
    succeeded = summary.get("succeeded", 0)
    failed = summary.get("failed", 0)
    actual_cost = summary.get("actual_cost_usd", 0)
    elapsed = summary.get("elapsed_seconds", 0)
    timestamp = summary.get("timestamp", "?")
    model_label = summary.get("model_label", "PuLID + Qwen + Kontext")
    interrupted = summary.get("interrupted", False)

    # Tally failures by stage for the failure summary band
    failure_stages: dict[str, int] = {}
    for r in records:
        if r.get("final_status") != "success":
            stage = r.get("error_stage", "unknown")
            failure_stages[stage] = failure_stages.get(stage, 0) + 1

    # Tally Stage 3 outcomes across the batch
    step_3_counts = {"ran": 0, "failed": 0, "skipped": 0, "disabled": 0, "unknown": 0}
    for r in records:
        step_3_counts[_classify_step_3(r)] += 1

    cards = []
    for r in records:
        sc = r.get("scenario", {}) or {}
        sc_id = sc.get("id", "?")
        ok = r.get("final_status") == "success"
        qc_failed = r.get("final_status") == "qc_failed"

        if ok:
            badge_class = "badge-ok"
            badge_text = "SUCCESS"
        elif qc_failed:
            badge_class = "badge-warn"
            badge_text = "QC FAILED"
        else:
            badge_class = "badge-fail"
            badge_text = "FAILED"

        # Pick the canonical final image: Stage 3 if it ran, else Stage 2
        step_3_state = _classify_step_3(r)
        if step_3_state == "ran":
            final_img = f"{sc_id}/07_step3_realism.jpg"
            final_img_label = "Stage 3 (Kontext realism)"
        else:
            final_img = f"{sc_id}/05_step2_final.jpg"
            final_img_label = "Stage 2 (Qwen composite)"

        chain_path = f"{sc_id}/chain.html"

        step_2_meta = r.get("step_2_meta") or {}
        elapsed_s = step_2_meta.get("elapsed_seconds")
        elapsed_str = (
            f"{elapsed_s:.1f}s" if isinstance(elapsed_s, (int, float)) else "—"
        )

        step_2_wc = r.get("step_2_output", {}).get("word_count", "—")
        error_stage = r.get("error_stage", "")

        # Stage 3 pill
        step_3_meta = r.get("step_3_meta") or {}
        if step_3_state == "ran":
            s3_elapsed = step_3_meta.get("elapsed_seconds")
            s3_str = (
                f"{s3_elapsed:.1f}s" if isinstance(s3_elapsed, (int, float)) else "✓"
            )
            step_3_pill = (
                f'<span class="meta-pill" style="background:#ECFEFF;color:#0F766E">'
                f'kontext: {html.escape(s3_str)}</span>'
            )
        elif step_3_state == "failed":
            step_3_pill = (
                '<span class="meta-pill" style="background:#FEE2E2;color:#991B1B">'
                'kontext: failed</span>'
            )
        elif step_3_state == "skipped":
            step_3_pill = (
                '<span class="meta-pill" style="background:#FEF3C7;color:#92400E">'
                'kontext: skipped</span>'
            )
        else:  # disabled / unknown
            step_3_pill = (
                '<span class="meta-pill" style="background:#F3F4F6;color:#6B7280">'
                'kontext: —</span>'
            )

        error_pill = ""
        if not ok and error_stage:
            error_pill = (
                f'<span class="meta-pill" style="background:#FEE2E2;color:#991B1B">'
                f'failed: {html.escape(error_stage)}</span>'
            )

        cards.append(
            f"""
<div class="card">
  <div class="card-header">
    <span class="card-id">{html.escape(sc_id)}</span>
    <span class="badge {badge_class}">{badge_text}</span>
  </div>
  <a href="{html.escape(chain_path)}" class="card-img-link" title="Showing: {html.escape(final_img_label)}">
    <div class="card-img-wrap">
      <img src="{html.escape(final_img)}" alt="final"
           onerror="this.outerHTML='<div class=card-empty>—</div>'">
    </div>
  </a>
  <div class="card-meta">
    <span class="meta-pill">{html.escape(sc.get('category', '?'))}</span>
    <span class="meta-pill">{html.escape(sc.get('archetype', '?'))}</span>
    <span class="meta-pill">{html.escape(sc.get('difficulty', '?'))}</span>
    <span class="meta-pill">qwen: {html.escape(elapsed_str)}</span>
    <span class="meta-pill">{html.escape(str(step_2_wc))}w prompt</span>
    {step_3_pill}
    {error_pill}
  </div>
</div>"""
        )

    title_suffix = " (interrupted)" if interrupted else ""

    failure_summary = ""
    if failure_stages:
        failure_summary = (
            "Failures by stage: "
            + " · ".join(
                f"<strong>{html.escape(k)}</strong>: {v}"
                for k, v in sorted(failure_stages.items())
            )
        )

    # Stage 3 summary band
    s3_summary_parts = []
    if step_3_counts["ran"]:
        s3_summary_parts.append(f'<strong style="color:#0F766E">ran:</strong> {step_3_counts["ran"]}')
    if step_3_counts["failed"]:
        s3_summary_parts.append(f'<strong style="color:#991B1B">failed:</strong> {step_3_counts["failed"]}')
    if step_3_counts["skipped"]:
        s3_summary_parts.append(f'<strong style="color:#92400E">skipped:</strong> {step_3_counts["skipped"]}')
    if step_3_counts["disabled"]:
        s3_summary_parts.append(f'<strong style="color:#6B7280">disabled:</strong> {step_3_counts["disabled"]}')
    s3_summary = (
        f'Stage 3 (Kontext): ' + ' · '.join(s3_summary_parts)
    ) if s3_summary_parts else ""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Alluvi batch — {html.escape(timestamp)}</title>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif;
       margin:0;padding:0;background:#F4F6F8;color:#1F2937}}
  header{{background:#0F766E;color:#fff;padding:24px 32px}}
  header h1{{margin:0 0 6px;font-size:22px;font-weight:600}}
  header .meta{{font-size:13px;opacity:.85;font-family:ui-monospace,'SF Mono',Menlo,monospace;line-height:1.7}}
  .summary-bar{{display:flex;gap:22px;padding:16px 32px;background:#fff;
               border-bottom:1px solid #E5E7EB;font-size:14px;flex-wrap:wrap}}
  .summary-bar .stat{{display:flex;gap:6px}}
  .summary-bar .stat-label{{color:#6B7280}}
  .summary-bar .stat-value{{font-weight:600}}
  .failure-summary{{padding:10px 32px;background:#FFF7ED;border-bottom:1px solid #FED7AA;
                  font-size:12px;color:#7C2D12}}
  .stage3-summary{{padding:10px 32px;background:#F0FDFA;border-bottom:1px solid #99F6E4;
                  font-size:12px;color:#134E4A}}
  .legend{{padding:10px 32px;background:#F9FAFB;border-bottom:1px solid #E5E7EB;
          font-size:12px;color:#6B7280}}
  .legend strong{{color:#1F2937}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
        gap:20px;padding:24px 32px}}
  .card{{background:#fff;border:1px solid #E5E7EB;border-radius:10px;overflow:hidden;
        transition:box-shadow .15s ease}}
  .card:hover{{box-shadow:0 4px 12px rgba(0,0,0,.08)}}
  .card-header{{padding:10px 14px;border-bottom:1px solid #E5E7EB;
               display:flex;justify-content:space-between;align-items:center}}
  .card-id{{font-family:ui-monospace,monospace;font-size:11px;color:#6B7280}}
  .card-img-link{{display:block;text-decoration:none;color:inherit}}
  .card-img-wrap{{background:#F3F4F6;position:relative}}
  .card-img-wrap img{{width:100%;display:block;aspect-ratio:9/16;object-fit:cover}}
  .card-empty{{aspect-ratio:9/16;background:#F3F4F6;display:flex;align-items:center;
              justify-content:center;color:#9CA3AF;font-size:12px}}
  .card-meta{{padding:8px 14px;border-top:1px solid #E5E7EB}}
  .meta-pill{{display:inline-block;background:#F3F4F6;padding:3px 9px;
             border-radius:4px;color:#4B5563;font-size:11px;margin-right:4px;
             margin-bottom:2px}}
  .badge{{display:inline-block;padding:3px 9px;border-radius:12px;
         font-size:10.5px;font-weight:600}}
  .badge-ok{{background:#D1FAE5;color:#065F46}}
  .badge-warn{{background:#FEF3C7;color:#92400E}}
  .badge-fail{{background:#FEE2E2;color:#991B1B}}
</style>
</head>
<body>
<header>
  <h1>Alluvi pipeline batch — full from-scratch{html.escape(title_suffix)}</h1>
  <div class="meta">
    {html.escape(timestamp)} · model: {html.escape(model_label)}<br>
    PuLID Stage 1 + Qwen-Image-Edit-2511 Stage 2 + FLUX.1 Kontext Pro Stage 3 (realism)
  </div>
</header>
<div class="summary-bar">
  <div class="stat"><span class="stat-label">Total:</span> <span class="stat-value">{n}</span></div>
  <div class="stat"><span class="stat-label">Success:</span> <span class="stat-value" style="color:#065F46">{succeeded}</span></div>
  <div class="stat"><span class="stat-label">Failed:</span> <span class="stat-value" style="color:#991B1B">{failed}</span></div>
  <div class="stat"><span class="stat-label">Cost:</span> <span class="stat-value">${actual_cost:.2f}</span></div>
  <div class="stat"><span class="stat-label">Elapsed:</span> <span class="stat-value">{elapsed:.0f}s ({elapsed/60:.1f} min)</span></div>
</div>
{f'<div class="failure-summary">{failure_summary}</div>' if failure_summary else ''}
{f'<div class="stage3-summary">{s3_summary}</div>' if s3_summary else ''}
<div class="legend">
  Each card shows the <strong>canonical final image</strong> — Stage 3 (Kontext realism) output if it ran, otherwise Stage 2 (Qwen composite). Click a card to open the 4-panel chain.html with persona reference, Stage 1, Stage 2, Stage 3, and all prompts side by side.
</div>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>
"""
    (batch_dir / "overview.html").write_text(html_doc, encoding="utf-8")