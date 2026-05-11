# Alluvi Final Image Generation Pipeline

End-to-end image generation pipeline that produces premium-looking TikTok ad images for the Alluvi Tirzepatide 40mg product, using a 2-stage architecture:

- **Stage 1** — `fal-ai/flux-pulid` generates a persona-in-scene image (no product) with locked identity (~99% face fidelity)
- **Stage 2** — `fal-ai/qwen-image-edit-2511` composites the Alluvi product naturally into her hand or onto a surface in the scene

Each stage's prompt is generated dynamically by **Claude Opus 4.7** using a hand-tuned master prompt for that stage.

This repo is a clean single-flow extraction of the `qwen_tuned_prompt` experiment from the prototype repo. No A/B comparisons, no Nano Banana baseline, no reuse of historical runs — everything generated from scratch for every scenario.

---

## Architecture

```
scenarios.yaml entry
       │
       ▼
[ Step 1 prompt builder (Opus 4.7 + master_prompt_step1.md) ]
       │
       │  produces: step_1_image_prompt + per-scenario fal_pulid_params
       ▼
[ Stage 1: fal-ai/flux-pulid + assets/persona.jpg ]
       │
       │  produces: persona-in-scene image (no product, both hands visible)
       ▼
[ Step 2 prompt builder (Opus 4.7 + master_prompt_step2_qwen.md) ]
       │
       │  produces: step_2_image_prompt (qwen-tuned, 280-380 words)
       ▼
[ Stage 2: fal-ai/qwen-image-edit-2511 + persona scene + assets/product.jpg ]
       │
       ▼
Final 9:16 image (TikTok-ready)
```

---

## Quick start

### 1. Install

```powershell
# Activate your venv first
python -m pip install -r requirements.txt
```

### 2. Set API keys

Copy `.env.example` to `.env` and fill in:

```
FAL_KEY=your_fal_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 3. Run a single scenario (cost: ~$0.36)

```powershell
python run.py --scenario bedroom_robe_with_product_13
```

Output goes to `outputs/<timestamp>_<scenario_id>/` and includes:
- `01_scenario.yaml` — the scenario record
- `02_step1_prompt.json` — Opus-generated Step 1 prompt envelope
- `03_step1_persona.jpg` — PuLID output (persona in scene, no product)
- `03_step1_meta.json` — PuLID call meta (seed, elapsed, params)
- `04_step2_prompt.json` — Opus-generated Step 2 prompt envelope
- `05_step2_final.jpg` — Qwen output (final composited image)
- `05_step2_meta.json` — Qwen call meta
- `chain.html` — 3-panel viewer (persona reference / Step 1 / Step 2) with click-to-zoom

### 4. Run a full batch (all 30 scenarios, ~$10.80, ~35 min sequential)

```powershell
python run_batch.py
```

Filters:
```powershell
python run_batch.py --only bedroom_robe_with_product_13,kitchen_matcha_morning_handheld_16
python run_batch.py --exclude flat_lay_white_marble_29
python run_batch.py --yes              # skip cost confirmation
```

Batch output goes to `outputs/<timestamp>_batch/` and includes:
- `overview.html` — grid of all scenarios with click-to-zoom
- `batch_manifest.json` — machine-readable summary
- `<scenario_id>/` — per-scenario subdirectories (same shape as single run)

---

## Cost per scenario

| Stage | Cost | What |
|---|---|---|
| Step 1 prompt (Opus 4.7) | ~$0.10 | builds the PuLID prompt envelope |
| Stage 1 (PuLID via fal) | ~$0.04 | renders persona scene |
| Step 2 prompt (Opus 4.7) | ~$0.18 | builds the Qwen prompt envelope |
| Stage 2 (Qwen via fal) | ~$0.04 | composites product |
| **Total per scenario** | **~$0.36** | |
| **30-scenario batch** | **~$10.80** | |

Wall time: ~60-80s per scenario sequential.

---

## Repo layout

```
Final_Image_generation/
├── README.md                              ← this file
├── requirements.txt
├── .env.example
├── .gitignore
├── config.yaml                            ← endpoints, costs, defaults
│
├── assets/
│   ├── persona.jpg                        ← reference photo (locked)
│   ├── persona.yaml                       ← prompt_descriptors (verbatim into prompts)
│   ├── product.jpg                        ← Alluvi box (cropped, no lab bg)
│   └── product.yaml                       ← product validation (internal only)
│
├── brand/
│   ├── brand.yaml
│   └── do_dont.md                         ← UK ASA + MHRA compliance rules
│
├── prompts/
│   ├── master_prompt_step1.md             ← Step 1 system prompt (drives Opus)
│   └── master_prompt_step2_qwen.md        ← Step 2 Qwen-tuned system prompt
│
├── scenarios/
│   └── scenarios.yaml                     ← 30 hand-curated scenarios
│
├── src/
│   ├── __init__.py
│   ├── step_1_prompt_builder.py           ← Opus → Step 1 prompt envelope
│   ├── step_1_pulid.py                    ← fal PuLID caller
│   ├── step_2_prompt_builder.py           ← Opus → Step 2 prompt envelope
│   ├── step_2_qwen_edit.py                ← fal Qwen-Image-Edit caller
│   ├── trace_html.py                      ← per-scenario chain.html viewer
│   └── overview_html.py                   ← batch overview.html viewer
│
├── run.py                                 ← single-scenario CLI
├── run_batch.py                           ← multi-scenario CLI
│
├── cache/                                 ← auto-created
│   └── fal_uploads.json                   ← caches fal upload URLs by abs path
│
└── outputs/                               ← auto-created
    └── (timestamped scenario or batch dirs)
```

---

## Configuration

Edit `config.yaml` to change:

- **Endpoints** (`step_1.endpoint`, `step_2.endpoint`)
- **Cost estimates** (used for pre-flight cost confirmation)
- **PuLID and Qwen default params** (overridable per-scenario by Opus prompt builders)

The Step 1 master prompt (`prompts/master_prompt_step1.md`) emits per-scenario `fal_pulid_params` (including scenario-specific `id_weight`, `guidance_scale`, `true_cfg`, etc.) — these override the defaults in `config.yaml`.

---

## Reliability

All batch runs:
- **Pre-flight check** — verifies all input files + env vars BEFORE any API call
- **Fail-safe per scenario** — single scenario error doesn't kill the batch
- **Mid-batch HTML refresh** — `overview.html` rewritten every 5 scenarios so you can monitor live
- **Ctrl+C safe** — partial outputs preserved with `interrupted: true` flag
- **Cost confirmation** — pre-flight prompt shows estimated cost (skip with `--yes`)
- **fal upload caching** — `cache/fal_uploads.json` keys uploads by absolute path, no redundant re-uploads

---

## Why this architecture works

Two-stage compositing — "**lock identity, free posture**" — consistently produces better identity fidelity than asking any single model to do persona + product compositing in one call. PuLID excels at identity preservation (~99% face match) but struggles with simultaneously inserting a product reference. Qwen-Image-Edit excels at adding objects to existing scenes but drifts on identity if asked to generate from scratch. Splitting the problem lets each model do what it does best.

The Qwen-tuned Step 2 master prompt is the result of v1→v5 iteration against documented Qwen-Image-Edit-2511 failure modes (mirrored text, position drift, duplicate products, anatomy artifacts). See `prompts/master_prompt_step2_qwen.md` for the operating principles, banned phrases, and anti-examples.