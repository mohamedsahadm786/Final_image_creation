# Alluvi — Final Image Generation Pipeline

End-to-end image generation pipeline that produces premium-looking TikTok ad images for the **Alluvi Tirzepatide 40mg** product. Two-stage architecture, two LLM backends, dynamic prompt generation, automated quality control.

---

## Table of Contents

1. [What this is](#what-this-is)
2. [How it works (pipeline stages)](#how-it-works-pipeline-stages)
3. [Two flows: Claude vs Ollama](#two-flows-claude-vs-ollama)
4. [Repository structure](#repository-structure)
5. [Prerequisites](#prerequisites)
6. [Installation (clone → venv → dependencies)](#installation-clone--venv--dependencies)
7. [Configuration (.env files)](#configuration-env-files)
8. [Running the Claude flow](#running-the-claude-flow-recommended-for-production)
9. [Running the Ollama flow](#running-the-ollama-flow-free-iteration)
10. [Output directory structure](#output-directory-structure)
11. [How JSON sanity check + retry works](#how-json-sanity-check--retry-works)
12. [How QC validation + retry works (Claude flow only)](#how-qc-validation--retry-works-claude-flow-only)
13. [Cost summary](#cost-summary)
14. [Troubleshooting](#troubleshooting)

---

## What this is

A 2-stage image generation system that:

- **Stage 1** — Generates a person-in-scene image (persona + outfit + setting, no product yet) using `fal-ai/flux-pulid`. The persona's identity (face) is locked to ~99% fidelity using a reference photo at `assets/persona.jpg`.
- **Stage 2** — Composites the Alluvi product naturally into her hand (or onto a surface) using `fal-ai/qwen-image-edit-2511`. Single product, correct orientation, no anatomy defects.

The text prompt for each stage is generated **dynamically per scenario** by an LLM (Claude Opus 4.7 OR a local Ollama model), using hand-tuned master prompts that encode our quality rules.

The pipeline also includes:

- **JSON sanity check + retry** — catches malformed LLM output and retries the same scenario
- **QC validation + retry** (Claude flow only) — uses Claude Sonnet 4.6 vision to reject obviously broken images (3 hands, 6 fingers, warped products) and re-run Stage 2 up to 2 times before skipping
- **SQLite tracking** — every run, every generation, every retry recorded in `data/alluvi.db`
- **Per-scenario HTML traces** — `chain.html` shows scenario → Step 1 prompt → persona image → Step 2 prompt → final image side-by-side
- **Batch overview HTML** — `overview.html` shows all 30 scenarios in one page for visual review

---

## How it works (pipeline stages)

```
  scenarios.yaml entry
         │
         │  (one of 30 hand-curated scenes: persona pose + outfit
         │   + setting + product placement spec)
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 1. STEP 1 PROMPT BUILDER                                │
  │    LLM (Opus 4.7 or qwen2.5:7b)                         │
  │    + master_prompt_step1.md                             │
  │    + persona.yaml + product.yaml + scenario data        │
  │    →                                                    │
  │    step_1_image_prompt   (130-160 words, no product)    │
  │    fal_pulid_params      (id_weight, guidance, etc.)    │
  │                                                         │
  │    🔁 JSON retry if output isn't valid JSON             │
  └─────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 2. STAGE 1: fal-ai/flux-pulid                           │
  │    + assets/persona.jpg (face reference)                │
  │    →                                                    │
  │    03_step1_persona.jpg                                 │
  │    (persona in scene, correct outfit, no product)       │
  └─────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 3. STEP 2 PROMPT BUILDER                                │
  │    LLM (Opus 4.7 or qwen2.5:7b)                         │
  │    + master_prompt_step2_qwen.md                        │
  │    + scenario data + step 1 output                      │
  │    →                                                    │
  │    step_2_image_prompt   (320-410 words, product-aware) │
  │    fal_qwen_params       (guidance, steps, etc.)        │
  │                                                         │
  │    🔁 JSON retry if output isn't valid JSON             │
  └─────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 4. STAGE 2: fal-ai/qwen-image-edit-2511                 │
  │    + 03_step1_persona.jpg + brand/box_front.jpg         │
  │    →                                                    │
  │    05_step2_final.jpg                                   │
  │    (final image — persona holding Alluvi product)       │
  └─────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 5. QC VALIDATION  (Claude flow only)                    │
  │    Claude Sonnet 4.6 vision check                       │
  │    Lenient rules: only obviously illogical defects fail │
  │      - 3+ legs / 3+ hands / 3+ arms                     │
  │      - 6+ fingers on a hand                             │
  │      - Fused limbs into impossible shapes               │
  │      - Multiple distinct product copies                 │
  │      - Product warped/melted                            │
  │                                                         │
  │    🔁 If QC fails: re-run Stage 2 only (same persona)   │
  │       up to 2 retries. After 3 total attempts: SKIP.    │
  └─────────────────────────────────────────────────────────┘
         │
         ▼
   06_qc_result.json + chain.html + DB record
```

---

## Two flows: Claude vs Ollama

The pipeline has **two parallel implementations** with different cost/quality trade-offs. Both produce the same output structure.

| Aspect                       | Claude flow (`run.py`)           | Ollama flow (`ollama_flow/run_ollama.py`) |
|------------------------------|----------------------------------|-------------------------------------------|
| **Purpose**                  | Production batches                | Free iteration / prompt tweaking          |
| **Prompt LLM**               | Claude Opus 4.7 (API)             | Local Ollama (default `qwen2.5:7b`)       |
| **Stage 1 model**            | `fal-ai/flux-pulid`               | `fal-ai/flux-pulid` (same)                |
| **Stage 2 model**            | `fal-ai/qwen-image-edit-2511`     | `fal-ai/qwen-image-edit-2511` (same)      |
| **Prompt cost per scenario** | ~$0.28                            | $0 (local)                                |
| **fal cost per scenario**    | ~$0.08                            | ~$0.08 (same)                             |
| **JSON sanity check**        | ✅ + 1 retry                       | ✅ + 2 retries                             |
| **QC validation**            | ✅ Sonnet 4.6 vision               | ❌ none                                    |
| **QC retry (Stage 2 only)**  | ✅ up to 2 retries → skip          | ❌ none                                    |
| **Cost per 30-scenario run** | ~$12                              | ~$2.40                                    |
| **Wall time (30 sequential)**| ~35-55 min                        | ~50-60 min                                |
| **API keys required**        | `FAL_KEY` + `ANTHROPIC_API_KEY`   | `FAL_KEY` only                            |

**Use the Claude flow when** you're producing the actual ad images for downstream video generation. The QC step filters out broken images so only usable ones reach the video pipeline.

**Use the Ollama flow when** you're iterating on scenarios, master prompts, or testing pipeline changes. No API costs for the LLM portion, faster turnaround, but no quality gate.

---

## Repository structure

```
Final_Image_generation/
├── README.md                     ← this file
├── requirements.txt              ← Python dependencies (both flows)
├── .env.example                  ← template for API keys
├── .gitignore
├── config.yaml                   ← Claude flow config (model, defaults)
│
├── run.py                        ← Claude flow: single scenario
├── run_batch.py                  ← Claude flow: batch of scenarios
├── preflight.py                  ← Claude flow: pre-run sanity check
│
├── assets/
│   └── persona.jpg               ← face reference for PuLID (~1024×1024)
├── brand/
│   └── box_front.jpg             ← Alluvi product reference for Qwen
├── prompts/
│   ├── master_prompt_step1.md    ← Step 1 system prompt
│   ├── master_prompt_step2_qwen.md  ← Step 2 system prompt
│   ├── persona.yaml              ← persona descriptors (face, hair, build)
│   └── product.yaml              ← product descriptors (box, text, colors)
├── scenarios/
│   └── scenarios.yaml            ← 30 scenarios in 11 categories
│
├── src/                          ← shared modules used by BOTH flows
│   ├── __init__.py
│   ├── db.py                     ← SQLite tracker (runs + generations)
│   ├── json_utils.py             ← shared JSON sanity-check validator
│   ├── qc_validator.py           ← Sonnet 4.6 QC validator (Claude flow uses this)
│   ├── scenario_loader.py        ← scenarios.yaml parser + validation
│   ├── step_1_prompt_builder.py  ← Claude Opus → Step 1 prompt JSON
│   ├── step_1_pulid.py           ← fal PuLID caller
│   ├── step_2_prompt_builder.py  ← Claude Opus → Step 2 prompt JSON
│   ├── step_2_qwen_edit.py       ← fal Qwen-Image-Edit caller
│   ├── trace_html.py             ← chain.html generator
│   └── overview_html.py          ← overview.html generator (batch view)
│
├── data/                         ← created on first run
│   └── alluvi.db                 ← SQLite database (runs + generations + retries)
│
├── outputs/                      ← created per run
│   ├── <ts>_<scenario_id>/       ← single-scenario outputs
│   └── <ts>_batch/               ← batch outputs
│
└── ollama_flow/                  ← self-contained Ollama flow
    ├── README.md                 ← Ollama-specific notes
    ├── config.yaml               ← Ollama flow config (model, host, timeout)
    ├── .env.example
    ├── run_ollama.py             ← Ollama flow: single scenario (no QC)
    ├── run_batch_ollama.py       ← Ollama flow: batch (no QC)
    ├── preflight_ollama.py       ← Ollama flow: checks Ollama is running
    ├── ollama_src/               ← Ollama-specific prompt builders
    │   ├── __init__.py
    │   ├── ollama_client.py      ← thin HTTP client for Ollama /api/generate
    │   ├── step_1_prompt_builder_ollama.py
    │   └── step_2_prompt_builder_ollama.py
    └── outputs/                  ← Ollama flow's outputs (isolated)
```

---

## Prerequisites

| Tool         | Version    | Notes                                              |
|--------------|------------|----------------------------------------------------|
| Python       | 3.10+      | tested on 3.12 Windows                             |
| pip          | latest     | comes with Python                                  |
| Git          | any        | for cloning                                        |
| fal account  | active     | get key from https://fal.ai/dashboard/keys         |
| Anthropic    | active     | **Claude flow only** — https://console.anthropic.com |
| Ollama       | latest     | **Ollama flow only** — https://ollama.com/download |

---

## Installation (clone → venv → dependencies)

The following PowerShell commands assume Windows. For macOS/Linux replace `\` with `/` and use `source .venv/bin/activate` instead of the Windows activation line.

```powershell
# 1. Clone the repo
git clone <YOUR_REPO_URL> video_automation_prototype
cd video_automation_prototype\Final_Image_generation

# 2. Create a virtual environment in the parent folder
#    (we keep the venv outside Final_Image_generation/ so editor indexers
#    don't crawl it. Adjust if you prefer it inside.)
cd ..
python -m venv venv

# 3. Activate it
.\venv\Scripts\Activate.ps1

# If you get an execution-policy error, run this ONCE per machine:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then re-run the activation line.

# 4. Install dependencies
cd Final_Image_generation
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs:
- `anthropic` — Claude API client (used by Claude flow + QC validator)
- `fal-client` — fal API client (both flows use this for image generation)
- `httpx` — Ollama HTTP client (used by Ollama flow)
- `python-dotenv` — `.env` file loader
- `PyYAML` — config + scenarios parser
- `requests` — assorted HTTP

Verify the install:
```powershell
python -c "import anthropic, fal_client, httpx, yaml, dotenv; print('all imports ok')"
```

---

## Configuration (.env files)

The two flows use separate `.env` files so the Ollama flow stays isolated (you can run it without an Anthropic key).

### Claude flow `.env`

Create `Final_Image_generation\.env`:

```env
FAL_KEY=your-fal-key-here
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

Both keys are required for the Claude flow because:
- `FAL_KEY` → calls PuLID (Stage 1) + Qwen-Image-Edit (Stage 2)
- `ANTHROPIC_API_KEY` → calls Opus (prompt builds) + Sonnet (QC validation)

### Ollama flow `.env`

Create `Final_Image_generation\ollama_flow\.env`:

```env
FAL_KEY=your-fal-key-here

# Optional Ollama overrides — defaults shown
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODEL=qwen2.5:7b
# OLLAMA_TIMEOUT_SECONDS=180
```

Only `FAL_KEY` is required for the Ollama flow. **No Anthropic key needed** — there is no QC in the Ollama flow and prompt building uses your local Ollama server.

---

## Running the Claude flow (recommended for production)

### Step 1: Preflight (always run this first)

```powershell
cd D:\path\to\Final_Image_generation

# Activate venv if not already active
..\venv\Scripts\Activate.ps1

# Preflight (~10 seconds)
python preflight.py
```

What preflight checks:
- `assets/persona.jpg` exists and is a valid image
- `brand/box_front.jpg` exists
- `prompts/master_prompt_step1.md` and `master_prompt_step2_qwen.md` exist
- `scenarios/scenarios.yaml` parses and has at least 1 scenario
- All scenarios have required fields and valid persona/outfit references
- `FAL_KEY` and `ANTHROPIC_API_KEY` are present in environment
- fal API key is valid (sends a dry probe)
- Anthropic API key is valid (sends a tiny test request, ~$0.0001)

If any check fails, preflight prints the exact issue and exits non-zero. Fix and re-run.

### Step 2: Single-scenario test

Run one scenario end-to-end to confirm everything works before spending money on a batch:

```powershell
python run.py --scenario bedroom_robe_with_product_13
```

- Wall time: ~90–120 seconds (best case, QC passes first try)
- Cost: ~$0.37 best case / ~$0.45 worst case (2 QC retries)
- Output folder: `outputs\<timestamp>_bedroom_robe_with_product_13\`

To use a different scenario, pick an ID from `scenarios/scenarios.yaml`:
```powershell
python run.py --scenario kitchen_loungewear_morning_8
```

### Step 3: Pilot batch (5 scenarios)

Once a single scenario looks good, run a 5-scenario pilot to catch issues that only appear across multiple personas/scenes:

```powershell
python run_batch.py --pilot
```

- Wall time: ~10–15 minutes
- Cost: ~$2.00
- Output folder: `outputs\<timestamp>_batch\`
- Open `overview.html` in that folder to see all 5 scenarios side-by-side

### Step 4: Full batch (all 30 scenarios)

When the pilot looks clean:

```powershell
python run_batch.py
```

- Wall time: ~45–60 minutes
- Cost: ~$10.80–$13 depending on QC retry rate
- Output folder: `outputs\<timestamp>_batch\`
- `overview.html` updates mid-batch so you can monitor progress in your browser

### Useful batch flags

```powershell
# Skip cost confirmation prompt
python run_batch.py --yes

# Run only specific scenarios
python run_batch.py --only bedroom_robe_with_product_13,kitchen_loungewear_8

# Skip specific scenarios
python run_batch.py --exclude gym_outdoor_running_22

# Skip preflight (NOT recommended — only for debugging)
python run_batch.py --skip-preflight
```

---

## Running the Ollama flow (free iteration)

The Ollama flow runs prompt building on your local machine for $0 cost. There is **no QC step** in this flow — you visually inspect the outputs yourself.

### One-time setup: Install Ollama and pull the model

1. **Download Ollama** from https://ollama.com/download (Windows/macOS/Linux installer)
2. **Open a NEW terminal** and start the Ollama server:
   ```powershell
   ollama serve
   ```
   Leave this terminal running. It must stay up while you run the pipeline.
3. **In another terminal, pull the default model** (only first time):
   ```powershell
   ollama pull qwen2.5:7b
   ```
   This downloads ~4.7 GB. After this, you don't need internet for the prompt LLM.

   Want a different model? See https://ollama.com/library for options like `llama3.1:8b`, `mistral:7b`, etc. To use a different one, set `OLLAMA_MODEL` in `ollama_flow\.env`.

4. **Verify it's working**:
   ```powershell
   ollama list                                    # should show qwen2.5:7b
   curl http://localhost:11434/api/tags           # should return JSON
   ```

### Step 1: Preflight

```powershell
cd D:\path\to\Final_Image_generation\ollama_flow

# Activate venv from parent
..\..\venv\Scripts\Activate.ps1

# Preflight (~10 seconds)
python preflight_ollama.py
```

What preflight checks:
- All the file/scenario checks the Claude flow does
- `FAL_KEY` is present
- Ollama server at `OLLAMA_HOST` is reachable
- The configured `OLLAMA_MODEL` is pulled and responsive (sends a small test query)
- **Does NOT** check for `ANTHROPIC_API_KEY` (not needed)

### Step 2: Single-scenario test

```powershell
python run_ollama.py --scenario bedroom_robe_with_product_13
```

- Wall time: ~60–90 seconds (Ollama is ~5-15s slower per call than Claude API)
- Cost: ~$0.08 (fal calls only — Ollama is free)
- Output folder: `ollama_flow\outputs\<timestamp>_bedroom_robe_with_product_13_ollama\`

### Step 3: Pilot batch (5 scenarios)

```powershell
python run_batch_ollama.py --pilot
```

- Wall time: ~8–12 minutes
- Cost: ~$0.40
- Output folder: `ollama_flow\outputs\<timestamp>_batch_ollama\`

### Step 4: Full batch (all 30 scenarios)

```powershell
python run_batch_ollama.py
```

- Wall time: ~50–60 minutes
- Cost: ~$2.40
- Output folder: `ollama_flow\outputs\<timestamp>_batch_ollama\`

### Useful batch flags (same as Claude flow)

```powershell
python run_batch_ollama.py --yes
python run_batch_ollama.py --only ID1,ID2
python run_batch_ollama.py --exclude ID3
python run_batch_ollama.py --skip-preflight
```

---

## Output directory structure

### Single scenario (Claude flow)

```
outputs\<ts>_<scenario_id>\
  ├── 01_scenario.yaml                  scenario JSON dump
  ├── 02_step1_prompt.json              Opus output for Step 1
  ├── 03_step1_persona.jpg              Stage 1 result (PuLID)
  ├── 03_step1_meta.json                fal call metadata for Stage 1
  ├── 04_step2_prompt.json              Opus output for Step 2
  ├── 05_step2_final.jpg                latest accepted image
  ├── 05_step2_final_attempt_1.jpg      each Qwen attempt kept
  ├── 05_step2_final_attempt_2.jpg      (only if retried)
  ├── 05_step2_final_attempt_3.jpg      (only if retried twice)
  ├── 05_step2_meta.json                fal call metadata for Stage 2
  ├── 06_qc_result_attempt_1.json       QC verdict per attempt
  ├── 06_qc_result_attempt_2.json
  ├── 06_qc_result_attempt_3.json
  ├── 06_qc_result.json                 final QC verdict + attempts log
  └── chain.html                        open in browser to review
```

### Single scenario (Ollama flow)

```
ollama_flow\outputs\<ts>_<scenario_id>_ollama\
  ├── 01_scenario.yaml
  ├── 02_step1_prompt.json              Ollama output for Step 1
  ├── 03_step1_persona.jpg
  ├── 03_step1_meta.json
  ├── 04_step2_prompt.json              Ollama output for Step 2
  ├── 05_step2_final.jpg                single attempt — no retry
  ├── 05_step2_meta.json
  └── chain.html
```

No `06_qc_result.json` and no per-attempt files because there is no QC in this flow.

### Batch (Claude flow)

```
outputs\<ts>_batch\
  ├── overview.html                     scenario grid — open this first
  ├── batch_manifest.json               machine-readable index
  └── <scenario_id>\                    one folder per scenario
        ├── (same layout as single Claude run above)
        └── chain.html
```

### Batch (Ollama flow)

```
ollama_flow\outputs\<ts>_batch_ollama\
  ├── overview.html
  ├── batch_manifest.json
  └── <scenario_id>\
        ├── (same layout as single Ollama run above)
        └── chain.html
```

---

## How JSON sanity check + retry works

LLMs occasionally return malformed JSON — code fences (` ```json `), leading prose ("Here is the JSON:"), trailing commas, or in rare cases an outright refusal. Smaller models (like `qwen2.5:7b`) drift more often than Claude Opus.

### The validator (`src/json_utils.py`)

A single function `validate_json_output(text, required_keys=[...])` does all the work, used by BOTH flows. It runs four defensive strategies in order:

1. **Strip markdown fences** — removes `` ```json `` and `` ``` `` wrappers
2. **Find outermost balanced `{...}` object** — handles "Here is the JSON: {actual JSON} let me know if..." cases
3. **Strip trailing commas** — removes `,` before `}` or `]` (Python-valid but JSON-invalid)
4. **Validate required keys** — checks all required top-level keys are present and non-empty

If all strategies fail, it raises `JSONSanityError` with a snippet of the raw response for debugging.

### The retry loop

Inside `process_scenario`, prompt-build calls are wrapped in `_call_with_json_retry()`. It catches **only** `JSONSanityError` and retries the same call. Other exceptions (network errors, fal failures, file-not-found) propagate immediately — retrying those won't help.

| Flow   | Max retries | Total attempts | Cost per retry         |
|--------|-------------|----------------|------------------------|
| Claude | 1           | 2              | ~$0.14 (Opus call)     |
| Ollama | 2           | 3              | $0 (local)             |

After all attempts fail, that **one scenario** is marked failed in the DB and the batch continues. You do NOT lose the rest of the batch.

### What you'll see in the terminal

Successful first try (most scenarios):
```
[step_1_prompt_builder] Step 1 -> Opus 4.7 for scenario bedroom_robe_with_product_13
[step_1_prompt_builder]   Step 1 done: word_count=148, slot_type=held_product_low
```

Retry triggered:
```
[step_1_prompt_builder] Step 1 -> Opus 4.7 for scenario bedroom_robe_with_product_13
[run] bedroom_robe_with_product_13 Step 1: JSON sanity error on attempt 1/2: Could not extract valid JSON...
[run]   retrying same scenario...
[step_1_prompt_builder] Step 1 -> Opus 4.7 for scenario bedroom_robe_with_product_13
[step_1_prompt_builder]   Step 1 done: word_count=152, slot_type=held_product_low
```

---

## How QC validation + retry works (Claude flow only)

After Stage 2 produces the final image, the Claude flow runs an automated quality check using Claude Sonnet 4.6 vision. The Ollama flow does NOT run QC — that flow is for iteration only.

### The QC rubric (lenient mode)

We only flag **obviously illogical** defects. Minor imperfections (slight text blur, mild lighting drift, subtle face asymmetry) pass through.

**Hard fails** (image will be regenerated):
- `person_count != 1` — wrong number of people
- `has_extra_limbs` — 3+ arms, 3+ legs, or 3+ hands
- `has_extreme_finger_issue` — 6+ fingers on a hand, or fingers fused into a mass
- `has_fused_or_warped_limbs` — impossible body shapes
- `face_grossly_distorted` — multiple faces, severely distorted, or missing
- `product_visible == false` — product missing entirely
- `multiple_distinct_products` — 2+ separate product copies (mirror reflections count as 1)
- `product_shape_broken` — product warped into non-rectangular shape

**Pass-through** (ignored):
- Minor text artifacts on packaging (e.g., "ALUUVI" instead of "ALLUVI")
- Slight lighting inconsistencies
- Small face asymmetry
- Finger curl in normal-count hands
- Background imperfections

### The retry loop

When QC fails, the pipeline re-runs **only Stage 2** (Qwen-Image-Edit) using the same persona image from Stage 1. PuLID is NOT re-run — that would cost more and risk drifting the persona.

| Attempt | What runs | If QC fails              |
|---------|-----------|--------------------------|
| 1       | Qwen      | Retry once               |
| 2       | Qwen      | Retry once more          |
| 3       | Qwen      | **Skip scenario**, mark `qc_failed` in DB |

After all 3 attempts fail, the scenario gets `final_status = qc_failed` in the database. The batch continues to the next scenario. The skipped scenario is NOT eligible for the video generation pipeline downstream.

### What you'll see in the terminal

QC passes first try (most scenarios):
```
[step_2_qwen]    composited in 6.2s
[qc_validator] bedroom_robe_with_product_13#a1: running QC via claude-sonnet-4-6-20250929...
[qc_validator] bedroom_robe_with_product_13#a1: PASS (score=1.00, issues=0, rec=use)
[run] bedroom_robe_with_product_13: QC PASSED on attempt 1
```

QC fails once then passes:
```
[qc_validator] bedroom_robe_with_product_13#a1: FAIL (score=0.80, issues=1, rec=regenerate)
  - obviously extra limbs detected (3+ arms/legs/hands)
[run] bedroom_robe_with_product_13: QC failed on attempt 1/3 — retrying Stage 2 only
[step_2_qwen]    composited in 5.8s
[qc_validator] bedroom_robe_with_product_13#a2: PASS (score=1.00, issues=0, rec=use)
[run] bedroom_robe_with_product_13: QC PASSED on attempt 2
```

QC fails all 3 attempts:
```
[qc_validator] bedroom_robe_with_product_13#a3: FAIL (score=0.80, issues=1)
[run] bedroom_robe_with_product_13: QC failed on final attempt 3/3 — skipping scenario
======================================================================
 QC_FAILED
======================================================================
  qc:         FAILED after 3 attempts
              - obviously extra limbs detected
  outcome:    image skipped, not eligible for video pipeline
```

### Why Sonnet 4.6 instead of a local detector

Standard image-quality metrics (FID, IS, CLIP-score) don't detect anatomical distortions. Pose-landmark methods like MediaPipe are trained to find 21 keypoints on a normal hand — they output 5 plausible landmarks even on a 7-fingered hand, because they LOCATE expected keypoints rather than VALIDATE anatomy. A vision-language model looks at the actual pixels and can count.

Cost: ~$0.01 per QC check, ~$0.30 per 30-scenario batch.

### Disabling QC

If you want to skip QC for a specific run (debugging, testing without API costs):

```powershell
$env:QC_ENABLED = "false"
python run.py --scenario bedroom_robe_with_product_13
```

When QC is disabled, the first Qwen attempt is always accepted and saved as `05_step2_final.jpg`.

---

## Cost summary

Per-scenario costs (approximate, depend on prompt length + retry rate):

| Flow   | Prompt LLM | Stage 1 (PuLID) | Stage 2 (Qwen) | QC      | Total best | Total worst (max retries) |
|--------|------------|-----------------|----------------|---------|------------|---------------------------|
| Claude | ~$0.28     | ~$0.04          | ~$0.04         | ~$0.01  | ~$0.37     | ~$0.45 (3 Qwen attempts)  |
| Ollama | $0         | ~$0.04          | ~$0.04         | $0      | ~$0.08     | ~$0.08 (no QC retry)      |

Per-batch costs (30 scenarios):

| Flow   | Cost range  | Wall time  |
|--------|-------------|------------|
| Claude | $11 – $14   | 45–60 min  |
| Ollama | ~$2.40      | 50–60 min  |

---

## Troubleshooting

### `ImportError: cannot import name 'db' from 'src'` (Ollama flow)

You probably have an old `ollama_flow/src/` folder colliding with the parent's `src/`. The Ollama flow uses `ollama_flow/ollama_src/` to avoid this. If you see this error, check that there is NO `ollama_flow/src/` directory.

### `OllamaError: Cannot connect to Ollama at http://localhost:11434`

The Ollama server isn't running. Open a separate terminal and run:
```powershell
ollama serve
```
Leave it running while you use the pipeline.

### `OllamaError: model not found, try pulling it`

Run:
```powershell
ollama pull qwen2.5:7b
```
Or whichever model you set in `OLLAMA_MODEL`.

### `RuntimeError: ANTHROPIC_API_KEY missing`

Either you're running the Claude flow without an Anthropic key, or you're trying to run QC. Add `ANTHROPIC_API_KEY=sk-ant-...` to your `.env` file, or disable QC with `$env:QC_ENABLED = "false"`.

### Preflight fails with "fal probe returned 401"

Your `FAL_KEY` is invalid or expired. Get a fresh one from https://fal.ai/dashboard/keys and update `.env`.

### A specific scenario keeps failing QC after 3 attempts

This is a real signal — Qwen-Image-Edit can't reliably handle that particular scene + product placement combo. Check `06_qc_result.json` in the scenario folder for the specific issues. Common fixes:
- Edit the scenario in `scenarios/scenarios.yaml` (change product placement, simplify pose)
- Increase the Qwen guidance scale in `config.yaml`
- Switch to a different scenario from the 30

### "all 30 attempts failed JSON sanity check" with Ollama

Your local model is producing malformed output consistently. Try:
- A larger model: `ollama pull llama3.1:8b` then set `OLLAMA_MODEL=llama3.1:8b`
- Switching to the Claude flow temporarily to verify the rest of the pipeline works

### PowerShell: `cannot be loaded because running scripts is disabled`

One-time fix per machine:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then re-run the venv activation.

### Outputs folder filling up

Each batch creates a timestamped folder under `outputs/`. Delete old runs periodically:
```powershell
# Keep only the 5 most recent batches
Get-ChildItem outputs\*_batch\ | Sort-Object LastWriteTime -Descending | Select-Object -Skip 5 | Remove-Item -Recurse
```

---

## License

[Add your license info here.]

## Contact

[Add your contact info here.]