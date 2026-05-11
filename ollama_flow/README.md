# Alluvi — Ollama Flow (Local LLM Iteration)

This folder is a **self-contained Ollama-based iteration mode** for the Alluvi
image generation pipeline. It does everything the production Claude flow at
the parent repo does — same scenarios, same fal Stage 1 (PuLID), same fal
Stage 2 (Qwen-Image-Edit), same DB schema, same `chain.html` and
`overview.html` reports — but swaps Claude Opus 4.7 for a **local Ollama
model (default: `qwen2.5:7b`)** as the prompt builder.

## Why this exists

| | Claude flow (parent) | Ollama flow (this folder) |
|---|---|---|
| LLM | Claude Opus 4.7 (cloud) | Local Ollama (qwen2.5:7b by default) |
| Cost per scenario | ~$0.36 | ~$0.08 (fal only, LLM is free) |
| Cost per 30-scenario batch | ~$10.80 | ~$2.40 |
| Wall time per scenario | ~70-80s | ~80-120s (Ollama adds 5-15s/call) |
| Prompt quality | Production-grade | Noticeably weaker (see "Expected quality" below) |
| API keys needed | `FAL_KEY` + `ANTHROPIC_API_KEY` | `FAL_KEY` only |
| Network needed | Yes (Anthropic + fal) | Yes (fal only, LLM is local) |
| Recommended for | Production ad-image generation | Pipeline iteration, scenarios.yaml changes, DB testing |

**Use this folder when** you want to iterate freely — change scenarios, tweak
the master prompts, test new fal endpoint params, validate DB writes,
verify `chain.html` rendering — without burning ~$10.80 per 30-scenario
batch on Anthropic credits.

**Use the parent Claude flow when** you're ready to ship actual ad-image
frames into your video pipeline.

The two flows are **fully isolated**: the Ollama flow has its own DB
(`ollama_flow/data/alluvi_ollama.db`), its own outputs folder, its own
config. Production data is never touched.

---

## Prerequisites

### 1. Ollama installed and the model pulled

Download Ollama from <https://ollama.com/download> and install it. On Windows
the installer registers a service that auto-starts. Then pull the model:

```powershell
ollama pull qwen2.5:7b
```

This downloads roughly 4.7 GB. One-time only.

Verify Ollama is reachable (from any terminal):

```powershell
curl http://localhost:11434/api/tags
```

You should get back JSON listing your pulled models. If you see `qwen2.5:7b`
in the list, you're set.

### 2. Python deps (already in parent `requirements.txt`)

Nothing new to install — the Ollama flow uses libraries the parent already
needs (`httpx` for the `/api/generate` call, `python-dotenv`, `PyYAML`,
`fal-client`). From the parent repo root:

```powershell
pip install -r requirements.txt
```

### 3. fal API key

This folder calls fal.ai for Stage 1 (PuLID) and Stage 2 (Qwen-Image-Edit) —
the LLM is local but the image generation is still cloud-hosted. Grab your
key from <https://fal.ai/dashboard/keys>.

**You do NOT need an Anthropic key for this folder.** And **Ollama does not
have an API key** — it's a local server, no auth.

---

## Setup

### Create `.env` in this folder

```powershell
cd ollama_flow
copy .env.example .env
notepad .env
```

In `.env`, replace the placeholder with your real fal key. The only
required line is:

```
FAL_KEY=your_actual_fal_key_here
```

The `OLLAMA_*` lines in `.env.example` are all commented out because the
defaults work for a standard Ollama install:

- `OLLAMA_HOST=http://localhost:11434` (default Ollama address)
- `OLLAMA_MODEL=qwen2.5:7b` (matches what you pulled)
- `OLLAMA_TIMEOUT_SECONDS=180` (3 min per call)

Uncomment and edit these only if you're running Ollama on a different host,
using a different model, or your first call is timing out while the model
loads.

---

## Running the flow

All commands below assume you're in the `ollama_flow/` folder:

```powershell
cd D:\video_automation_prototype\Final_Image_generation\ollama_flow
```

### Step 1 — Preflight

```powershell
python preflight_ollama.py
```

Runs 7 checks:

1. `.env` keys — confirms `FAL_KEY` is present
2. Ollama reachability — pings `/api/tags`, confirms `qwen2.5:7b` is pulled
3. Shared asset files in parent (`../assets/persona.jpg`, etc.)
4. Shared brand + scenarios + prompts in parent
5. `scenarios.yaml` structure (via parent's `scenario_loader`)
6. Python dependencies (`fal_client`, `httpx`, `dotenv`, `yaml`, `sqlite3`)
7. Module imports (parent's infrastructure + this folder's Ollama builders)

If you see `OLLAMA-FLOW PREFLIGHT PASSED` at the bottom, you're cleared to run.

### Step 2 — Single scenario test (~$0.08, ~90s)

Recommended for first run. Picks one known-good scenario and walks the full
pipeline end-to-end:

```powershell
python run_ollama.py --scenario bedroom_robe_with_product_13
```

When it finishes, open the report:

```
ollama_flow/outputs/<timestamp>_bedroom_robe_with_product_13_ollama/chain.html
```

That HTML page shows: scenario YAML, Step 1 prompt (built by Ollama), the
PuLID-generated persona image, Step 2 prompt (built by Ollama), the final
composited image.

### Step 3 — Pilot batch (~$0.40, ~8 min)

Five hand-picked scenarios that exercise different archetypes:

```powershell
python run_batch_ollama.py --pilot
```

Open `ollama_flow/outputs/<timestamp>_batch_ollama/overview.html` to see
all 5 side-by-side.

### Step 4 — Full batch (~$2.40, ~50 min)

All 30 scenarios:

```powershell
python run_batch_ollama.py
```

Ctrl+C is safe at any point — `overview.html` and `batch_manifest.json` get
written for completed scenarios so far. The overview also auto-refreshes
every 5 scenarios during the batch.

### Useful batch flags

```powershell
python run_batch_ollama.py --only gym_post_workout_mirror_01,pilates_reformer_mirror_06
python run_batch_ollama.py --exclude yoga_home_practice_07
python run_batch_ollama.py --yes                 # skip cost confirmation
python run_batch_ollama.py --skip-preflight      # not recommended
```

---

## Expected quality (be ready for this)

`qwen2.5:7b` is roughly **two orders of magnitude smaller** than Claude
Opus 4.7. Expect the following common failure modes — each one logged as a
visible `WARNING` line during the run:

### Step 1 (lower stakes — fewer required clauses)

- **Word count drift** — target is 130-160; expect occasional 80-word or
  220-word outputs.
- **Persona descriptor paraphrasing** — Opus copies `face_descriptor_short`
  verbatim. Ollama may say "Mediterranean woman" instead of "25-year-old
  Mediterranean woman with sun-kissed deep-tan skin, green almond eyes…"
- **Missing photoreal anchors** — sometimes drops "candid amateur smartphone
  snapshot" or "Real photograph, not AI-generated".
- **Hallucinating product mention** — Step 1 is supposed to leave the
  product slot empty, but small models sometimes mention "the box" anyway.
  Caught by the warning system.

### Step 2 (higher stakes — many required clauses)

The Step 2 master prompt has 6+ required clauses. A 7B model **will drop at
least one per output**, often more. The builder warns for each:

- Positional reference syntax ("the person from the first image" / "the
  product from the second image") — Qwen-Image-Edit requires this exact
  phrasing
- "Keep X unchanged" anchors in Sentence 1
- Rigid-rotation orientation clause in Sentence 2
- Two-leg + occlusion anatomy clause in Sentence 3
- Single-product clause in Sentence 4
- White base preservation clause

### What this means in practice

The pipeline **still runs end-to-end either way**. fal accepts whatever
prompt the LLM produces. The final images will work — they'll just have
more compositing artifacts than the Claude versions (wrong product
orientation, extra fingers, occasional double-product, etc.).

**For iteration this is fine.** You verify plumbing, scenarios changes, DB
writes, `chain.html` rendering — all for ~$0.08/scenario. When you're ready
to ship, switch back to the parent Claude flow.

The DB tags every run with its provider:

```sql
-- Production Claude runs
SELECT * FROM runs WHERE plan = 'pulid_qwen_tuned';

-- Ollama iteration runs
SELECT * FROM runs WHERE plan = 'pulid_qwen_tuned_ollama';
```

Note the **two DBs are separate files**, so you'd need to query each
independently:

- Production: `Final_Image_generation/data/alluvi.db`
- Ollama: `Final_Image_generation/ollama_flow/data/alluvi_ollama.db`

---

## File layout

```
Final_Image_generation/                  ← parent repo (Claude flow, untouched)
├── config.yaml
├── .env
├── preflight.py
├── run.py
├── run_batch.py
├── data/alluvi.db                       ← production DB (Claude)
├── outputs/                             ← production outputs (Claude)
├── assets/                              ← SHARED — both flows read this
├── brand/                               ← SHARED
├── prompts/                             ← SHARED — master_prompt_step1.md etc.
├── scenarios/                           ← SHARED — scenarios.yaml
├── src/                                 ← parent's modules (Claude flow + shared infrastructure)
│   ├── db.py                            ← SHARED — same schema
│   ├── scenario_loader.py               ← SHARED
│   ├── step_1_pulid.py                  ← SHARED — fal PuLID caller
│   ├── step_2_qwen_edit.py              ← SHARED — fal Qwen caller
│   ├── step_1_prompt_builder.py         ← Claude-only
│   ├── step_2_prompt_builder.py         ← Claude-only
│   ├── trace_html.py                    ← SHARED
│   └── overview_html.py                 ← SHARED
│
└── ollama_flow/                         ← THIS FOLDER (self-contained)
    ├── README.md                        ← this file
    ├── config.yaml                      ← Ollama-specific config
    ├── .env                             ← FAL_KEY only
    ├── .env.example
    ├── preflight_ollama.py
    ├── run_ollama.py
    ├── run_batch_ollama.py
    ├── data/alluvi_ollama.db            ← Ollama DB (auto-created)
    ├── outputs/                         ← Ollama outputs (auto-created)
    └── ollama_src/                      ← our local Python package
        ├── __init__.py                  ← (named `ollama_src/` not `src/`
        ├── ollama_client.py                to avoid name collision with
        ├── step_1_prompt_builder_ollama.py  parent's src/ package on sys.path)
        └── step_2_prompt_builder_ollama.py
```

The Ollama flow reads **shared assets/brand/prompts/scenarios** from the
parent (no duplication) and reuses **parent's infrastructure modules**
(`db.py`, `step_1_pulid.py`, `step_2_qwen_edit.py`, `trace_html.py`,
`overview_html.py`) — only the LLM call swaps out.

---

## Troubleshooting

### `Cannot connect to Ollama at http://localhost:11434`

Ollama isn't running. Open a separate terminal and run:

```powershell
ollama serve
```

(On Windows, Ollama usually runs as a background service after install, but
sometimes the service gets stopped. The above forces it back up.)

Verify with:

```powershell
curl http://localhost:11434/api/tags
```

### `Ollama returned 404` / `model 'qwen2.5:7b' is not pulled`

You haven't pulled the model. Run:

```powershell
ollama pull qwen2.5:7b
```

### `Ollama at http://localhost:11434 did not respond within 180s`

The model is loading for the first time in this session. Either:

- Wait — the next call will be fast once it's in RAM.
- Or set a longer timeout in `.env`:

  ```
  OLLAMA_TIMEOUT_SECONDS=300
  ```

### `Could not extract valid JSON from model output`

`qwen2.5:7b` produced output the defensive JSON parser couldn't recover.
This is rare but happens — small models occasionally produce non-JSON
prose. Just re-run the scenario:

```powershell
python run_ollama.py --scenario <id>
```

Different seed, usually fine on the retry. If the same scenario fails
repeatedly, the master prompt may need adjustment for small-model
robustness — open `prompts/master_prompt_step1.md` (or
`master_prompt_step2_qwen.md`) and consider whether the structure is too
complex for 7B.

### Lots of `WARNINGS for <scenario>` in the output

That's expected — see "Expected quality" above. A 7B model will drop some
clauses. The pipeline runs anyway. If the dropped clauses matter for your
iteration (e.g., you're specifically testing the rigid-rotation behavior),
either:

- Re-run the scenario (different seed often produces a different set of
  drops)
- Try a larger model: `ollama pull qwen2.5:14b` and set
  `OLLAMA_MODEL=qwen2.5:14b` in `.env`
- Switch to the parent Claude flow for that scenario

### Confusion about which DB / outputs to look at

Always check the path printed at the top of the run output. The Ollama flow
prints:

```
DB: D:\video_automation_prototype\Final_Image_generation\ollama_flow\data\alluvi_ollama.db
```

If the path doesn't contain `ollama_flow`, you ran the parent's `run.py` by
mistake.

---

## When to switch back to Claude

You're done iterating with Ollama when:

1. `scenarios.yaml` is stable — no more edits planned.
2. Master prompts are stable — no more edits to
   `prompts/master_prompt_step1.md` or `master_prompt_step2_qwen.md`.
3. Pipeline plumbing is verified — preflight passes, single scenarios run
   clean, `chain.html` renders correctly, DB writes are consistent.
4. You've validated the fal endpoints and parameters are correct (look at
   actual generated images, not just success/fail).

Then, from the parent repo root:

```powershell
cd D:\video_automation_prototype\Final_Image_generation
python preflight.py
python run_batch.py
```

The production flow uses the exact same scenarios, master prompts, fal
endpoints, and infrastructure — only the LLM differs. Anything that works
in Ollama mode will work in Claude mode, just with sharper prompts.