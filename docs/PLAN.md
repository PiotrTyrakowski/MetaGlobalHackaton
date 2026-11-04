# MixPilot: plan (v0.6, 2026-11-04)

Working name. An agent built on Meta Muse Spark that takes a concrete plant's Excel files, gets the data ready with the user, then retrains and tunes Meta's BOxCrete model on that plant's data. The trained model stays saved so the plant can keep using it.

Background research (BOxCrete, Quadrel, Meta API): https://claude.ai/artifact/JUbkNNRZgWKYZgGmLEdSvJ

This version combines the two parallel builds: Michal's `alt-full-stack` (batch-level data, generic input testing, BoTorch AutoML with re-checks, Next.js frontend) and Piotr's `mixpilot-backend` (ID matching, row cleanup, button-only training, streaming, scripted LLM, tests).

## Constraints

- **Hackathon:** Meta Global AI Developer Hackathon, **Indie track**. Judged on creativity, technical ambition and whether it works.
- **Deadline:** submission due **Nov 4 2026, 9:00 PM ET**.
- **Meta Model API:** key in `backend/.env` (`META_API_KEY`, gitignored). `muse-spark-1.3` tool calling on the Responses API verified on 2026-11-03.
- **Compute:** runs locally. The laptop (Apple M5, 10 cores, 16 GB) is enough for BOxCrete: one full fit on Meta's 670 rows takes **~13 s** (measured 2026-11-03; upstream's "~85 s" is from CI). 12 inputs: ~40 s.
- **Out of scope:** a PR to upstream BOxCrete, cylinder photos / SAM, PDFs. The demo uses Excel files only.

## Demo pipeline

1. **Upload:** the plant uploads its Excel files. For the demo these are synthetic, each with its own column names and US units:
   - `mix_designs.xlsx`: recipes in lb/yd³;
   - `cylinder_breaks.xlsx`: one row per cylinder;
   - `daily_QC.xlsx`: QC log per batch.
2. **Prepare (back and forth with the agent):**
   - The agent maps the columns to BOxCrete's schema and converts units.
   - **Feature discovery is generic, not hard-coded.** Our code tests every extra column the files have; the agent explains the results and proposes the useful ones; the user approves before training. Rejected columns are shown as checked and rejected. Two kinds of column, two tests:
     - **Batch measurements** (air, mill cert, slump, truck…): does the column explain the scatter between batches *of the same recipe*? That scatter is exactly what BOxCrete's 10 inputs can't see. Permutation test within each design (5,000 shuffles, ~8 s, no model fit). ID-like columns that pass (ticket no., cement lot) are not used as inputs; the agent explains what they stand for (the cement lot, which the mill cert measures).
     - **Recipe ingredients BOxCrete has no slot for** (silica fume): honest model comparison, refit with vs without the column and vs a shuffled copy.
   - Tried and rejected: ranking by the baseline's errors on one-batch-per-mix data (chance correlations as big as the real effects on 46 mixes), and comparing BOxCrete's lengthscales against shuffled shadow columns (too noisy).
   - Columns that help but are incomplete get a different message: "if you recorded this from now on, the model would improve," with an estimate of the gain.
   - In the synthetic demo the agent should find:

     | Column | Coverage | Measured (2026-11-03) | Verdict |
     |---|---|---|---|
     | Air (%) | 100% | within-recipe r = 0.77, p < 0.001 | use it |
     | Silica fume (lb/cy) | 100% | with air: 507 → 450 psi | use it |
     | Mill cert 28d (psi) | 69% | r = 0.41, p < 0.001; 475 → 401 psi on batches that have it | track it / send it |
     | Cement lot, ticket no. | 100% | p = 0.002 (they track the lot) | not inputs; explain the lot effect |
     | Conc temp, slump, truck, operator | 100% | p = 0.09–0.79 | rejected |
   - Mill certs come **per cement lot**; 4 of 13 lots have none, and the Data step names them.
   - The incomplete column is not used in the saved model. Its value is estimated on the batches that have it (with vs without, same batches). Filling gaps from a later upload is cut: nothing can join a separate certificate file to the batches yet.
   - Cleanup is automatic and shown under Details: IDs match ignoring case and punctuation, cylinders a notes column marks as invalid are excluded, and breaks after 28 days are dropped. A mapping that leaves no usable tests fails with an error.
3. **Train (button):** an AutoML run on the plant's data only (Meta's 670 rows are not pooled). The **baseline is Meta's default setup retrained on the plant's data**; we always retrain, so Meta's untrained model is not shown. One button, fixed budget, no user knobs. Leaderboard rows:
   - Meta default, retrained (baseline);
   - plus tuned settings;
   - plus the approved new inputs;
   - plus both (the expected winner).

   Headline metric: **28-day error in psi**, 5-fold refit holding out whole mix designs; errors at all ages in a small table below. Measured on the demo data: Meta default 684 psi → + air 507 → + air + silica fume 450 (−34%), before tuning.
4. **Results:** what the model gives the plant, as a CO₂-vs-strength chart and the next mixes to test.
5. **Models tab:** the trained model is saved as a version (date, error, inputs used). The user can keep asking about it in chat ("what will this mix reach at 7 days?") or add new results and retrain.

**Rules:**
- The agent proposes; the human approves.
- The LLM only quotes numbers our code computed.

## UI principle

As simple as possible. **Everything on screen is either something we add on top of BOxCrete, or something that has to be there.** Nothing else. That means:
- the new inputs, the incomplete input, the rejected columns;
- three artifacts (Data, Training, Results) plus the Models tab;
- short agent messages (≤ 2 sentences; cards and panels carry the numbers);
- **extremely clean**: each panel shows only the headline and one table; everything else (sources, rejected columns, by-age errors, search chart, full recipes) sits behind a "Details" toggle; finished tool steps collapse to one line.

## AutoML: what we tune

**Fitted automatically by every BOxCrete fit (not our job):** lengthscales, output scale, noise, correlation between material sources.

**Set by hand in BOxCrete's code, so this is our search space:**

| Knob | Where | Default |
|---|---|---|
| Time-gate timescale | `features.py` `GATE_TAU` | 0.10 |
| Lengthscale shrinkage prior σ | `priors.py` | 0.001 |
| Training loss mix (block-LOO vs likelihood) | `block_loo.train_block_loo(mll_weight=…)` | 0.5 |
| Inputs | `utils.py` `DEFAULT_X_COLUMNS` | 10 columns; we add the approved new ones |
| Engineered features | `features.py` `F5_ALLLOG_FEATURES` | 7 log features (kept as is for the demo) |

Meta picked these defaults with their own sweeps on their 647 public rows (comments in `features.py` and `priors.py`). Our point: those values suit Meta's data, not necessarily this plant's. `GATE_TAU` was also capped at 0.10 only to keep their in-browser JS port matching Python; that constraint doesn't apply to us, so we search above it.

**Search:** our own AutoML loop, using **BoTorch** (Meta's Bayesian optimization library, which BOxCrete is built on) as the search engine: a GP over the knobs plus an expected-improvement acquisition. Each trial is scored by 5-fold refit (~17 s on the demo data; 119 batches, 357 rows, ~6 s per fit). About 15–25 trials, ~5–7 min. A finished run is cached so a live demo can't stall; the video speeds the run up.

**Score:** average over test ages of RMSE ÷ mean strength (early ages matter: 7-day breaks forecast 28-day). Headline on screen: 28-day RMSE in psi.

**Measured on the demo data (2026-11-03, 92 s):**

| Setup | All ages | 28 days | Folds won vs baseline | vs Meta's settings |
|---|---|---|---|---|
| Meta's default setup | 13.5% | 687 psi | – | – |
| + tuned settings | 13.2% | 690 psi | 6/10 | 6/10 |
| + air + silica fume | 8.6% | 422 psi | 10/10 | – |
| **+ tuned settings + air + silica fume** (winner) | **8.4%** | 425 psi | 10/10 | 7/10 |

Tuning's gain is small (−2.3% relative, mostly at 3 and 7 days); the inputs do most of the work. Tuned σ sits at the upper bound (0.3) of the search space.

**Scoring (guard against luck):**
- Every score is a **5-fold refit holding out whole mix designs**. BOxCrete's closed-form leave-one-mix-out score is not used: BOxCrete's training minimizes that same score, so on plant-sized data it rewards any extra column, even pure noise (measured: a no-effect column "improved" it 377 → 278 psi).
- After the search: the top 2 setups and Meta's default are re-checked on new fold splits. The leaderboard shows these numbers.
- A setup only wins if it beats the baseline on most folds.

### Adding new input columns (any number)

The output doesn't change, so the model still predicts strength. The patch accepts **any number of extra inputs**, not just air. **Done** (2026-11-03), see `backend/vendor/SustainableConcrete/VENDORED.md`:
- Extra columns go **after** Time, so every existing index (`features.IDX`, the binder/aggregate prior groups, the source dim) stays valid. The kernels already treat dims beyond the first 10 as extras with free lengthscales.
- Fixed: the input transform assumed Time is the last column; it now finds it by name.
- Fixed: leave-one-mix-out blocks now include the extra columns.
- `gate_tau` and `shrinkage_sigma` are now arguments of `fit_strength_gp`.
- The `if d_in != 10` check in `priors.py` turned out to be dead code (nothing calls it).
- Checked: defaults reproduce upstream exactly; with an informative and a noise column added, the model gives the informative one a short lengthscale (1.4) and the noise one a long one (20.8). Upstream tests for the patched files pass.
- Still to do: hold extra columns fixed at plant-typical values during mix optimization (Results step).

**Cut if behind:** the Models tab keeps the version table and "Use in chat"; "Add new results and retrain" goes.

## Meta models used

| Model | Job |
|---|---|
| `muse-spark-1.3` | Agent loop (Responses API, tool calling) |

Built with Claude Code (not Muse Code).

## Synthetic demo data

`backend/scripts/make_demo_data.py` (seed 7). **24 mix designs, each batched 4–6 times: 119 batches, 833 usable cylinders** at 3, 7 and 28 days (Meta's data stops at 28), plus 33 the cleanup must leave out. Batches of the same recipe differ in measured air, cement lot and noise. "True" strength:
- Meta's pretrained BOxCrete prediction (material source 2, Meta's ready-mix-like concrete);
- × plant material factor 0.95;
- × **slower early strength gain than Meta's lab mixes** (Type IL cement, slag): `1 − (0.20 + 0.25 × slag share) · e^(−age/4 d)`, i.e. −9% to −14% at 3 days, ≈0% at 28 days. Added so that re-tuning Meta's settings has a real difference to fix: without it the synthetic plant is Meta's own model, and tuning found nothing (−1%, noise);
- × −5% per 1% of air above 2%;
- × silica-fume gain, growing with age;
- × cement-lot effect (mill cert);
- × batch noise (3%) × cylinder noise (4%).

Messy on purpose: title rows above headers; water in gal/cy, HRWR in oz/cwt, everything else lb/cy; ages as "7 day"; some air values as "5.8%"; 15 mix codes retyped in the QC log; 6 "cap failure, retest" cylinders; 27 breaks at 56 days; cylinders link to batches by ticket, batches to recipes by mix code. The mess has its own random stream, so the cleaned data is unchanged. Hidden truth in `demo_data/truth.json`. `scripts/check_signal.py` re-runs the checks above. The video says the data is synthetic. AutoML has to find the gains itself, and the video says the data is synthetic.

## Architecture

```
frontend/  Next.js 16 + the mockup's own CSS (AI Elements dropped: its look doesn't match the approved mockup)
backend/   FastAPI (Python 3.12)
  mixpilot/
    agent.py      loop on Meta Responses API (OpenAI SDK, base_url=https://api.meta.ai/v1), streamed text,
                  previous_response_id, undelivered tool results kept for the next turn
    scripted_llm.py  replays the demo without a key (MIXPILOT_LLM=scripted)
    app.py        FastAPI + SSE
    excel.py      read sheets, find header rows, profile columns, guess join keys
    units.py      lb/cy, gal/cy, oz/cwt, °F, MPa -> BOxCrete units
    dataset.py    apply the agent's mapping: recipes / batches / tests -> one table + candidate columns
    validate.py   coverage, ranges vs Meta's training data, cleaned cells, assumptions
    inputs.py     suggest_inputs: within-recipe permutation test + honest refits
    boxfit.py     fit / predict / k-fold refits (parallel worker pool)
    automl.py     BoTorch search and re-check; models.py saved versions; design.py CO2 proposals
    tools.py      Session + the tool functions (train_automl is called by the button, never by the model)
  tests/        pytest: data layer against the generator's truth, agent loop, API guards
  vendor/SustainableConcrete/  vendored BOxCrete with the extra-input patch (uv path dependency)
  demo_data/mapping.json  the demo's correct mapping (scripted LLM, tests, scripts)
  models/       saved model versions
  API: POST /sessions · POST /sessions/{id}/messages · GET /sessions/{id}/events (SSE)
```

## Order of work

1. [x] UI mockup (static HTML): `mockups/ui.html`.
2. [x] Vendor BOxCrete and run a local fit (~13 s); make the input count adjustable.
3. [x] Synthetic Excel generator + signal check.
4. [x] Tools: read → map → validate → suggest inputs. `scripts/run_pipeline.py --suggest` runs them on the demo files (suggest_inputs: 23 s with parallel refits). `scripts/try_llm_mapping.py`: Muse Spark wrote a correct mapping first try (29 s, high reasoning); it skipped the optional `other_cementitious` hint for silica fume, so the step-6 prompt must ask for it.
5. [x] AutoML + leaderboard + model save (`models/<plant>/v<n>/`), results (CO₂ vs strength, 3 proposals ~30% less CO₂, batch conditions like air kept fixed), predict. `scripts/run_automl.py`, `scripts/run_train.py`.
6. [x] Agent loop on Muse Spark + SSE (`mixpilot/agent.py`, `mixpilot/app.py`). Reasoning effort `low` (tool call in ~2 s vs 29 s at the default `high`). Only the Train model button (`POST /approve`) starts training; the model has no such tool and is told when it finishes. Finished AutoML runs are cached by (data, inputs, budget) and a hit is reported as such. Measured: first turn (read → map → validate → suggest → summary) ~55–65 s; training turn ~90 s fresh, ~10 s cached.
   - Run: `cd backend && PYTHONPATH=. uv run uvicorn mixpilot.app:app --port 8811` (port 8765 is taken on this Mac). Terminal version without the server: `PYTHONPATH=. uv run python scripts/run_agent.py --train`.
   - Without a key: add `MIXPILOT_LLM=scripted`. Tests: `cd backend && uv run pytest`.
   - SSE event types: `text_delta` (stream, text), `message` (role, phase commentary|final, stream), `tool_start` (label), `progress`, `tool_end`, `artifact` (kind data|training|results|models), `error`, `done`.
7. [x] Frontend: chat + 3 artifacts + Models tab (`frontend/components/{App,Chat,Panels}.tsx`, `lib/useSession.ts` follows the SSE stream and replays it after a reload). Run: `cd frontend && npx next dev -p 3311` (expects the backend on :8811; override with `NEXT_PUBLIC_API`).
   - Demo prep: the AutoML cache is keyed by the mapped data, and the agent's mapping can differ slightly between sessions (e.g. whether silica fume is listed as cementitious), so do one warm-up run before recording.
8. [ ] Demo video (2–3 min) + README + submission.

## Open questions

- [ ] Official rules: video length, whether a public repo is required.
- [x] Real plant data: the demo stays synthetic.
