# MixPilot: plan (v0.7, 2026-11-04)

Working name. An agent built on Meta Muse Spark that takes a concrete plant's Excel files, gets the data ready with the user, then retrains and tunes Meta's BOxCrete model on that plant's data. The trained model stays saved so the plant can keep using it.

This version runs Michal's engine under Piotr's app. From Michal's `alt-full-stack` and `mixpilot-combined`: batch-level data, testing of every extra column, BoTorch AutoML with re-checks, proposals that name the recipe they replace, the demo data generator. From Piotr's `mixpilot-backend`: the page, one-process server, session event log, agent loop, the "Use it in training" button. Michal's Next.js frontend is not used.

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
   - **Feature discovery is generic, not hard-coded.** Our code tests every extra column the files have, and nothing in the agent's prompt names the demo's columns. Useful full columns are added automatically; the agent says so in one message. Two kinds of column, two tests:
     - **Batch measurements** (air, mill cert, slump, truck…): does the column explain the scatter between batches *of the same recipe*? That scatter is exactly what BOxCrete's 10 inputs can't see. Permutation test within each design (4,000 shuffles, no model fit). Columns that pass but only stand in for something else (a cement lot, a date) are not used as inputs.
     - **Recipe ingredients BOxCrete has no slot for** (silica fume): the first step of training is an honest model comparison, refit with vs without the column and vs a shuffled copy; it must cut the 28-day error by at least 3%, or training leaves it out. No model is fitted before the user presses Train.
   - Tried and rejected: ranking by the baseline's errors on one-batch-per-mix data (chance correlations as big as the real effects on 46 mixes), and comparing BOxCrete's lengthscales against shuffled shadow columns (too noisy).
   - **Partly logged columns are the user's call.** If a useful column only covers part of the batches, the agent says what it is, how much of the data has it and since when, and asks. The card has a "Use it in training" button, and agreeing in chat works too. Gaps are filled with the plant's median.
   - On the synthetic demo (measured 2026-11-04):

     | Column | Coverage | Measured | Verdict |
     |---|---|---|---|
     | Air (%) | 100% | explains 42% of the scatter within recipes, p < 0.001 | added |
     | Silica fume (lb/cy) | 100% | tested in training, with air: 399 → 383 psi (−4%) | added |
     | Mill cert 28d (psi) | 69%, since 2026-06-12 | explains 69% of the scatter air doesn't, p < 0.001 | offered, the user decides |
     | Cement lot, date | 100% | p < 0.001, but they stand in for the cement lot | not inputs |
     | Operator, truck, batch time, slump, conc temp | 100% | no clear link (p ≥ 0.01) | rejected |
   - Mill certs come **per cement lot**. The plant started logging them later, so the 4 earliest of 13 lots have none.
   - Cleanup is automatic and shown in the Prepare view: IDs match ignoring case and punctuation, cylinders a notes column marks as invalid are excluded (the agent decides from the note values), and breaks after 28 days are dropped. `map_columns` returns `notes_still_counted`, so the agent reads what's left and can call it again: without that, it skipped the exclusions on 2 of 3 runs. A mapping that leaves no usable tests fails with an error. On the demo, Muse Spark excludes "cap failure" and also "chipped edge" (6 each), which the generator treats as valid: 841 of 880 cylinders used on 3 of 3 runs, against 847 with the hand-written mapping. Harmless, and defensible.
3. **Train (button):** an AutoML run on the plant's data only (Meta's 670 rows are not pooled). The **baseline is Meta's default setup retrained on the plant's data**; we always retrain, so Meta's untrained model is not shown. One button, fixed budget, no user knobs. Leaderboard rows:
   - Meta default, retrained (baseline);
   - plus tuned settings;
   - plus the approved new inputs;
   - plus both (the expected winner).

   Headline metric: **28-day error in psi**, 5-fold refit holding out whole mix designs. Measured on the demo data: −61% with mill certs, −49% without (table below).
4. **Results:** what the model gives the plant, as a CO₂-vs-strength chart and the next mixes to test. Each proposal names the mix it replaces and keeps that mix's air and mill cert.
5. **Models tab:** the trained model is saved as a version (date, error, inputs used). The user can keep asking about it in chat ("what will this mix reach at 7 days?").

**Rules:**
- The agent proposes; the human approves.
- The LLM only quotes numbers our code computed.

## UI principle

As simple as possible. **Everything on screen is either something we add on top of BOxCrete, or something that has to be there.** Nothing else. That means:
- a timeline across the top with five phases (Upload, Prepare, Review, Train, Results), each with its own full view, plus Models; the chat floats over the views and stays visible in every phase;
- the new inputs and the partly logged input in the Review view; every other column and why it isn't used as one small table there;
- short agent messages (1–3 sentences; cards and panels carry the numbers), with improvements as percentages, not "drops from X to Y";
- **extremely clean**: each panel shows a headline and one chart or table.

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

**Search:** our own AutoML loop, using **BoTorch** (Meta's Bayesian optimization library, which BOxCrete is built on) as the search engine: a GP over the knobs plus LogEI. For each input set: Meta's default, 3 Sobol points, 10 LogEI trials, each scored by 5-fold refit. The page shows real progress (15 rounds, including the re-check). A finished run is cached by (data, inputs, budget) under `backend/models/_cache/`, so a repeat can't stall; delete that folder to show the real progress bar.

**Score:** average over test ages of RMSE ÷ mean strength (early ages matter: 7-day breaks forecast 28-day). Headline on screen: 28-day RMSE in psi.

**Measured on the demo data (2026-11-04), with mill certs (browser run on Muse Spark's mapping, 175 s):**

| Setup | All ages | 28 days | Folds won vs baseline | vs Meta's settings |
|---|---|---|---|---|
| Meta's default setup, retrained | 17.1% | 773 psi | – | – |
| + tuned settings | 14.9% | 718 psi | 7/10 | 7/10 |
| **+ air + silica fume + mill cert** (winner) | **6.8%** | **299 psi** | 10/10 | – |
| + tuned settings + air + silica fume + mill cert | 6.6% | 293 psi | 10/10 | 4/10 |

**Without mill certs (hand-written mapping, 249 s on a busy machine):** Meta's default 771 psi, + tuned settings 717, **+ air + silica fume 392 (winner, −49%)**, + tuned settings + both 395.

Tuning helps Meta's 10 inputs (−7% at 28 days on 7 of 10 folds), but doesn't beat Meta's settings once the new inputs are in, so the saved model keeps Meta's settings and both the panel and the agent say so. The inputs do the work. The 299 psi is about 5% of a typical mix's strength, close to the noise we put in the data (batch 3%, cylinder 4%), so these gains show what the method finds on data with strong known effects. They are not a claim about real plants.

**Scoring (guard against luck):**
- Every score is a **5-fold refit holding out whole mix designs**. BOxCrete's closed-form leave-one-mix-out score is not used: BOxCrete's training minimizes that same score, so on plant-sized data it rewards any extra column, even pure noise (measured: a no-effect column "improved" it 377 → 278 psi).
- After the search: Meta's settings and the best tuned settings of each input set are re-checked on 2 new fold splits. The leaderboard shows these numbers.
- A setup only wins if it beats the baseline on most folds, and tuned settings must also beat Meta's settings on most folds.

### Adding new input columns (any number)

The output doesn't change, so the model still predicts strength. The patch accepts **any number of extra inputs**, not just air. **Done** (2026-11-03), see `backend/vendor/SustainableConcrete/VENDORED.md`:
- Extra columns go **after** Time, so every existing index (`features.IDX`, the binder/aggregate prior groups, the source dim) stays valid. The kernels already treat dims beyond the first 10 as extras with free lengthscales.
- Fixed: the input transform assumed Time is the last column; it now finds it by name.
- Fixed: leave-one-mix-out blocks now include the extra columns.
- `gate_tau` and `shrinkage_sigma` are now arguments of `fit_strength_gp`.
- The `if d_in != 10` check in `priors.py` turned out to be dead code (nothing calls it).
- Checked: defaults reproduce upstream exactly; with an informative and a noise column added, the model gives the informative one a short lengthscale (1.4) and the noise one a long one (20.8). Upstream tests for the patched files pass.
- Extra columns stay fixed during mix optimization: each proposal keeps the air and mill cert of the mix it replaces.

The Models tab has the version table and "Use in chat". Adding new results and retraining from there was cut.

## Meta models used

| Model | Job |
|---|---|
| `muse-spark-1.3` | Agent loop (Responses API, tool calling) |

## Synthetic demo data

`backend/scripts/make_demo_data.py` (seed 7). **24 mix designs, each batched 4–6 times: 121 batches, 847 usable cylinders** at 3, 7 and 28 days (Meta's data stops at 28), plus 33 the cleanup must leave out. Batches of the same recipe differ in measured air, cement lot and noise. "True" strength:
- Meta's pretrained BOxCrete prediction (material source 2, Meta's ready-mix-like concrete);
- × plant material factor 0.95;
- × **slower early strength gain than Meta's lab mixes** (Type IL cement, slag): `1 − (0.20 + 0.25 × slag share) · e^(−age/4 d)`, i.e. −9% to −14% at 3 days, ≈0% at 28 days. Added so that re-tuning Meta's settings has a real difference to fix;
- × −3.5% per 1% of air above 2%;
- × silica-fume gain, growing with age;
- × cement-lot effect (mill cert);
- × batch noise (3%) × cylinder noise (4%).

Messy on purpose: title rows above headers; water in gal/cy, HRWR in oz/cwt, everything else lb/cy; ages as "7 day"; some air values as "5.8%"; 15 mix codes retyped in the QC log; 6 "cap failure, retest" cylinders; harmless notes such as "chipped edge"; 27 breaks at 56 days; cylinders link to batches by ticket, batches to recipes by mix code; mill certs missing for the 4 earliest lots (CL-26001 to CL-26004). The mess has its own random stream, so the cleaned data is unchanged. Hidden truth in `demo_data/truth.json`, which the app never reads. `scripts/check_signal.py` re-runs the signal checks. AutoML has to find the gains itself. The README says the data is synthetic. Truck labels have their own random stream: on the first deal they happened to pass the test by chance (300 random label columns gave 0.33% below p = 0.01, so the test itself is sound), which a plant's engineer would find odd, so they were dealt again. Nothing else changed.

`backend/scripts/make_second_plant.py` (seed 23) makes a second plant for testing, not for the video: other mixes and strengths, other column names, metric units, reworded notes, other file names (`demo_data/plant_metric/`).

## Other plants' layouts

The agent's mapping also covers: a workbook with several sheets; CSV exports (comma, semicolon or tab, decimal commas, title lines, latin-1); cylinders that name only the mix (no QC log); one column per age (`tests.age_columns`); one table with everything, with or without a mix ID; repeated mix rows (the last one counts); blank fly ash, slag, HRWR or other binder cells (read as 0); blank cement, water or aggregate (the mix is left out, and the app says so). `.xls` works too. Fewer than 5 mixes, or no breaks from 21 to 28 days, stops with a plain message, because the model can't be scored on mixes it never saw. Files that aren't concrete data get a plain answer, not a made-up mapping. Results and targets next to real inputs (a 28-day result in the QC log, a specified strength) go in `ignore`, by the agent's judgement.

## Architecture

```
backend/   FastAPI (Python 3.12), one process serves the API and the page
  mixpilot/
    server.py     API + SSE + the static page; warms up the worker pool on start
    session.py    a session and its event log (the page follows it and replays it after a reload)
    agent.py      loop on Meta Responses API (OpenAI SDK, base_url=https://api.meta.ai/v1), streamed text,
                  previous_response_id, undelivered tool results kept for the next turn
    tools.py      the agent's tools, plus train() for the button (the model has no train tool)
    excel.py      read sheets, find header rows, profile columns, guess join keys
    units.py      lb/cy, gal/cy, oz/cwt, °F, MPa -> BOxCrete units
    dataset.py    apply the agent's mapping: recipes / batches / tests -> one table + candidate columns
    validate.py   coverage, ranges vs Meta's training data, cleaned cells, assumptions
    inputs.py     suggest_inputs: within-recipe permutation test; check_ingredients: honest refits at the start of training
    boxfit.py     fit / predict / k-fold refits (parallel worker pool)
    automl.py     BoTorch search and re-check; models.py saved versions; design.py CO2 proposals
    static/index.html  the page (one file, no build step)
  tests/        pytest: data layer against the generator's truth, agent loop (offline stand-in for Muse Spark, tests only), API guards
  vendor/SustainableConcrete/  vendored BOxCrete with the extra-input patch (uv path dependency)
  demo_data/    the plant's Excel files, truth.json, mapping.json (hand-written mapping for tests and scripts)
  models/       saved model versions and the run cache (gitignored)
  API: POST /api/sessions · POST /api/sessions/{id}/files · POST /api/sessions/{id}/messages · POST /api/sessions/{id}/inputs (use or leave out) · POST /api/sessions/{id}/train · GET /api/sessions/{id}/events (SSE) · GET /api/models
```

## Order of work

1. [x] UI mockup (static HTML): `mockups/ui.html`.
2. [x] Vendor BOxCrete and run a local fit (~13 s); make the input count adjustable.
3. [x] Synthetic Excel generator + signal check.
4. [x] Tools: read → map → validate → suggest inputs. `scripts/run_pipeline.py` runs them on the demo files.
5. [x] AutoML + leaderboard + model save (`models/<plant>/v<n>/`), results (CO₂ vs strength, 3 proposals ~30% less CO₂, batch conditions like air kept fixed), predict.
6. [x] Agent loop on Muse Spark + SSE (`mixpilot/agent.py`, `mixpilot/server.py`). Reasoning effort `low`. Only the Train model button starts training; the model has no such tool and is told when it finishes. Measured: first turn ~80 s, training 3–4 min fresh, seconds when cached.
   - Run: `cd backend && PYTHONPATH=. uv run uvicorn mixpilot.server:app --port 8000`. Tests: `cd backend && uv run pytest`.
7. [x] Frontend: phase timeline + floating chat + one view per phase + Models in one static page (`mixpilot/static/index.html`), following the session's SSE stream and replaying it after a reload.
   - Demo prep: the AutoML cache is keyed by the mapped data, and the agent's mapping can differ slightly between sessions, so do one warm-up run before recording.
8. [x] Merge Michal's engine under this app, and move the partly logged story to mill certs (branch `mixpilot-merged`).
9. [ ] Demo video (2–3 min) + submission.

## Open questions

- [ ] Official rules: video length, whether a public repo is required.
- [x] Real plant data: the demo stays synthetic.
