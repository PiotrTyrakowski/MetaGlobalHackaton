# MixPilot: plan (v0.6, 2026-11-03)

Working name. An agent built on Meta Muse Spark that takes a concrete plant's Excel files, gets the data ready with the user, then retrains and tunes Meta's BOxCrete model on that plant's data. The trained model stays saved so the plant can keep using it.

## Constraints

- **Hackathon:** Meta Global AI Developer Hackathon, **Indie track**. Judged on creativity, technical ambition and whether it works.
- **Deadline:** submission due **Nov 4 2026, 9:00 PM ET**.
- **Meta Model API:** we'll use it. Claim the $150 credits by the deadline (needs a card; set a spend limit).
- **Compute:** runs locally. The laptop (Apple M5, 10 cores, 16 GB) is enough for BOxCrete.
- **Out of scope:** a PR to upstream BOxCrete, cylinder photos / SAM, PDFs. The demo uses Excel files only.

## Demo pipeline

1. **Upload:** the plant uploads its Excel files. For the demo these are synthetic, each with its own column names and US units:
   - `mix_designs.xlsx`: recipes in lb/yd³;
   - `cylinder_breaks.xlsx`: one row per cylinder;
   - `daily_QC.xlsx`: QC log per batch.
2. **Prepare (back and forth with the agent):**
   - The agent maps the columns to BOxCrete's schema and converts units.
   - It finds **one new useful input**: **air content (%)**, recorded for 100% of mixes. BOxCrete doesn't have it. Each 1% of air costs roughly 3.5% of strength, so it explains part of the scatter. The agent adds it.
   - It finds **one useful input that is incomplete**: **water added on site (gal/yd³)**. The plant only started logging it in August 2025, so it is there for the newest 42 of 70 mixes (60%). Drivers add it to bring the slump back, and each gallon per yd³ costs roughly 6% of strength (about 300 psi). The agent measures the gain on the mixes that have it and asks whether to use it. The user clicks **Use it in training** on the card (or says yes in chat). Mixes without a value get the plant's typical value.
3. **Train (button):** an AutoML run on the plant's data. The **baseline is Meta's default setup retrained on the plant's data**; we always retrain, so Meta's untrained model is not shown. Leaderboard rows:
   - Meta default, retrained (baseline);
   - plus tuned hyperparameters (`GATE_TAU`, `mll_weight`);
   - plus air content;
   - plus water added on site, if the user chose to use it.

   Every number comes from real fits scored by 5-fold cross-validation grouped by mix: each mix is held out once and predicted by a model that never saw it.
4. **Results:** what the model gives the plant, as a CO₂-vs-strength chart and the next mixes to test.
5. **Models tab:** the trained model is saved as a version (date, error, inputs used). The user can keep asking about it in chat ("what will this mix reach at 7 days?") or add new results and retrain.

**Rules:**
- The agent proposes; the human approves.
- The LLM only quotes numbers our code computed.
- Gains are said as percentages ("42% more accurate"); before-and-after errors stay in the panels.

## UI principle

As simple as possible. **Everything on screen is either something we add on top of BOxCrete, or something that has to be there.** Nothing else. That means:
- one new input, one missing input;
- three artifacts (Data, Training, Results) plus the Models tab;
- short agent messages.

## AutoML: what we tune

**Fitted automatically by every BOxCrete fit (not our job):** lengthscales, output scale, noise, correlation between material sources.

**Set by hand in BOxCrete's code, so this is our search space:**

| Knob | Where | Default |
|---|---|---|
| Time-gate timescale | `features.py` `GATE_TAU` | 0.10 |
| Lengthscale shrinkage prior σ | `priors.py` | 0.001 |
| Training loss mix (block-LOO vs likelihood) | `block_loo.train_block_loo(mll_weight=…)` | 0.5 |
| Inputs | `utils.py` `DEFAULT_X_COLUMNS` | 10 columns; we add air content |
| Engineered features | `features.py` `F5_ALLLOG_FEATURES` | 7 log features (kept as is for the demo) |

Guard against luck: Meta's default is always the baseline, and another setup wins only if it is clearly better across most held-out mixes.

### Adding a new input column (air content)

The output doesn't change, so the model still predicts strength. The difficulty is that BOxCrete's internals are written for exactly 10 inputs:
- `kernels.py` reads the input count once at import (`_N_RAW_DIMS = len(DEFAULT_X_COLUMNS)`), and fixes which columns each kernel part sees (`active_dims`).
- `priors.py` turns the lengthscale prior **off without warning** when there aren't exactly 10 inputs (`if d_in != 10: return None`). An 11-input model would quietly become a different, weaker model, and the comparison would be unfair.
- Time must stay the last column.
- Optimization needs bounds for the new column.
- Meta's 670 rows (if pooled) have no air value.

**Plan:**
- Vendor BOxCrete (MIT) into the repo.
- Add `Air (%)` before `Time`.
- Fix the 10-input check in `priors.py`.
- Add bounds.
- Hold air fixed (e.g. 6%) during optimization.

Estimated at a few hours.

**Fallback:** before training, convert strengths to a reference air content (≈3.5% per 1% air), and leave the model untouched.

### Spike results (2026-11-02)

The code is in `backend/spike/`. BOxCrete is vendored in `backend/boxcrete/` and patched as follows:
- `MIXPILOT_EXTRA_X` adds input columns before `Time`;
- `MIXPILOT_GATE_TAU` and `MIXPILOT_PRIOR_SIGMA` override the gate timescale and prior σ;
- the block-LOO grouping follows the input count.

**Speed:**
- The full production fit on Meta's 670 rows takes **29 s** on the laptop.
- Scoring one setup on a 46-mix plant (5-fold CV by mix, refit per fold) takes **~16 s**.
- A 20-setup grid in 5 parallel processes takes **1 min 40 s**.

28-day error (psi) on two independently generated synthetic plants, 46 mixes each:

| Setup | Plant A | Plant B |
|---|---|---|
| Meta default (τ 0.10, mll 0.5) | 710 | 1,255 |
| Tuned, no air (τ 0.10, mll 0.3) | 576 | 1,213 |
| + air content | 488 | 382 |
| + air + tuned (τ 0.20, mll 0.8) | **438** | **377** |

Cement strength (the first choice of incomplete input, since replaced by water added on site), scored only on the mixes that record it (~20 per plant):

| Setup | Plant A | Plant B |
|---|---|---|
| air | 661 | 607 |
| air + cement strength | **406** | **460** |

**Takeaways:**
- The extra input column works end to end.
- Air content is the large, robust gain.
- Tuning helps a little and inconsistently; the best settings differ by plant, which is the reason to search per plant. In the demo, present it as "tuning: small gain; your data: big gain".
- The effect sizes come from our generator (≈5% per 1% air is a real rule of thumb), so the video must say the data is synthetic.

### Pipeline on the demo Excel files (2026-11-03)

`backend/scripts/demo_pipeline.py` runs everything except the LLM in about 3½ minutes: read Excel, map, build dataset, checks, AutoML, save model, propose mixes.

**Data preparation:**
- 652 cylinders read. 4 "cap failure, retest" rows excluded, 18 breaks at 56 days dropped (BOxCrete covers 0–28 days).
- 70 mixes × ages 3, 7, 28 days = 210 rows. The round-trip test (`tests/test_roundtrip.py`) checks that the dataset rebuilt from the Excel files matches the generator's truth.

**Water added on site, before training** (the agent's test, on the 42 mixes that log it): 721 → **421** psi (−42%). The user chooses whether to train with it.

**AutoML with both inputs (27 setups, 3 min; 28-day error in psi, 5-fold CV by mix):**

| Setup | Error |
|---|---|
| Meta's default setup, retrained | 872 |
| + tuned settings (τ 0.20, mll 0.3) | 786 |
| + air content | 589 |
| + water added on site (τ 0.20, mll 0.8) | **520** (−40%) |

Without site water the run has 18 setups and stops at the air row.

The saved model predicts R-101 at 28 days as 5,856 ± 60 psi (measured 5,909; R-101 is in its training data).

**What the numbers mean:** psi (pounds per square inch) is the US unit of concrete strength; 1,000 psi ≈ 6.9 MPa, and a typical mix here reaches about 4,950 psi at 28 days. The error is the model's typical miss (RMSE) at 28 days on mixes it never saw during training. 520 psi is about 10% of a typical mix.

**Mix proposals:** we do not use qNEHVI. Instead:
- screen 4,000 random mixes inside the plant's own ingredient ranges (SCM ≤ 50%), with aggregates, temperature and extra inputs held at the plant's typical values (or a value the user asks for);
- keep the Pareto front of CO₂ vs the strength the model is confident of (prediction minus one SD);
- pick, per strength class (25/50/75% of the plant's 28-day strengths), the lowest-CO₂ mix on that front that still reaches the class;
- when an input is held at a set value, compare only with existing mixes near it (an air-entrained proposal with air-entrained mixes); drop a proposal that saves no CO₂.

At typical values (all 70 mixes compared) the proposals save 9% and 20% CO₂. At 6% air (27 similar mixes) they save 12–14%, at 2% air (38 mixes) 18–20%. The cheapest candidates sit at the screen's limit (least binder, 50% SCM), so a proposal can land well above its class (P-2: 6,717 ± 722 psi for the 4,900 psi class).

## Meta models used

| Model | Job |
|---|---|
| `muse-spark-1.3` | Agent loop (Responses API, tool calling) |
| Muse Code | Used to build the project |

## Synthetic demo data

The generator creates plant mixes. Their "true" strength is:
- Meta's pretrained BOxCrete prediction;
- minus about 3.5% per 1% of air above 2%;
- minus about 6% per gal/yd³ of water added on site (0–3 gal/yd³);
- times an unrecorded batch-to-batch factor (5% scatter: materials, batching);
- plus cylinder-to-cylinder noise (3.5%).

Together these give each mix roughly 400 psi of scatter, which an industry guide (ACI 214) treats as good control.

It then writes the three messy Excel files for 70 mixes cast from January 2025 to July 2026. Air is recorded for all of them. Water added on site is logged only from August 2025 on, so the newest 42 mixes (60%) have it. Site water also explains part of why the plant runs weaker than Meta's model. AutoML has to find the gains itself, and the video says the data is synthetic.

## Architecture

```
frontend/  Next.js + AI Elements (chat) + our artifact panels
backend/   FastAPI (Python 3.12)
  mixpilot/     ingest (read Excel), prepare (mapping → dataset), checks (coverage, issues,
                candidate inputs), automl (grid, scoring, model save), worker (one subprocess
                per setup), propose (mix screening + Pareto), units, schema
    agent.py    loop on Meta Responses API (OpenAI SDK, base_url=https://api.meta.ai/v1),
                streamed text, previous_response_id; the app always runs Muse Spark
    tools.py    inspect_files, prepare_data, add_input, test_input, use_input, propose_mixes,
                predict_strength, list_models; app-side: offer_training (shown after a turn
                that left untrained data) and train (the Train button)
    server.py   FastAPI: POST /api/sessions · /files · /messages · /inputs (the card's
                "Use it" button) · /train, GET /api/sessions/{id}/events (SSE, replayable) ·
                GET /api/models
  tests/        round trip Excel → dataset, proposals, and the agent's prep turn driven by an
                offline stand-in for the model (scripted_llm.py, tests only)
  boxcrete/     vendored BOxCrete with the extra-input patch
  workspace/    per-session data and saved model versions (gitignored)
```

## Order of work

1. [x] UI mockup (static HTML): `mockups/ui.html`.
2. [x] Vendor BOxCrete, local fit, air column, AutoML grid (spike).
3. [x] Synthetic Excel generator: `backend/scripts/make_demo_excel.py`.
4. [x] Tools: read → map → validate → suggest inputs.
5. [x] AutoML + leaderboard + model save + mix proposals.
6. [x] Agent loop on Muse Spark + SSE: full demo run on muse-spark-1.3 (prep 49 s with the right mapping and
   inputs, training, proposals, what-if questions). The Train button is app state, not a model tool;
   text the model writes before tool calls is hidden.
7. [x] Frontend: chat + 3 artifacts + Models tab (single page served by FastAPI).
8. [ ] Demo video (2–3 min) + README + submission.

## Open questions

- [ ] Official rules: video length, whether a public repo is required.
- [ ] Real plant data (the user will send it).
