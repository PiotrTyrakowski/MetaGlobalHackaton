# MixPilot: plan (v0.4, 2026-11-02)

Working name. An agent built on Meta Muse Spark that takes a concrete plant's Excel files, gets the data ready with the user, then retrains and tunes Meta's BOxCrete model on that plant's data. The trained model stays saved so the plant can keep using it.

Background research (BOxCrete, Quadrel, Meta API): https://claude.ai/artifact/JUbkNNRZgWKYZgGmLEdSvJ

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
   - It finds **one new useful input**: **air content (%)**, recorded for 100% of mixes. BOxCrete doesn't have it. Each 1% of air costs roughly 5% of strength, so it explains part of the scatter. The agent says: "we can use this."
   - It finds **one useful input that is incomplete**: **cement strength from the mill certificate (MPa)**, recorded for only 40% of mixes. It varies by cement delivery. The agent says: "if you recorded this for every batch, the model would get better," with an estimate measured on the 40% that has it.
3. **Train (button):** an AutoML run on the plant's data. The **baseline is Meta's default setup retrained on the plant's data**; we always retrain, so Meta's untrained model is not shown. Leaderboard rows:
   - Meta default, retrained (baseline);
   - plus tuned hyperparameters (`GATE_TAU`, prior σ, `mll_weight`);
   - plus air content (the expected winner).

   Every number comes from real fits scored by leave-one-mix-out error.
4. **Results:** what the model gives the plant, as a CO₂-vs-strength chart and the next mixes to test.
5. **Models tab:** the trained model is saved as a version (date, error, inputs used). The user can keep asking about it in chat ("what will this mix reach at 7 days?") or add new results and retrain.

**Rules:**
- The agent proposes; the human approves.
- The LLM only quotes numbers our code computed.

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

**Fallback:** before training, convert strengths to a reference air content (≈5% per 1% air), and leave the model untouched.

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

Cement strength, scored only on the mixes that record it (~20 per plant):

| Setup | Plant A | Plant B |
|---|---|---|
| air | 661 | 607 |
| air + cement strength | **406** | **460** |

**Takeaways:**
- The extra input column works end to end.
- Air content is the large, robust gain.
- Tuning helps a little and inconsistently; the best settings differ by plant, which is the reason to search per plant. In the demo, present it as "tuning: small gain; your data: big gain".
- The effect sizes come from our generator (≈5% per 1% air is a real rule of thumb), so the video must say the data is synthetic.

### Pipeline on the demo Excel files (2026-11-02)

`backend/scripts/demo_pipeline.py` runs everything except the LLM in about 3 minutes: read Excel, map, build dataset, checks, AutoML, save model, propose mixes.

**Data preparation:**
- 436 cylinders read. 4 "cap failure, retest" rows excluded, 18 breaks at 56 days dropped (BOxCrete covers 0–28 days).
- 46 mixes × ages 3, 7, 28 days = 138 rows. The round-trip test (`tests/test_roundtrip.py`) checks that the dataset rebuilt from the Excel files matches the generator's truth.

**AutoML (28-day error, psi, 5-fold CV by mix):**

| Setup | Error |
|---|---|
| Meta default, retrained | 750 |
| + tuned settings (τ 0.10, mll 0.3) | 596 |
| + air content | 497 |
| + tuned + air (τ 0.20, mll 0.8) | **448** (−40%) |

Cement strength, scored on the 20 mixes that record it: 635 → **418** (−34%).

Saved model v1 predicts R-101 at 28 days as 5,084 ± 63 psi (measured 5,097).

**Mix proposals:** we do not use qNEHVI. Instead:
- screen 4,000 random mixes inside the plant's own ingredient ranges (SCM ≤ 50%), with aggregates, temperature and air fixed;
- keep the CO₂-vs-predicted-strength Pareto front;
- pick one mix per strength class (25/50/75% of the plant's 28-day strengths) whose prediction minus one SD still reaches the class;
- compare it only with existing mixes of the same air class, and drop it if it saves no CO₂.

At 6% air (16 similar mixes) the proposals save 34% and 24% CO₂ against R-131. At 2% air (28 mixes) they save 10–20%.

## Meta models used

| Model | Job |
|---|---|
| `muse-spark-1.3` | Agent loop (Responses API, tool calling) |
| Muse Code | Used to build the project |

## Synthetic demo data

The generator creates plant mixes. Their "true" strength is:
- Meta's pretrained BOxCrete prediction;
- times a plant material factor;
- minus about 5% per 1% of air;
- plus a cement-strength effect;
- plus measurement noise.

It then writes the three messy Excel files, with air recorded for all mixes and cement strength for 40% of them. AutoML has to find the gains itself, and the video says the data is synthetic.

## Architecture

```
frontend/  Next.js + AI Elements (chat) + our artifact panels
backend/   FastAPI (Python 3.12)
  mixpilot/     ingest (read Excel), prepare (mapping → dataset), checks (coverage, issues,
                candidate inputs), automl (grid, scoring, model save), worker (one subprocess
                per setup), propose (mix screening + Pareto), units, schema
    agent.py    loop on Meta Responses API (OpenAI SDK, base_url=https://api.meta.ai/v1),
                streamed text, previous_response_id; scripted_llm.py replays the demo without a key
    tools.py    inspect_files, prepare_data, add_input, test_input, offer_training,
                propose_mixes, predict_strength, list_models; train (the Train button)
    server.py   FastAPI: POST /api/sessions[/demo] · /files · /messages · /train,
                GET /api/sessions/{id}/events (SSE, replayable) · GET /api/models
  boxcrete/     vendored BOxCrete with the extra-input patch
  workspace/    per-session data and saved model versions (gitignored)
```

## Order of work

1. [x] UI mockup (static HTML): `mockups/ui.html`.
2. [x] Vendor BOxCrete, local fit, air column, AutoML grid (spike).
3. [x] Synthetic Excel generator: `backend/scripts/make_demo_excel.py`.
4. [x] Tools: read → map → validate → suggest inputs.
5. [x] AutoML + leaderboard + model save + mix proposals.
6. [~] Agent loop on Muse Spark + SSE: built and run end to end with the scripted stand-in; not yet run on Muse Spark (needs the API key).
7. [ ] Frontend: chat + 3 artifacts + Models tab.
8. [ ] Demo video (2–3 min) + README + submission.

## Open questions

- [ ] Official rules: video length, whether a public repo is required.
- [ ] Real plant data (the user will send it).
