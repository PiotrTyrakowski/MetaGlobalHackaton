# MixPilot: plan (v0.3, 2026-11-02)

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
  agent.py      loop on Meta Responses API (OpenAI SDK, base_url=https://api.meta.ai/v1)
  tools/        read_excel, map_columns, validate, suggest_inputs, train_automl, results, predict
  boxcrete/     vendored BOxCrete with the extra-input patch
  models/       saved model versions
  API: POST /sessions · POST /sessions/{id}/messages · GET /sessions/{id}/events (SSE)
```

## Order of work

1. [x] UI mockup (static HTML): `mockups/ui.html`.
2. [ ] Vendor BOxCrete and run a local fit (check the ~85 s fit time); then add the air column.
3. [ ] Synthetic Excel generator.
4. [ ] Tools: read → map → validate → suggest inputs.
5. [ ] AutoML + leaderboard + model save.
6. [ ] Agent loop on Muse Spark + SSE.
7. [ ] Frontend: chat + 3 artifacts + Models tab.
8. [ ] Demo video (2–3 min) + README + submission.

## Open questions

- [ ] Official rules: video length, whether a public repo is required.
- [ ] Real plant data (the user will send it).
