# MixPilot

**An agent on Meta Muse Spark that turns a concrete plant's messy Excel files into the plant's own version of Meta's BOxCrete model, and finds which of the things the plant records make that model more accurate.**

Built for the Meta Global AI Developer Hackathon (Indie track).

## The problem

Concrete causes about 6–8% of the world's CO₂, mostly because of its cement, so using less cement matters. Testing a new mix takes time: you pour a cylinder, wait 28 days and break it to learn how strong the concrete is.

[BOxCrete](https://github.com/facebookresearch/SustainableConcrete), from Meta, predicts that strength and proposes lower-CO₂ mixes. Two things stand between it and a typical plant:

1. **It needs clean data.** It expects one rigid table in fixed units. A plant has several spreadsheets in its own layout and units, with notes like "cap failure" in the margins.
2. **It only sees 10 inputs.** Plants also record things like air content, silica fume and the cement supplier's test results. Batches of the same recipe can differ in strength, and with only the recipe as input, BOxCrete has no way to learn why.

## What MixPilot does

| Step | Who does it | What happens |
|---|---|---|
| **Read** | Muse Spark | Reads the sheets, works out which column is what and in which units, and reads notes columns to find invalid tests |
| **Join and convert** | Our code | Converts units and joins recipes, batches and cylinder breaks into one table |
| **Find new inputs** | Our code | Checks every other column against the plant's own data, without fitting a model. Useful ones go in, new ingredients wait for their test in training, and the rest stay out with a reason on screen |
| **Ask** | Muse Spark and the user | If a useful column was only logged on part of the data, the agent says so and asks whether to use it |
| **Train** | Our code, on the user's click | Tests each new ingredient and leaves out any that don't help, then retrains BOxCrete on the plant's data, with and without the new inputs and re-tuned settings, and keeps what proves better |
| **Use** | Both | Saves the model, proposes lower-CO₂ mixes to test next, and answers questions in chat such as "what does this mix reach at 7 days with 7% air?" |

## Why this is new

- **The model gets new inputs as well as new data.** Retraining on a plant's data is the usual idea. MixPilot also changes what the model can see. We patched BOxCrete to take any number of extra inputs, and with no extra inputs it gives the same result as Meta's original code.
- **A new input only goes in if a test shows it helps.** The LLM's opinion doesn't decide, and neither does a score that can be fooled. Each column has to pass a statistical test on the plant's own data (see below).
- **The LLM and the statistics each do what they're good at.** Reading a messy sheet and understanding "cap failure, retest" takes judgement, so the LLM does that. Every number comes from code. The agent is told to quote only what a tool returned, and it has no tool that starts training.
- **Bayesian optimization tunes a Bayesian model.** BOxCrete's authors fixed some of its settings for their own data. Our AutoML pipeline uses BoTorch to re-tune them for each plant.
- **It reports when something didn't help.** A setup only wins if it beats the baseline on most held-out folds, on splits the search never used. On the demo plant, re-tuning does not improve the final model, and the app says so.
- **Partly logged data is the plant's decision.** The app notices when a column only exists since the plant started logging it, measures its value on the batches that have it, and lets the plant choose.

## The model

**BOxCrete** is a Gaussian process built on BoTorch. It gives a strength curve from day 0 to day 28, with an uncertainty for every prediction, and learns by comparing similar mixes.

- **Its 10 inputs:** cement, fly ash, slag, water, HRWR (superplasticizer), fine aggregate, coarse aggregate, material source, curing temperature and age.
- **7 features it derives from those:** water-to-binder ratio, share of fly ash and slag, HRWR-to-binder ratio, water-to-cement ratio, coarse-to-fine ratio, aggregate-to-paste ratio and concrete maturity.
- **Its training:** it fits by predicting each mix with that mix left out, combined with the usual likelihood.

**What we changed** (details in [VENDORED.md](backend/vendor/SustainableConcrete/VENDORED.md)):

- **Extra inputs after age.** Each one gets its own learned lengthscale, which is how a Gaussian process learns how much an input matters. In a check with one informative and one noise column, the model gave them lengthscales of 1.4 and 20.8, so it relied on the first and largely ignored the second.
- **Tunable settings.** Three fixed settings can now be searched: the time-gate timescale, the lengthscale prior and the mix of the two training objectives.

**What went back to BOxCrete:**

- **A small fix, accepted.** While working in BOxCrete's code, we found that its default limits on total mass and binder were in lb/yd³, while its data is in kg/m³. As a result, its default search space ruled out all 80 concrete mixes in its own dataset. Michal opened the [pull request](https://github.com/facebookresearch/SustainableConcrete/pull/60), which converts the limits and lets 42 of the 80 through, and it was merged into BOxCrete on November 4, 2026. MixPilot doesn't use those limits, so its results don't depend on this fix.
- **Extra inputs, waiting for review.** We also [proposed](https://github.com/facebookresearch/SustainableConcrete/pull/61) the extra-inputs change above for BOxCrete itself. It hasn't been accepted yet.

**How a column gets in:**

- **Batch measurements** such as air content: does the column explain the strength differences between batches *of the same recipe*? BOxCrete's inputs can't see those differences, because the recipe is identical. This is a permutation test within each recipe, with 4,000 shuffles and a threshold of p < 0.01.
- **Ingredients BOxCrete has no input for** such as silica fume: the data alone can't say whether they help, so this test is the first step of training. It refits the model with the column, without it and with a shuffled copy, holding out whole mixes. The column must cut the 28-day error by at least 3% and beat its shuffled copy by at least 3%, or it's left out.
- **Labels and dates** can pass the test but can't go in the model. The app reports them as standing in for something else, such as a cement lot standing in for its certificate.
- **A column that repeats a strength result** is caught and rejected, because predicting strength from strength proves nothing.

**How setups are scored:** 5-fold refits that hold out whole mix designs, so every score is on mixes the model never saw. BoTorch runs 1 trial on Meta's settings, 3 Sobol trials and 10 LogEI trials per input set. The best setups are then re-checked on 2 new fold splits, and those numbers make the leaderboard.

## Results on the demo plant

The demo plant is synthetic: 24 mix designs, 121 batches and 880 cylinders, generated from Meta's pretrained model plus known effects and noise. These numbers show the method working on data we built. **They are not a claim about real plants.** Measured on 2026-11-04.

| Column | What the tests found | Result |
|---|---|---|
| Air | Explains 42% of the strength differences between batches of the same mix | Added |
| Silica fume | Tested in training: about 4% lower error on mixes the model never saw | Added |
| Mill certificate (the cement's strength test) | Logged only since June, on 83 of 121 batches; on those batches it explains 69% of the differences air doesn't | The user decides |
| Cement lot, date | Linked to strength, but a label or a date | Not used |
| Operator, truck, batch time, slump, concrete temperature | No clear link | Not used |

| Setup | 28-day prediction error |
|---|---|
| Meta's default setup, retrained on the plant's data | 773 psi |
| + re-tuned settings | 718 psi |
| **+ air + silica fume + mill certificate (winner)** | **299 psi** |
| + re-tuned settings + air + silica fume + mill certificate | 293 psi, but it beat Meta's settings on only 4 of 10 folds, so it doesn't win |

So predictions are **61% more accurate than Meta's default setup retrained on the same data**, and the new inputs did that. If the user leaves out the mill certificate, the winner is 392 psi (49% more accurate).

The model then proposes mixes with about 30% less CO₂ at equivalent predicted strength. Each one names the mix it replaces, for the lab to test next.

## Questions you might have

**Why not use BOxCrete directly?** You can, if your data is already clean and you only want its 10 inputs. MixPilot is for a plant that has neither a clean table nor a data scientist, and it looks for what those 10 inputs miss.

**Why is an LLM needed?** Every plant's files look different. Column names, units, layouts and notes can't be covered by fixed rules, and Muse Spark works them out from the names and values.

**Where exactly is Muse Spark used?** It runs the whole conversation through the Meta Model API with tool calling. It reads the sheet profiles, writes the mapping, fixes it when a tool returns an error, decides which notes mean an invalid test, explains results and answers questions.

**Is the demo scripted?** No. The agent runs live, nothing in its prompt names the demo's columns, and a test checks that. `backend/demo_data/plant_metric/` is a second plant with different data, column names, metric units and file names to try.

**Is a 61% improvement realistic?** Not as a promise. The demo data has strong effects we put there on purpose, and a real plant may gain less, or nothing. In that case the app says no column helped and trains without extra inputs. The noise we added (3% per batch, 4% per cylinder) alone leaves about 220–240 psi that no model can predict, so 299 psi is close to the limit on this data.

**How do you know it isn't overfitting?**

- Every score holds out whole mixes.
- A setup has to win on most folds, on splits the search never used.
- We compare against BOxCrete retrained on the same data, not the untouched pretrained model, which would flatter us.
- We fed the within-recipe test 300 columns of random labels, and 0.33% passed at p < 0.01, in line with chance.
- We dropped the closed-form leave-one-out score after a column with no effect "improved" it from 377 to 278 psi.

**What does psi mean?** Pounds per square inch, the US unit of concrete strength. 1 MPa is 145 psi, and a typical demo mix reaches about 5,500 psi at 28 days. "299 psi error" means the 28-day predictions are typically off by about that much, around 5% of a typical mix.

**What does the ± mean on a prediction?** The model's 90% uncertainty for that mix's mean strength.

**How are the lower-CO₂ mixes chosen?** Candidates stay close to what the plant already makes: a blend with a similar recipe, with up to 30% of the cement moved to slag or fly ash. Air and other batch conditions stay at the original mix's value. A candidate only counts if its cautious estimate (the 90% lower bound) reaches the original mix's predicted 28-day strength. CO₂ uses BOxCrete's emission factors. The silica fume factor is our assumption.

**Can a plant pour these mixes tomorrow?** No. They are the next mixes to test in the lab, and the plant's QC engineer signs off. Predictions are for lab-cured cylinders.

**What if a useful column has gaps?** The agent asks. If you agree, batches without a value get the plant's median for that column.

**What file layouts work?** Excel (`.xlsx`, `.xls`) and CSV, with title rows above the headers, and:

- one workbook with several sheets, or separate files;
- comma, semicolon or tab separators, and decimal commas;
- with or without a QC log;
- one row per cylinder, or one column per age;
- one table with everything, even with no mix IDs;
- US or metric units.

**What if the data isn't good enough?** It says so plainly. Files that aren't concrete data get an explanation of what's missing. Training needs at least 5 mix designs and at least one break at 21 days or later. A mix with a blank main ingredient is left out, and the Prepare view lists what was left out.

**Where does my data go?** The files stay on the machine running the app. Column names, up to 12 sample values per column and our computed results go to the Meta Model API, so Muse Spark can do its job. The full rows aren't sent.

**How long does it take?** On an Apple M5 laptop, about 1.5 minutes for the agent's first turn and 3 to 8 minutes for training, depending on what else the machine is running.

## Run it

You need [uv](https://docs.astral.sh/uv/), Python 3.12 and a Meta Model API key.

```bash
cd backend
```

```bash
echo "META_API_KEY=your-key" > .env
```

```bash
PYTHONPATH=. uv run uvicorn mixpilot.server:app --port 8000
```

Open http://localhost:8000 and upload the three files in `backend/demo_data/plant/`. Then:

1. Press **Start preparing**, and wait for the agent to move to **Review**.
2. Click **Use it in training** on the mill certificate card, then press **Train model**.
3. When **Results** opens, use the timeline at the top to look back at any phase, and ask something in chat.

A finished training run is cached under `backend/models/_cache/`, so a repeat is fast. Delete that folder to run it again for real.

Tests (no key or network needed; the agent tests use a stand-in for Muse Spark):

```bash
uv run pytest
```

## How it's built

```
backend/mixpilot/
  server.py    FastAPI: one process serves the API and the page
  session.py   a session and its event log (the page follows it and replays it after a reload)
  agent.py     the agent loop on the Meta Responses API (streaming, tool calls)
  tools.py     the tools the agent calls, and training for the button
  excel.py     read sheets, find header rows, profile columns
  units.py     lb/yd³, gal/yd³, fl oz/cwt, °F, MPa and others to BOxCrete's units
  dataset.py   apply the agent's mapping: recipes, batches, tests into one table
  validate.py  coverage and range checks
  inputs.py    test every candidate column
  boxfit.py    fit, predict, k-fold refits in a worker pool
  automl.py    BoTorch search and re-check
  models.py    saved model versions
  design.py    lower-CO₂ proposals
  static/      the page: a phase timeline, one view per phase and a floating chat (one HTML file, no build step)
backend/vendor/SustainableConcrete/   BOxCrete with a patch for extra inputs
backend/scripts/                      the demo data generators and checks
```

More detail and measurements are in [docs/PLAN.md](docs/PLAN.md).

## The demo data

`backend/scripts/make_demo_data.py` generates the plant, messy on purpose: title rows, US units, retyped mix codes and invalid tests. Strength starts from Meta's pretrained BOxCrete and adds known effects: air, silica fume, the cement lot and noise. The first four cement lots have no mill certificate, because the plant started logging them later. The agent and AutoML have to find all of this on their own. The hidden truth is in `backend/demo_data/truth.json`, which the app never reads. `backend/scripts/make_second_plant.py` makes the second plant.

## Limits

- It hasn't been tried on a real plant's data yet.
- An LLM's mapping can differ a little between runs, for example in which notes count as invalid tests. The Prepare view shows what was left out.
- Each install keeps one list of saved models, so a second plant's models sit beside the first's.
- Only silica fume has an emission factor among the extra ingredients. Another extra binder would count as zero CO₂.
- It doesn't read PDFs or photos of lab sheets.

## Meta models and libraries

| What | Used for |
|---|---|
| Muse Spark (`muse-spark-1.3`) through the Meta Model API | the agent: reading files, mapping columns, explaining results, answering questions |
| [BOxCrete](https://github.com/facebookresearch/SustainableConcrete) ([paper](https://arxiv.org/abs/2603.21525)) | the strength model and mix proposals |
| [BoTorch](https://botorch.org/) | the AutoML search |

## Credits

Piotr Tyrakowski and Michal Zajaczkowski. BOxCrete is by Meta (MIT license, see `backend/vendor/SustainableConcrete/LICENSE`).
