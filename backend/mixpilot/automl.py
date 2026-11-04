"""AutoML for BOxCrete: re-tune the settings Meta hard-coded, on the plant's data.

Two searches run side by side, one per input set (Meta's 10 inputs, and the 10
plus the approved columns). Each is a small Bayesian optimization with BoTorch
over (gate_tau, shrinkage_sigma, mll_weight); every trial is a 5-fold refit
holding out whole recipes, scored by the average relative error over all test
ages (early-age predictions matter too, e.g. 7-day breaks forecasting 28-day
strength). Meta's default settings are always the first trial.

Then the default and the best tuned settings of both input sets are re-checked on
new fold splits; those numbers make the leaderboard. A setup only wins if it beats
Meta's default on most held-out folds, and tuned settings only count if they also
beat Meta's settings on the same inputs on most folds.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from mixpilot import boxfit
from mixpilot.boxfit import META_DEFAULT, KFoldJob, Settings
from mixpilot.dataset import Dataset

# (name, low, high, log scale)
SPACE = [("gate_tau", 0.03, 0.5, True), ("shrinkage_sigma", 1e-4, 0.3, True), ("mll_weight", 0.0, 1.0, False)]
SEARCH_SEED = 0  # fold split used during the search
RECHECK_SEEDS = (1, 2)  # new fold splits for the leaderboard


@dataclass
class Trial:
    setup: str
    settings: Settings
    score: float  # average relative error over ages
    error_28d: float
    source: str  # "meta default" | "sobol" | "bayes-opt"


@dataclass
class Evaluation:
    score: float  # mean over ages of RMSE / mean strength at that age
    error_28d: float
    errors_by_age: dict[float, float]
    fold_scores: list[float] = field(default_factory=list)


def to_unit(s: Settings) -> list[float]:
    out = []
    for name, lo, hi, log in SPACE:
        v = getattr(s, name)
        out.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if log else (v - lo) / (hi - lo))
    return out


def from_unit(u) -> Settings:
    vals = {}
    for (name, lo, hi, log), x in zip(SPACE, u):
        x = float(min(max(x, 0.0), 1.0))
        v = math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo))) if log else lo + x * (hi - lo)
        vals[name] = float(f"{v:.3g}")
    return Settings(**vals)


def evaluate_many(ds: Dataset, configs: list[tuple[list[str], Settings]], seeds=(SEARCH_SEED,)
                  ) -> list[Evaluation]:
    """5-fold refit errors for (inputs, settings) pairs, pooled over fold seeds."""
    jobs, keys = [], []
    for i, (extras, settings) in enumerate(configs):
        X, Y = ds.tensors(extras)
        for seed in seeds:
            jobs.append(KFoldJob(X, Y, settings, k=5, seed=seed, groups=ds.groups))
            keys.append(i)
    results = boxfit.kfold_many(jobs)
    ages = ds.rows["age"].to_numpy()
    strength = ds.rows["strength"].to_numpy()
    a28 = 28.0 if (ages == 28).any() else float(ages.max())

    def score(resid: np.ndarray, mask: np.ndarray) -> float:
        per_age = [boxfit.rmse(resid[mask & (ages == a)]) / strength[mask & (ages == a)].mean()
                   for a in np.unique(ages) if (mask & (ages == a)).any()]
        return float(np.mean(per_age))

    out = []
    for i in range(len(configs)):
        folds = [(r, f) for (r, f), k in zip(results, keys) if k == i]
        resid = np.concatenate([r for r, _ in folds])
        if not np.isfinite(resid).all():
            out.append(Evaluation(math.inf, math.inf, {}, []))
            continue
        age_all = np.tile(ages, len(seeds))
        by_age = {float(a): boxfit.rmse(resid[age_all == a]) for a in np.unique(ages)}
        overall = float(np.mean([score(r, np.ones_like(ages, dtype=bool)) for r, _ in folds]))
        fold_scores = [score(r, f == j) for r, f in folds for j in range(5)]
        out.append(Evaluation(overall, by_age[a28], by_age, fold_scores))
    return out


def n_rounds(n_init: int = 3, n_bayes: int = 10) -> int:
    """Rounds of a run: Meta's settings, the search, then the re-check."""
    return 1 + n_init + n_bayes + 1


def run_automl(ds: Dataset, inputs: list[str], n_init: int = 3, n_bayes: int = 10,
               progress: Callable[[str], None] = lambda _: None, seed: int = 0,
               on_round: Callable[[int, float], None] = lambda done, best_28d: None) -> dict:
    """on_round gets the rounds finished and the 28-day error of the best trial so far."""
    t_start = time.perf_counter()
    setups = {"meta10": [], "plus": list(inputs)} if inputs else {"meta10": []}
    trials: list[Trial] = []
    sobol = torch.quasirandom.SobolEngine(len(SPACE), scramble=True, seed=seed)
    init_points = [from_unit(u) for u in sobol.draw(n_init).tolist()]
    rounds_done = 0

    def run_round(proposals: dict[str, tuple[Settings, str]], label: str) -> None:
        nonlocal rounds_done
        keys = list(proposals)
        evals = evaluate_many(ds, [(setups[k], proposals[k][0]) for k in keys])
        for k, e in zip(keys, evals):
            trials.append(Trial(k, proposals[k][0], e.score, e.error_28d, proposals[k][1]))
        best = {k: min(t.score for t in trials if t.setup == k) for k in keys}
        progress(f"{label}: " + ", ".join(f"{_setup_name(k, inputs)} best {best[k]:.1%} error" for k in keys))
        rounds_done += 1
        on_round(rounds_done, min(trials, key=lambda t: t.score).error_28d)

    run_round({k: (META_DEFAULT, "meta default") for k in setups}, "trial 1 (Meta's settings)")
    for i, s in enumerate(init_points):
        run_round({k: (s, "sobol") for k in setups}, f"trial {i + 2}")
    for i in range(n_bayes):
        run_round({k: (_propose(trials, k, seed + i), "bayes-opt") for k in setups},
                  f"trial {n_init + 2 + i}")

    # re-check Meta's default and the best tuned settings of each input set on new splits
    best_settings = {k: min((t for t in trials if t.setup == k), key=lambda t: t.score).settings
                     for k in setups}
    rows_cfg = [("meta10", META_DEFAULT, False), ("meta10", best_settings["meta10"], True)]
    if "plus" in setups:
        rows_cfg += [("plus", META_DEFAULT, False), ("plus", best_settings["plus"], True)]
    progress("re-checking the best setups on new fold splits")
    evals = evaluate_many(ds, [(setups[k], s) for k, s, _ in rows_cfg], seeds=RECHECK_SEEDS)
    on_round(rounds_done + 1, min(trials, key=lambda t: t.score).error_28d)

    base = evals[0]
    untuned = {k: e for (k, _, tuned), e in zip(rows_cfg, evals) if not tuned}
    rows = []
    for (k, s, tuned), e in zip(rows_cfg, evals):
        won = sum(a < b for a, b in zip(e.fold_scores, base.fold_scores))
        won_vs_untuned = sum(a < b for a, b in zip(e.fold_scores, untuned[k].fold_scores))
        rows.append(dict(
            name=_row_name(k, tuned, inputs), inputs=setups[k], settings=s.as_dict(), tuned=tuned,
            error_28d_psi=round(e.error_28d), errors_by_age_psi={_age(a): round(v) for a, v in e.errors_by_age.items()},
            error_all_ages=round(e.score, 4), error_all_ages_pct=round(100 * e.score, 1),
            folds_won_vs_baseline=f"{won}/{len(base.fold_scores)}", baseline=(k == "meta10" and not tuned),
            vs_baseline=round(e.score / base.score - 1, 3),
            vs_baseline_28d=round(e.error_28d / base.error_28d - 1, 3),
            **({"folds_won_vs_meta_settings": f"{won_vs_untuned}/{len(e.fold_scores)}"} if tuned else {}),
        ))

    def most(frac: str) -> bool:
        won, n = map(int, frac.split("/"))
        return won > n / 2

    eligible = [r for r in rows[1:] if most(r["folds_won_vs_baseline"])
                and r["error_all_ages"] < rows[0]["error_all_ages"]
                and (not r["tuned"] or most(r["folds_won_vs_meta_settings"]))]
    winner = min(eligible, key=lambda r: r["error_all_ages"]) if eligible else rows[0]
    tuning_helped = winner["tuned"]  # about the model we keep, not any row on the way
    for r in rows:
        r["best"] = r is winner

    return dict(
        leaderboard=rows,
        winner=dict(name=winner["name"], inputs=winner["inputs"], settings=winner["settings"],
                    error_28d_psi=winner["error_28d_psi"], error_all_ages=winner["error_all_ages"],
                    error_all_ages_pct=winner["error_all_ages_pct"],
                    vs_baseline=winner["vs_baseline"], vs_baseline_28d=winner["vs_baseline_28d"]),
        meta_default=META_DEFAULT.as_dict(),
        tuning_helped=tuning_helped,
        trials=[dict(setup=_setup_name(t.setup, inputs), source=t.source, **t.settings.as_dict(),
                     error_all_ages=round(t.score, 4) if math.isfinite(t.score) else None,
                     error_28d_psi=round(t.error_28d) if math.isfinite(t.error_28d) else None) for t in trials],
        method=dict(search=f"BoTorch: {n_init} Sobol + {n_bayes} LogEI trials per input set, "
                           f"Meta's default first",
                    score="average over test ages of RMSE / mean strength, 5-fold refit holding out whole recipes; "
                          "headline: 28-day RMSE in psi",
                    recheck=f"{len(RECHECK_SEEDS)} new fold splits; wins only if better on most folds",
                    space={n: [lo, hi, "log" if log else "linear"] for n, lo, hi, log in SPACE}),
        seconds=round(time.perf_counter() - t_start),
    )


def _propose(trials: list[Trial], setup: str, seed: int) -> Settings:
    """Next settings to try: LogEI on a GP over the trials so far (-log error)."""
    ts = [t for t in trials if t.setup == setup]
    X = torch.tensor([to_unit(t.settings) for t in ts], dtype=torch.float64)
    finite = [t.score for t in ts if math.isfinite(t.score)]
    worst = max(finite) * 1.5 if finite else 1.0
    y = torch.tensor([[-math.log(t.score if math.isfinite(t.score) else worst)] for t in ts],
                     dtype=torch.float64)
    y = (y - y.mean()) / (y.std() if len(ts) > 1 and y.std() > 0 else 1.0)
    gp = SingleTaskGP(X, y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    acq = qLogExpectedImprovement(gp, best_f=y.max())
    torch.manual_seed(seed)
    bounds = torch.stack([torch.zeros(len(SPACE)), torch.ones(len(SPACE))]).to(torch.float64)
    cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=8, raw_samples=256)
    return from_unit(cand[0].tolist())


def _setup_name(k: str, inputs: list[str]) -> str:
    return "Meta's 10 inputs" if k == "meta10" else "+ " + " + ".join(_short(c) for c in inputs)


def _row_name(k: str, tuned: bool, inputs: list[str]) -> str:
    if k == "meta10":
        return "+ tuned settings" if tuned else "Meta's default setup, retrained"
    added = " + ".join(_short(c) for c in inputs)
    return f"+ tuned settings + {added}" if tuned else f"+ {added}"


def _short(c: str) -> str:
    return c.split(" (")[0].strip().lower()


def _age(a: float) -> str:
    return f"{int(a)}d" if float(a).is_integer() else f"{a}d"
