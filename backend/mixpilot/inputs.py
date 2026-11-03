"""Which extra columns should BOxCrete use? (the suggest_inputs tool)

Two kinds of candidate, two tests:
- batch measurements (air, mill cert, truck...): do they explain the scatter
  between batches of the same recipe? That scatter is what BOxCrete's 10 inputs
  can't see. Permutation test within each recipe; no model fit.
- recipe ingredients BOxCrete has no slot for (silica fume): refit BOxCrete with
  and without the column, and with a shuffled copy, holding out whole recipes.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from mixpilot import boxfit
from mixpilot.dataset import Dataset

P_THRESHOLD = 0.01
FULL_COVERAGE = 0.95
MIN_GAIN = 0.03  # a recipe column must cut the 28-day error by 3% and beat its shuffled copy
N_PERM = 4000


def suggest_inputs(ds: Dataset, progress: Callable[[str], None] = lambda _: None,
                   seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    per_batch = _batch_deviations(ds)
    results: dict[str, dict] = {}

    # 1. batch measurements, greedily: accept the strongest, condition on it, re-test the rest
    batch_cands = [c for c, k in ds.candidates.items() if k.level == "batch"]
    accepted: list[str] = []
    remaining = list(batch_cands)
    while remaining:
        progress(f"testing {len(remaining)} batch columns against same-recipe scatter")
        dev = _condition(per_batch, ds, accepted)
        tests = {c: _within_recipe_test(ds, per_batch, dev, c, rng) for c in remaining}
        for c, t in tests.items():
            results[c] = t
        best = [c for c in remaining
                if tests[c]["p"] < P_THRESHOLD and tests[c]["coverage"] >= FULL_COVERAGE
                and ds.candidates[c].kind == "numeric"]
        if not best:
            break
        top = min(best, key=lambda c: (tests[c]["p"], -tests[c]["strength"]))
        accepted.append(top)
        remaining.remove(top)
        results[top]["conditioned_on"] = accepted[:-1]

    for c in batch_cands:
        r = results[c]
        k = ds.candidates[c]
        if c in accepted:
            r["verdict"] = "use"
        elif r["p"] >= P_THRESHOLD:
            r["verdict"] = "reject"
        elif k.kind == "numeric" and r["coverage"] < FULL_COVERAGE:
            r["verdict"] = "collect"
        else:
            r["verdict"] = "proxy"  # matters, but a label or date: not a model input

    # 2. what the significant labels stand for
    for c in batch_cands:
        if results[c]["verdict"] == "proxy":
            results[c]["associated_with"] = _associations(ds, c, [x for x in batch_cands if x != c
                                                                   and results[x]["p"] < P_THRESHOLD])

    # 3. recipe ingredients: honest refits, each vs the same base and a shuffled copy
    recipe_cands = [c for c, k in ds.candidates.items() if k.level == "recipe"]
    numeric = []
    for c in recipe_cands:
        k = ds.candidates[c]
        if k.kind == "numeric":
            numeric.append(c)
        else:
            results[c] = dict(column=c, level="recipe", kind=k.kind, coverage=_coverage(ds, c),
                              verdict="not_tested", reason="a label, not a quantity")
    if numeric:
        progress(f"refitting BOxCrete with and without {', '.join(numeric)} (and shuffled copies)")
        jobs = [_job(ds, accepted)]
        for c in numeric:
            jobs += [_job(ds, accepted + [c]), _job(ds, accepted + [c], shuffle=c, rng=rng)]
        errs = _kfold28_many(jobs)
        base_err = errs[0]
        for i, c in enumerate(numeric):
            with_err, shuffled_err = errs[1 + 2 * i], errs[2 + 2 * i]
            gain = 1 - with_err / base_err
            beats_shuffle = 1 - with_err / shuffled_err
            results[c] = dict(
                column=c, level="recipe", kind="numeric", coverage=_coverage(ds, c),
                error_without_psi=round(base_err), error_with_psi=round(with_err),
                error_shuffled_psi=round(shuffled_err), gain=round(gain, 3),
                verdict="use" if gain >= MIN_GAIN and beats_shuffle >= MIN_GAIN else "reject",
            )
        accepted += [c for c in numeric if results[c]["verdict"] == "use"]

    # 4. incomplete but useful: how much would complete data help, and where are the gaps?
    collect = [c for c, r in results.items() if r.get("verdict") == "collect"]
    if collect:
        progress(f"estimating what complete {', '.join(collect)} would be worth")
        jobs = []
        for c in collect:
            sub = ds.rows[ds.rows["batch"].isin(set(ds.feature(c).dropna().index))]
            jobs += [_job(ds, accepted, rows=sub), _job(ds, accepted + [c], rows=sub)]
        errs = _kfold28_many(jobs)
        for i, c in enumerate(collect):
            without, with_c = errs[2 * i], errs[2 * i + 1]
            results[c].update(error_without_psi=round(without), error_with_psi=round(with_c),
                              gain=round(1 - with_c / without, 3),
                              batches_with_value=int(ds.feature(c).notna().sum()),
                              batches_total=int(ds.batches["batch"].nunique()),
                              missing_by=_missing_by(ds, c))

    order = {"use": 0, "collect": 1, "proxy": 2, "reject": 3, "not_tested": 4}
    ranked = sorted(results.values(), key=lambda r: (order[r["verdict"]], r.get("p", 1)))
    return dict(
        use=[r["column"] for r in ranked if r["verdict"] == "use"],
        collect=[r["column"] for r in ranked if r["verdict"] == "collect"],
        columns=ranked,
        method=dict(batch_test=f"within-recipe permutation test, {N_PERM} shuffles, p < {P_THRESHOLD}",
                    recipe_test=f"5-fold refit holding out whole recipes; must cut 28-day error by "
                                f"{MIN_GAIN:.0%} and beat a shuffled copy"),
    )


# --- within-recipe test ------------------------------------------------------------

def _batch_deviations(ds: Dataset) -> pd.DataFrame:
    """Per batch: how much stronger than its recipe's average (log units, mean over ages)."""
    r = ds.rows.copy()
    r["log_s"] = np.log(r["strength"])
    grp = r.groupby(["recipe", "age"])["log_s"]
    r["dev"] = r["log_s"] - grp.transform("mean")
    r = r[grp.transform("size") >= 2]
    return r.groupby("batch").agg(recipe=("recipe", "first"), dev=("dev", "mean"))


def _condition(per_batch: pd.DataFrame, ds: Dataset, accepted: list[str]) -> pd.Series:
    """Deviations with the accepted columns' within-recipe effect regressed out."""
    dev = per_batch["dev"]
    if not accepted:
        return dev
    Z = pd.DataFrame({c: ds.feature(c).reindex(per_batch.index) for c in accepted})
    ok = Z.notna().all(axis=1)
    Zc = Z[ok] - Z[ok].groupby(per_batch.loc[ok, "recipe"]).transform("mean")
    d = dev[ok] - dev[ok].groupby(per_batch.loc[ok, "recipe"]).transform("mean")
    beta, *_ = np.linalg.lstsq(Zc.to_numpy(), d.to_numpy(), rcond=None)
    out = dev.copy()
    out[ok] = dev[ok] - Zc.to_numpy() @ beta
    return out


def _within_recipe_test(ds: Dataset, per_batch: pd.DataFrame, dev: pd.Series, c: str,
                        rng: np.random.Generator) -> dict:
    kind = ds.candidates[c].kind
    raw = ds.batches.set_index("batch")[c].reindex(per_batch.index)
    x = ds.feature(c).reindex(per_batch.index) if kind != "category" else raw
    b = pd.DataFrame({"recipe": per_batch["recipe"], "dev": dev, "x": x}).dropna()
    b = b[b.groupby("recipe")["x"].transform("size") >= 2]
    out = dict(column=c, level="batch", kind=kind, coverage=_coverage(ds, c), batches_tested=len(b))
    if len(b) < 8:
        return {**out, "p": 1.0, "strength": 0.0, "note": "too few batches with a value"}

    groups = [idx for idx in b.groupby("recipe").indices.values()]
    order = np.concatenate(groups)
    b = b.iloc[order]
    bounds = np.cumsum([0] + [len(g) for g in groups])
    d = (b["dev"] - b.groupby("recipe")["dev"].transform("mean")).to_numpy()

    def permute(v):
        return np.concatenate([rng.permutation(v[bounds[i]:bounds[i + 1]]) for i in range(len(groups))])

    if kind == "category":
        labels = b["x"].astype(str).to_numpy()
        total = float((d ** 2).sum())

        def stat(lab):
            s = pd.Series(d).groupby(lab).agg(["mean", "size"])
            return float((s["size"] * s["mean"] ** 2).sum() / total)

        s0 = stat(labels)
        null = np.array([stat(permute(labels)) for _ in range(N_PERM)])
        means = pd.Series(d).groupby(labels).mean()
        out.update(levels=int(len(means)), spread_pct=round(100 * (np.exp(means.max()) - np.exp(means.min())), 1))
    else:
        xv = (b["x"] - b.groupby("recipe")["x"].transform("mean")).to_numpy(dtype=float)
        if xv.std() == 0:
            return {**out, "p": 1.0, "strength": 0.0, "note": "constant within every recipe"}

        def stat(v):
            return abs(float(np.corrcoef(v, d)[0, 1]))

        s0 = stat(xv)
        null = np.array([stat(permute(xv)) for _ in range(N_PERM)])
        slope = float(np.polyfit(xv, d, 1)[0])
        out.update(pct_strength_per_unit=round(100 * (np.exp(slope) - 1), 3),
                   within_recipe_range=[_r(b["x"].min()), _r(b["x"].max())])
    p = (1 + int((null >= s0).sum())) / (N_PERM + 1)
    return {**out, "strength": round(s0, 3), "p": round(p, 4),
            "scatter_explained": round(s0 ** 2 if kind != "category" else s0, 3)}


def _associations(ds: Dataset, c: str, others: list[str]) -> list[dict]:
    """Columns that a significant label or date stands for (e.g. lot -> mill cert)."""
    b = ds.batches.set_index("batch")
    out = []
    for o in others:
        x, y = b[c].astype(str), b[o]
        if ds.candidates[o].kind == "category":
            # same partition? (each value of c maps to one value of o, or vice versa)
            pairs = pd.DataFrame({"c": x, "o": y.astype(str)})
            purity = pairs.groupby("c")["o"].nunique().eq(1).mean()
            if purity > 0.9:
                out.append(dict(column=o, how=f"each {c} has one {o}"))
            continue
        yv = ds.feature(o)
        ok = yv.notna()
        if ok.sum() < 8:
            continue
        if ds.candidates[c].kind == "category":
            const = pd.DataFrame({"c": x[ok], "y": yv[ok]}).groupby("c")["y"].nunique().eq(1).mean()
            if const > 0.9:
                out.append(dict(column=o, how=f"{o} is fixed per {c}"))
        else:
            r = abs(float(np.corrcoef(ds.feature(c)[ok], yv[ok])[0, 1]))
            if r > 0.5:
                out.append(dict(column=o, how=f"correlated (r = {r:.2f})"))
    return out


def _missing_by(ds: Dataset, c: str) -> dict | None:
    """If gaps follow a label (e.g. whole cement lots without a certificate), name them."""
    b = ds.batches.set_index("batch")
    v = ds.feature(c)
    for o, k in ds.candidates.items():
        if k.kind != "category" or o == c:
            continue
        lab = b[o].astype(str)
        known = pd.DataFrame({"lab": lab[v.notna()], "v": v[v.notna()]})
        if known.empty or not known.groupby("lab")["v"].nunique().eq(1).all():
            continue
        missing = sorted(set(lab[v.isna()]) - set(known["lab"]))
        if missing and set(lab[v.isna()]) <= set(missing):
            return dict(column=o, missing_values=missing,
                        batches_affected=int(v.isna().sum()))
    return None


# --- refits -------------------------------------------------------------------------

def _job(ds: Dataset, extras: list[str], rows: pd.DataFrame | None = None,
         shuffle: str | None = None, rng: np.random.Generator | None = None) -> boxfit.KFoldJob:
    rows = ds.rows if rows is None else rows
    X, Y = ds.tensors(extras, rows)
    if shuffle is not None:  # shuffle a recipe column across recipes
        j = 10 + extras.index(shuffle)
        recipes = rows["recipe"].to_numpy()
        uniq = np.unique(recipes)
        vals = {r: float(X[np.flatnonzero(recipes == r)[0], j]) for r in uniq}
        perm = dict(zip(uniq, rng.permutation([vals[r] for r in uniq])))
        X[:, j] = X.new_tensor([perm[r] for r in recipes])
    return boxfit.KFoldJob(X, Y, k=5, seed=0, groups=rows["recipe"].to_numpy())


def _kfold28_many(jobs: list[boxfit.KFoldJob]) -> list[float]:
    """28-day RMSE (psi) of each job, all folds run in parallel."""
    return [boxfit.rmse(resid[_age28(job.X)]) for job, (resid, _) in zip(jobs, boxfit.kfold_many(jobs))]


def _age28(X) -> np.ndarray:
    t = X[:, boxfit.TIME].numpy()
    return t == (28 if (t == 28).any() else t.max())


def _coverage(ds: Dataset, c: str) -> float:
    return round(float(ds.batches[c].notna().mean()), 3)


def _r(x: float) -> float:
    return float(f"{x:.4g}")
