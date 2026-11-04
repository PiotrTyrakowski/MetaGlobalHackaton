"""Check the synthetic data: can the hidden effects be found, and do they help BOxCrete?

Uses the true compositions (kg/m3) from demo_data/truth.json and the cylinder file.
"""

import json
import time

import numpy as np
import pandas as pd
import torch

from mixpilot import boxfit

t = json.load(open("demo_data/truth.json"))
batches = pd.DataFrame(t["batches"])
batches.loc[~batches.mill_cert_recorded, "mill_cert_psi"] = np.nan
cyl = pd.read_excel("demo_data/plant/cylinder_breaks.xlsx")
cyl["age"] = cyl["Age"].str.extract(r"(\d+)").astype(float)
means = cyl.groupby(["Ticket #", "age"])["Strength (psi)"].mean().reset_index()
df = means.merge(batches, left_on="Ticket #", right_on="ticket")
df["temp_c"] = (df["cure_temp_f"] - 32) * 5 / 9
BASE = ["cement", "fly_ash", "slag", "water", "hrwr_kg_m3", "fine", "coarse"]


def tensors(extras, rows):
    cols = [rows[c] for c in BASE] + [pd.Series(2.0, index=rows.index), rows["temp_c"], rows["age"]]
    cols += [rows[c] for c in extras]
    return (torch.tensor(np.column_stack(cols), dtype=torch.float64),
            torch.tensor(rows["Strength (psi)"].values, dtype=torch.float64))


# --- 1. Within-recipe test for batch measurements -------------------------------
def within_design_test(rows, candidates, n_perm=5000, seed=0):
    """Does the column explain batch-to-batch scatter of the same recipe?"""
    rng = np.random.default_rng(seed)
    log_s = np.log(rows["Strength (psi)"])
    rows = rows.assign(dev=log_s - log_s.groupby([rows.code, rows.age]).transform("mean"))
    per_batch = rows.groupby("ticket").agg(code=("code", "first"), dev=("dev", "mean"),
                                           **{c: (c, "first") for c in candidates})
    out = []
    for c in candidates:
        b = per_batch.dropna(subset=[c])
        b = b[b.groupby("code")[c].transform("size") >= 2]
        if pd.api.types.is_numeric_dtype(b[c]):
            x = (b[c] - b.groupby("code")[c].transform("mean")).to_numpy()
            d = (b.dev - b.groupby("code").dev.transform("mean")).to_numpy()
            stat = lambda xx: abs(np.corrcoef(xx, d)[0, 1]) if xx.std() > 0 else 0.0
            perm = lambda: np.concatenate([rng.permutation(x[idx]) for idx in b.groupby("code").indices.values()])
            # keep row order aligned with the concatenated permutation
            order = np.concatenate(list(b.groupby("code").indices.values()))
            x, d = x[order], d[order]
        else:
            d = (b.dev - b.groupby("code").dev.transform("mean")).to_numpy()
            cats = b[c].to_numpy()
            def stat(cc):
                s = pd.Series(d).groupby(cc).agg(["mean", "size"])
                return float((s["size"] * s["mean"] ** 2).sum() / (d ** 2).sum())
            groups = list(b.groupby("code").indices.values())
            order = np.concatenate(groups)
            x, d = cats[order], d[order]
            sizes = [len(g) for g in groups]
            def perm():
                parts, i = [], 0
                for n in sizes:
                    parts.append(rng.permutation(x[i:i + n])); i += n
                return np.concatenate(parts)
        s0 = stat(x)
        null = np.array([stat(perm()) for _ in range(n_perm)])
        out.append(dict(column=c, coverage=rows.groupby("ticket")[c].first().notna().mean(),
                        r=round(s0, 3), p=(1 + (null >= s0).sum()) / (n_perm + 1)))
    return pd.DataFrame(out).sort_values("p")


# --- 2. Honest model comparison (refit, whole designs held out) -----------------
def kfold28(extras, rows, seeds=range(2)):
    X, Y = tensors(extras, rows)
    a28 = (X[:, boxfit.TIME] == 28).numpy()
    groups = rows["code"].to_numpy()
    return np.mean([boxfit.rmse(boxfit.kfold_residuals(X, Y, k=5, seed=s, groups=groups)[0][a28])
                    for s in seeds])


if __name__ == "__main__":
    print(f"{len(df)} rows, {df.ticket.nunique()} batches, {df.code.nunique()} designs")
    cands = ["air", "mill_cert_psi", "slump_in", "conc_temp_f", "ticket", "truck", "operator",
             "cement_lot"]
    t0 = time.perf_counter()
    print(within_design_test(df, cands).to_string(index=False),
          f"\n({time.perf_counter() - t0:.0f}s)")

    print("--- 28-day RMSE, 5-fold refit holding out whole designs")
    rng = np.random.default_rng(1)
    sf_by_design = df.groupby("code").sf.first()
    df["sf_shuffled"] = df.code.map(dict(zip(sf_by_design.index, rng.permutation(sf_by_design.values))))
    for name, extras in [("Meta's 10 inputs", []), ("+ air", ["air"]),
                         ("+ silica fume", ["sf"]), ("+ shuffled silica fume", ["sf_shuffled"]),
                         ("+ air + silica fume", ["air", "sf"]),
                         ("+ air + sf + conc temp (no effect)", ["air", "sf", "conc_temp_f"])]:
        t0 = time.perf_counter()
        print(f"  {name:36s} {kfold28(extras, df):5.0f} psi  ({time.perf_counter() - t0:.0f}s)")
    sub = df[df.mill_cert_psi.notna()]
    print(f"--- on the {sub.ticket.nunique()} batches with a mill cert")
    for name, extras in [("+ air + sf", ["air", "sf"]), ("+ air + sf + mill cert", ["air", "sf", "mill_cert_psi"])]:
        print(f"  {name:36s} {kfold28(extras, sub):5.0f} psi")
