"""Saved model versions: models/<plant>/v<n>/ with weights, training data and metadata."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from boxcrete import fit_strength_gp
from boxcrete.utils import derive_bounds_from_X
from botorch.models import SingleTaskGP

from mixpilot import boxfit
from mixpilot.boxfit import Settings
from mixpilot.dataset import RECIPE_FIELDS, Dataset

ROOT = Path(__file__).resolve().parent.parent / "models"


def slug(plant: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", plant.lower()).strip("-") or "plant"


def versions(plant: str) -> list[dict]:
    d = ROOT / slug(plant)
    out = [json.loads(p.read_text()) for p in sorted(d.glob("v*/meta.json"), key=lambda p: int(p.parent.name[1:]))]
    return out


def save(plant: str, ds: Dataset, automl: dict, mapping: dict) -> dict:
    """Fit the winning setup on all the plant's data and save it as the next version."""
    win = automl["winner"]
    inputs, settings = win["inputs"], Settings(**win["settings"])
    X, Y = ds.tensors(inputs)
    model = boxfit.fit(X, Y, settings)

    d = ROOT / slug(plant)
    n = len(versions(plant)) + 1
    vdir = d / f"v{n}"
    vdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), vdir / "model.pt")
    torch.save({"X": X, "Y": Y}, vdir / "train.pt")
    # the training rows with their extra inputs, so a saved model can be used without the plant's files
    ds.rows.assign(**{c: ds.filled(c).reindex(ds.rows["batch"]).to_numpy() for c in inputs}).to_csv(
        vdir / "rows.csv", index=False)

    typical = {c: _typical(ds, c) for c in inputs}
    best_row = next(r for r in automl["leaderboard"] if r["best"])
    meta = dict(
        plant=plant, version=f"v{n}", trained=datetime.now().isoformat(timespec="minutes"),
        inputs=inputs, input_levels={c: ds.candidates[c].level for c in inputs},
        settings=settings.as_dict(), setup=win["name"],
        error_28d_psi=win["error_28d_psi"], vs_meta_default=win["vs_baseline"],
        vs_meta_default_28d=win["vs_baseline_28d"], errors_by_age_psi=best_row["errors_by_age_psi"],
        recipes=int(ds.rows["recipe"].nunique()), batches=int(ds.rows["batch"].nunique()),
        cylinders=ds.cylinders, typical_extra_values=typical,
        ranges={f: [float(ds.rows[f].min()), float(ds.rows[f].max())] for f in RECIPE_FIELDS},
        leaderboard=automl["leaderboard"], mapping=mapping, active=True,
    )
    for old in versions(plant):
        if old.get("active"):
            old["active"] = False
            (d / old["version"] / "meta.json").write_text(json.dumps(old, indent=1, default=str))
    (vdir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def load(plant: str, version: str | None = None) -> tuple[SingleTaskGP, dict]:
    """Rebuild a saved model without refitting (same recipe as Meta's pretrained loader)."""
    metas = versions(plant)
    if not metas:
        raise FileNotFoundError(f"no saved models for {plant}")
    meta = next((m for m in metas if m["version"] == version), None) if version else \
        next((m for m in metas if m.get("active")), metas[-1])
    if meta is None:
        raise FileNotFoundError(f"{plant} has no {version}; have {[m['version'] for m in metas]}")
    vdir = ROOT / slug(plant) / meta["version"]
    train = torch.load(vdir / "train.pt", weights_only=True)
    s = meta["settings"]
    model = fit_strength_gp(X=train["X"], Y=train["Y"].reshape(-1, 1), X_bounds=derive_bounds_from_X(train["X"]),
                            max_optimizer_iter=0, gate_tau=s["gate_tau"], shrinkage_sigma=s["shrinkage_sigma"])
    model.load_state_dict(torch.load(vdir / "model.pt", weights_only=True))
    model.train()
    model.eval()
    return model, meta


def _typical(ds: Dataset, c: str) -> dict:
    v = ds.feature(c).dropna()
    return dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
