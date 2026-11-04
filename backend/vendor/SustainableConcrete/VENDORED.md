# Vendored BOxCrete

Upstream: https://github.com/facebookresearch/SustainableConcrete (MIT, see `LICENSE`)
Commit: `108811e842afcd1782c16eccb55eaf034942f7dd` (2026-10-22)

Copied: `boxcrete/`, `data/boxcrete_data.csv`, `docs/model/strength_model.pt`,
`docs/model/test_vectors.json`, and the tests for the patched files.

## MixPilot patch

Every change is marked `MixPilot patch` in the code. Defaults reproduce upstream
exactly (block-LOO NLL −1.6502 on the public data, before and after).

- **Extra inputs after Time.** Plant-specific columns are appended after the 10
  `DEFAULT_X_COLUMNS`, so every index in `features.IDX` stays valid.
  - `strength_model.py`: the input transform finds Time by `IDX["time"]` instead of
    assuming it is the last column.
  - `kernels.py`: no change needed; dims `[10, d_aug)` were already treated as extras
    with free lengthscales.
  - `block_loo.py`: a held-out block is defined by every non-time input, including
    the extras.
- **Tunable settings.** `fit_strength_gp(..., gate_tau=, shrinkage_sigma=)` replaces the
  `GATE_TAU` and `_LENGTHSCALE_SHRINKAGE_SIGMA` constants (`mll_weight` was already an
  argument of `train_block_loo`).

`scripts/check_patch.py` (in `backend/`) checks both.
