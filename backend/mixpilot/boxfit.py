"""Fit and score BOxCrete on a plant's data.

X columns are BOxCrete's 10 DEFAULT_X_COLUMNS followed by any plant-specific
extras (see vendor/SustainableConcrete/VENDORED.md). Y is mean strength in psi,
one row per (mix, age).
"""

from __future__ import annotations

import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import torch
from boxcrete import fit_strength_gp
from boxcrete.block_loo import train_block_loo
from boxcrete.features import GATE_TAU, IDX
from boxcrete.priors import _LENGTHSCALE_SHRINKAGE_SIGMA
from boxcrete.utils import derive_bounds_from_X
from botorch.models import SingleTaskGP
from gpytorch.utils.cholesky import psd_safe_cholesky

warnings.filterwarnings("ignore", module="gpytorch")
warnings.filterwarnings("ignore", module="linear_operator")
warnings.filterwarnings("ignore", category=FutureWarning)

TIME = IDX["time"]


@dataclass(frozen=True)
class Settings:
    """The knobs BOxCrete hard-codes; defaults are Meta's production values."""

    gate_tau: float = GATE_TAU
    shrinkage_sigma: float = _LENGTHSCALE_SHRINKAGE_SIGMA
    mll_weight: float = 0.5

    def as_dict(self) -> dict:
        return asdict(self)


META_DEFAULT = Settings()


def fit(X: torch.Tensor, Y: torch.Tensor, settings: Settings = META_DEFAULT,
        seed: int = 0) -> SingleTaskGP:
    """Fit with Meta's deployed recipe: build the V2 GP, then block-LOO training."""
    X = X.to(torch.float64)
    Y = Y.to(torch.float64).reshape(-1, 1)
    model = fit_strength_gp(
        X=X, Y=Y, X_bounds=derive_bounds_from_X(X), seed=seed, max_optimizer_iter=0,
        gate_tau=settings.gate_tau, shrinkage_sigma=settings.shrinkage_sigma,
    )
    train_block_loo(model, X.shape[0], mll_weight=settings.mll_weight)
    model.eval()
    return model


def predict(model: SingleTaskGP, X: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and standard deviation in psi, in chunks so memory stays small."""
    y_max = float(model._study_y_std)
    means, sds = [], []
    with torch.no_grad():
        for chunk in X.to(torch.float64).split(1000):
            post = model.posterior(chunk)
            means.append(post.mean.squeeze(-1) * y_max)
            sds.append(post.variance.squeeze(-1).clamp_min(0).sqrt() * y_max)
    return torch.cat(means).numpy(), torch.cat(sds).numpy()


def loo_residuals(model: SingleTaskGP) -> np.ndarray:
    """Closed-form leave-one-mix-out residuals (observed - predicted, psi).

    Same block-inverse identity as boxcrete.block_loo.block_loo_loss: each mix's
    rows are held out together, with the model's hyperparameters fixed.
    """
    model.train()  # raw (untransformed) train inputs live in train mode
    train_X = model.train_inputs[0]
    with torch.no_grad():
        prior = model(train_X)
        K = model.likelihood(prior, train_X).lazy_covariance_matrix.to_dense()
        L = psd_safe_cholesky(K)
        y = model.train_targets.reshape(-1, 1) - prior.mean.reshape(-1, 1)
        alpha = torch.cholesky_solve(y, L).squeeze(-1)
        K_inv = torch.cholesky_solve(torch.eye(K.shape[-1], dtype=K.dtype), L)
    model.eval()

    blocks = mix_blocks(train_X)
    resid = torch.empty_like(alpha)
    for idx in blocks:
        resid[idx] = torch.linalg.inv(K_inv[idx][:, idx]) @ alpha[idx]
    return (resid * float(model._study_y_std)).numpy()


def mix_blocks(X: torch.Tensor) -> list[torch.Tensor]:
    """Row indices of each mix: rows sharing every input except Time."""
    keep = [i for i in range(X.shape[-1]) if i != TIME]
    _, inverse = torch.unique(X[:, keep], dim=0, return_inverse=True)
    return [(inverse == g).nonzero(as_tuple=True)[0] for g in range(int(inverse.max()) + 1)]


@dataclass
class KFoldJob:
    """One k-fold evaluation: refit on k-1 folds, predict the held-out whole groups."""

    X: torch.Tensor
    Y: torch.Tensor
    settings: Settings = META_DEFAULT
    k: int = 5
    seed: int = 0
    groups: np.ndarray | None = None  # a group id per row (the mix design); default mix_blocks


def kfold_residuals(X: torch.Tensor, Y: torch.Tensor, settings: Settings = META_DEFAULT,
                    k: int = 5, seed: int = 0,
                    groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Refit-based k-fold residuals (psi). Returns (residuals per row, fold id per row)."""
    return kfold_many([KFoldJob(X, Y, settings, k, seed, groups)])[0]


def kfold_many(jobs: list[KFoldJob]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run several k-fold evaluations, all folds in parallel worker processes."""
    tasks, folds = [], []
    for job in jobs:
        fold = _assign_folds(job)
        folds.append(fold)
        for f in range(job.k):
            test = fold == f
            tasks.append((job.X[~test], job.Y.reshape(-1)[~test], job.X[test], job.settings))
    means = list(_pool().map(_fit_predict, tasks)) if len(tasks) > 1 else [_fit_predict(tasks[0])]

    out, i = [], 0
    for job, fold in zip(jobs, folds):
        resid = np.empty(job.X.shape[0])
        Yf = job.Y.reshape(-1).to(torch.float64).numpy()
        for f in range(job.k):
            test = fold == f
            resid[test] = Yf[test] - means[i]
            i += 1
        out.append((resid, fold))
    return out


def _assign_folds(job: KFoldJob) -> np.ndarray:
    if job.groups is None:
        blocks = mix_blocks(job.X)
    else:
        blocks = [torch.from_numpy(np.flatnonzero(job.groups == g)) for g in np.unique(job.groups)]
    order = np.random.default_rng(job.seed).permutation(len(blocks))
    fold = np.empty(job.X.shape[0], dtype=int)
    for f, b in enumerate(order):
        fold[blocks[b].numpy()] = f % job.k
    return fold


def _fit_predict(task) -> np.ndarray:
    X_train, Y_train, X_test, settings = task
    try:
        return predict(fit(X_train, Y_train, settings), X_test)[0]
    except Exception:  # extreme settings can make the fit fail; score it as missing
        return np.full(X_test.shape[0], np.nan)


_POOL: ProcessPoolExecutor | None = None


def _pool() -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        workers = int(os.environ.get("MIXPILOT_WORKERS", min(10, os.cpu_count() or 4)))
        _POOL = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker)
    return _POOL


def _init_worker() -> None:
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")


def warm_up() -> None:
    """Start the workers now, so the first job doesn't wait for them to import torch."""
    pool = _pool()
    list(pool.map(_idle, range(pool._max_workers)))


def _idle(_: int) -> None:
    pass


def rmse(resid: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(resid))))
