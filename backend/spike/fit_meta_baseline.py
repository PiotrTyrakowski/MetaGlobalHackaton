"""Fit BOxCrete's production strength GP on Meta's own data and time it."""

import time

import torch

from boxcrete import fit_strength_gp, load_concrete_strength
from boxcrete.block_loo import train_block_loo

data = load_concrete_strength()
X, Y, Yvar, bounds = data.strength_data
print(f"rows={X.shape[0]} inputs={X.shape[-1]} threads={torch.get_num_threads()}")

t0 = time.time()
model = fit_strength_gp(X=X, Y=Y, Yvar=Yvar, X_bounds=bounds, seed=0, max_optimizer_iter=0)
loss = train_block_loo(model, X.shape[0], max_iter=150, lr=0.1)
print(f"fit seconds={time.time() - t0:.1f} final_bloo_loss={loss:.4f}")
