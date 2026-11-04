"""Time one deployed-recipe BOxCrete fit on Meta's public data."""

import time
import warnings

warnings.filterwarnings("ignore")

from boxcrete import fit_strength_gp, load_concrete_strength
from boxcrete.block_loo import train_block_loo

data = load_concrete_strength()
X, Y, Yvar, bounds = data.strength_data
print(f"n={X.shape[0]} rows, d={X.shape[-1]}")

t0 = time.perf_counter()
model = fit_strength_gp(X=X, Y=Y, Yvar=Yvar, X_bounds=bounds, seed=0, max_optimizer_iter=0)
loss = train_block_loo(model, X.shape[0], max_iter=150, lr=0.1)
print(f"fit: {time.perf_counter() - t0:.1f}s, block-LOO NLL {loss:.4f}")
