"""Check the patched BOxCrete: defaults unchanged, extra inputs after Time work."""

import time
import warnings

warnings.filterwarnings("ignore")

import torch
from boxcrete import fit_strength_gp, load_concrete_strength
from boxcrete.block_loo import train_block_loo

data = load_concrete_strength()
X, Y, Yvar, bounds = data.strength_data

t0 = time.perf_counter()
m = fit_strength_gp(X=X, Y=Y, Yvar=Yvar, X_bounds=bounds, seed=0, max_optimizer_iter=0)
loss = train_block_loo(m, X.shape[0])
print(f"defaults: {time.perf_counter() - t0:.1f}s, NLL {loss:.4f} (unpatched: -1.6502)")

# Two extra columns after Time: one informative (tracks strength), one pure noise.
g = torch.Generator().manual_seed(0)
informative = (Y / Y.max()) + 0.05 * torch.randn(Y.shape, generator=g, dtype=Y.dtype)
noise = torch.randn(Y.shape, generator=g, dtype=Y.dtype)
X2 = torch.cat([X, informative, noise], dim=-1)
b2 = torch.cat([bounds, torch.stack([X2[:, 10:].amin(0), X2[:, 10:].amax(0)])], dim=-1)

t0 = time.perf_counter()
m2 = fit_strength_gp(X=X2, Y=Y, Yvar=Yvar, X_bounds=b2, seed=0, max_optimizer_iter=0,
                     gate_tau=0.2, shrinkage_sigma=0.01)
loss2 = train_block_loo(m2, X2.shape[0], mll_weight=0.3)
blind = m2.covar_module.base_kernel.kernels[0].base_kernel
print(f"12 inputs: {time.perf_counter() - t0:.1f}s, NLL {loss2:.4f}, "
      f"tau={m2.covar_module.tau.item():.2f}, blind lengthscales={blind.lengthscale.shape[-1]}")
ls = blind.lengthscale.squeeze()
print(f"  lengthscale informative={ls[9]:.3g}  noise={ls[10]:.3g}  (smaller = more relevant)")
with torch.no_grad():
    post = m2.posterior(X2[:3])
print("  predict ok:", post.mean.shape)
