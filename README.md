# JAX-NG

Modular JAX framework for second-order PINN optimization with Gauss-Newton variants.

Python imports:
- `import jax_ng`

## Project layout
```text
jax_ng/
  models/        # activations, init, MLP, jets
  samplers/      # box and triangle/wedge samplers
  linesearch/    # grid, armijo, wolfe, fixed-step
  optimizers/    # gauss_newton, multistage, windowed_gn, stokes_gn
  problems/      # helmholtz, kovasznay, kdv, ks1d, stokes_wedge
  utils/         # trainer, metrics, checkpointing, plotting
  examples/      # runnable scripts
  tests/         # pytest suite
```

## Included problems
- Helmholtz
- Kovasznay
- KdV (windowed hard-IC ansatz)
- KS1d (windowed hard-IC ansatz)
- Stokes wedge (with pressure anchor)

## Included optimizers
- `GaussNewton`
- `MultiStageGN`
- `WindowedJacobiGN`
- `StokesJacobiGN`

## Examples
- `jax_ng/examples/helmholtz_gn.py`
- `jax_ng/examples/kovasznay_gn.py`
- `jax_ng/examples/kdv_windowed_gn.py`
- `jax_ng/examples/ks1d_windowed_gn.py`
- `jax_ng/examples/stokes_gn.py`

## Data
The examples expect data files under `./data`:
- `kdv.mat`
- `ks_chaotic.mat`
- `st_flow.csv`

## Install
```bash
pip install -e .
```

## Quick usage: simple problem
```python
import jax
import jax.numpy as jnp
from jax import random
from jax_ng import linesearch, models, optimizers, samplers, utils

jax.config.update("jax_enable_x64", True)

class SimplePoisson1D:
    def exact_u(self, x):
        return jnp.sin(jnp.pi * x[0])

    def forcing(self, x):
        return -(jnp.pi ** 2) * jnp.sin(jnp.pi * x[0])

    def interior_res(self, params, x):
        u, lap_u = models.jet_laplacian(params, x)
        return lap_u[0] - self.forcing(x)

    def boundary_res(self, params, x):
        u, _ = models.jet_laplacian(params, x)
        return u[0] - self.exact_u(x)

    def init_params(self, key):
        sizes = models.layer_sizes(input_dim=1, width=32, depth=3, output_dim=1)
        return models.glorot_init(sizes, key)

pde = SimplePoisson1D()
params = pde.init_params(random.PRNGKey(0))
sampler = lambda key: samplers.uniform_box(key, 256, 64, ((-1.0, 1.0),))
ls = linesearch.build("grid_search", n_steps=12)

opt = optimizers.GaussNewton(
    interior_res_fn=pde.interior_res,
    boundary_res_fn=pde.boundary_res,
    sampler_fn=sampler,
    linesearch_fn=ls,
    solve_config=optimizers.SolveConfig(mode="auto", damping=1e-8),
)

trainer = utils.Trainer(opt, n_iters=300, log_interval=50)
params, history = trainer.run(params, random.PRNGKey(1))
```

### Auto primal/dual selection
`SolveConfig(mode="auto")` selects the linear system based on Jacobian shape:
- `dual` when `N_params > N_residuals`
- `primal` otherwise

You can still force a mode with `mode="dual"` or `mode="primal"`.
