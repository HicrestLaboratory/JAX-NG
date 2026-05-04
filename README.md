# JAX-NG

JAX-NG is a modular JAX framework for second-order optimization of physics-informed neural networks (PINNs), with a focus on Gauss-Newton and natural-gradient-style methods that scale beyond the regime typically accessible to parameter-space solvers.

This repository accompanies the paper:

**Dual Natural Gradient Descent for Scalable Training of Physics-Informed Neural Networks**  
Anas Jnini, Flavio Vella  
Published in *Transactions on Machine Learning Research* (2025)

JAX-NG is designed as a research codebase for building and evaluating second-order PINN training pipelines across multiple PDE benchmarks, with reusable components for models, sampling, line search, optimization, training, and problem definitions.

## Highlights

- Modular JAX implementation for second-order PINN optimization
- Gauss-Newton solvers with automatic primal/dual system selection
- Problem implementations covering elliptic, fluid, and time-dependent PDEs
- Example scripts for reproducing supported experiments

## Current Support

### Optimizers

- `GaussNewton`
- `MultiStageGN`
- `WindowedJacobiGN`
- `StokesJacobiGN`

### Problems

- Helmholtz
- Kovasznay
- KdV with windowed hard-IC ansatz
- KS1d with windowed hard-IC ansatz
- Stokes wedge with pressure anchor
- Beltrami 3D (space-time Navier-Stokes benchmark)

### Example Scripts

- `jax_ng/examples/helmholtz_gn.py`
- `jax_ng/examples/kovasznay_gn.py`
- `jax_ng/examples/kdv_windowed_gn.py`
- `jax_ng/examples/beltrami_gn.py`

## Repository Layout

```text
jax_ng/
  models/        # activations, initialization, MLPs, jets
  samplers/      # box and triangle/wedge samplers
  linesearch/    # grid, Armijo, Wolfe, fixed-step rules
  optimizers/    # gauss_newton, multistage, windowed_gn, stokes_gn
  problems/      # helmholtz, kovasznay, kdv, ks1d, stokes_wedge, beltrami
  utils/         # trainer, metrics, checkpointing, plotting
  examples/      # runnable scripts
  tests/         # pytest suite
```

## Installation

For a clean setup, we recommend creating a fresh Conda environment first:

```bash
conda create -n jax-ng python=3.11 -y
conda activate jax-ng
git clone https://github.com/HicrestLaboratory/JAX-NG.git
cd JAX-NG
pip install -e .
```

If the repository includes a `requirements.txt`, install it before the editable package install:

```bash
pip install -r requirements.txt
pip install -e .
```

If you are using a CPU-only environment, a common setup is:

```bash
pip install --upgrade "jax[cpu]"
pip install -e .
```

Python imports follow the package namespace:

```python
import jax_ng
```

## Data Files

Some examples expect external data files under `./data`:

- `kdv.mat`
- `ks_chaotic.mat`
- `st_flow.csv`

Make sure these files are available before running the corresponding scripts.

## Running Examples

From the repository root, the bundled examples can be launched with:

```bash
python -m jax_ng.examples.helmholtz_gn
python -m jax_ng.examples.kovasznay_gn
python -m jax_ng.examples.kdv_windowed_gn
python -m jax_ng.examples.ks1d_windowed_gn
python -m jax_ng.examples.stokes_gn
```

The KdV, KS1d, and Stokes examples require the corresponding files in `./data`.

## Running Tests

Install `pytest` if needed:

```bash
pip install pytest
```

Then run the full test suite from the repository root:

```bash
pytest
```

You can also target the package tests directly:

```bash
pytest jax_ng/tests
```

For more verbose output:

```bash
pytest -v
```

## Sanity Checks

After installation, the following quick checks are useful:

```bash
python -c "import jax_ng; print('jax_ng import OK')"
python -c "import jax; print(jax.__version__)"
```

## Quick Start

The snippet below illustrates the basic workflow for defining a problem, sampling collocation points, constructing a Gauss-Newton optimizer, and running training.

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

## Solver Modes

JAX-NG supports automatic selection between primal and dual linear systems through:

```python
optimizers.SolveConfig(mode="auto")
```

The current rule is:

- `dual` when `N_params > N_residuals`
- `primal` otherwise

You can also force the system explicitly with:

- `mode="dual"`
- `mode="primal"`

## Intended Use

This codebase is intended as a companion research repository for the paper and as a starting point for:

- reproducing reported experiments
- extending second-order PINN optimizers
- adding new PDE benchmarks and sampling schemes
- experimenting with primal and residual-space Gauss-Newton formulations

## Citation

If you use this repository in academic work, please cite the accompanying paper:

```bibtex
@article{jnini2025dual,
  title   = {Dual Natural Gradient Descent for Scalable Training of Physics-Informed Neural Networks},
  author  = {Jnini, Anas and Vella, Flavio},
  journal = {Transactions on Machine Learning Research},
  year    = {2025}
}
```

## Status

JAX-NG is an active research-oriented codebase. The repository currently supports the optimizers, problems, and examples listed above, with the structure intentionally kept modular for further extensions.
