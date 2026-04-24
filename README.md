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
