"""jax_ng.optimizers
====================
Second-order PINN optimizers.

All optimizers share the interface::

    opt_state = optimizer.init(params)
    loss, params, opt_state = optimizer.step(params, opt_state, key)

Submodules
----------
gauss_newton   GaussNewton, SolveConfig
multistage     MultiStageGN, PhaseConfig
windowed_gn    WindowedJacobiGN
stokes_gn      StokesJacobiGN
"""
from jax_ng.optimizers.gauss_newton import GaussNewton, SolveConfig
from jax_ng.optimizers.multistage   import MultiStageGN, PhaseConfig
from jax_ng.optimizers.windowed_gn  import WindowedJacobiGN
from jax_ng.optimizers.stokes_gn    import StokesJacobiGN


def build(name: str, interior_res_fn, sampler_fn,
          linesearch_fn=None, boundary_res_fn=None, **kwargs):
    """Convenience factory.

    Parameters
    ----------
    name            : ``"gauss_newton"``
    interior_res_fn : ``(params, x) -> residual``
    sampler_fn      : ``(key) -> (x_int, x_bnd)``
    linesearch_fn   : required for ``"gauss_newton"``
    boundary_res_fn : optional
    **kwargs        : forwarded to ``SolveConfig``
    """
    if name == "gauss_newton":
        from jax_ng.linesearch import grid_search
        ls  = linesearch_fn or grid_search
        cfg_fields = SolveConfig.__dataclass_fields__
        cfg = SolveConfig(**{k: v for k, v in kwargs.items() if k in cfg_fields})
        extra = {k: v for k, v in kwargs.items() if k not in cfg_fields}
        return GaussNewton(interior_res_fn, boundary_res_fn, sampler_fn, ls,
                           solve_config=cfg, **extra)
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Choose 'gauss_newton'.")


__all__ = [
    "GaussNewton", "SolveConfig",
    "MultiStageGN", "PhaseConfig",
    "WindowedJacobiGN",
    "StokesJacobiGN",
    "build",
]
