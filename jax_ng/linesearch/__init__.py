"""jax_ng.linesearch
====================
Pluggable line-search strategies.

All callables share the signature::

    ls(loss_fn, params, direction, current_loss, **kwargs) -> (alpha, loss)

Submodules
----------
grid        grid_search  — vectorised geometric grid (JIT-friendly)
backtrack   armijo       — sufficient-decrease backtracking
            wolfe        — Armijo + curvature (weak Wolfe)
fixed       fixed_step   — constant step, no search
"""
from jax_ng.linesearch.grid      import grid_search
from jax_ng.linesearch.backtrack import armijo, wolfe
from jax_ng.linesearch.fixed     import fixed_step

from functools import partial


def build(name: str, **kwargs):
    """Return a partially applied line-search by name.

    Parameters
    ----------
    name : ``"grid_search"`` | ``"armijo"`` | ``"wolfe"`` | ``"fixed"``

    Example
    -------
    >>> ls = linesearch.build("grid_search", n_steps=20)
    >>> alpha, loss = ls(loss_fn, params, direction, f0)
    """
    registry = {
        "grid_search": grid_search,
        "armijo":      armijo,
        "wolfe":       wolfe,
        "fixed":       fixed_step,
    }
    if name not in registry:
        raise ValueError(f"Unknown line-search '{name}'. Choose from {list(registry)}")
    return partial(registry[name], **kwargs)


__all__ = ["grid_search", "armijo", "wolfe", "fixed_step", "build"]
