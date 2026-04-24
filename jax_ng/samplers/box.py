"""Uniform collocation on hyper-rectangle domains."""
from functools import partial

import jax
import jax.numpy as jnp
from jax import random


@partial(jax.jit, static_argnums=(1, 2, 3))
def uniform_box(key, n_interior: int, n_boundary: int, box: tuple):
    """Uniform interior points + random-face boundary points on a box.

    Parameters
    ----------
    box : tuple of ``(low, high)`` pairs, e.g. ``((-1.,1.), (-1.,1.))``

    Returns
    -------
    x_int : ``(n_interior, d)``
    x_bnd : ``(n_boundary, d)`` or ``(0, d)`` if n_boundary == 0
    """
    lows  = jnp.array([lo for lo, _ in box])
    highs = jnp.array([hi for _, hi in box])
    span  = highs - lows
    d     = len(box)

    k1, k2, k3 = random.split(key, 3)
    x_int = random.uniform(k1, (n_interior, d)) * span + lows

    if n_boundary > 0:
        face_ids = random.randint(k2, (n_boundary,), 0, 2 * d)
        dims  = face_ids // 2
        sides = face_ids % 2
        x_bnd = random.uniform(k3, (n_boundary, d)) * span + lows
        fixed = jnp.where(sides[:, None] == 0, lows[dims][:, None], highs[dims][:, None])
        x_bnd = x_bnd.at[jnp.arange(n_boundary), dims].set(fixed.squeeze(-1))
    else:
        x_bnd = jnp.zeros((0, d))

    return x_int, x_bnd
