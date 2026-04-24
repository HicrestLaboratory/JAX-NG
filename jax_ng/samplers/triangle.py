"""Collocation samplers for triangular / wedge domains."""
from functools import partial

import jax
import jax.numpy as jnp
from jax import random


@partial(jax.jit, static_argnums=(1,))
def interior(key, n_points: int, v1, v2, v3):
    """Uniform sampling inside a 2-D triangle (square-root trick).

    Returns
    -------
    x : ``(n_points, 2)``
    """
    k1, k2 = random.split(key)
    r1 = random.uniform(k1, (n_points, 1))
    r2 = random.uniform(k2, (n_points, 1))
    sqrt_r1 = jnp.sqrt(r1)
    w1 = 1.0 - sqrt_r1
    w2 = sqrt_r1 * (1.0 - r2)
    w3 = sqrt_r1 * r2
    return w1 * v1 + w2 * v2 + w3 * v3


@partial(jax.jit, static_argnums=(1,))
def boundary(key, n_boundary: int, v1, v2, v3):
    """Uniform sampling on the three edges of a triangle.

    Returns
    -------
    x_bnd : ``(n_boundary, 2)``
    """
    n_e = n_boundary // 3
    t   = random.uniform(key, (n_boundary, 1))
    xb_top   = v2 + t * (v3 - v2)
    xb_left  = v2 + t * (v1 - v2)
    xb_right = v3 + t * (v1 - v3)
    idx = jnp.arange(n_boundary)
    xb  = jnp.where((idx < n_e)[:, None],      xb_top,   xb_left)
    xb  = jnp.where((idx >= 2 * n_e)[:, None], xb_right, xb)
    return xb


def wedge(key, n_interior: int, n_boundary: int, v1, v2, v3):
    """Interior + boundary for a wedge (triangle) geometry.

    Returns
    -------
    x_int : ``(n_interior, 2)``
    x_bnd : ``(n_boundary, 2)``
    """
    k1, k2 = random.split(key)
    return interior(k1, n_interior, v1, v2, v3), boundary(k2, n_boundary, v1, v2, v3)


@partial(jax.jit, static_argnums=(1, 5))
def apex_heavy(key, n_points: int, v1, v2, v3, apex_fraction: float = 0.9):
    """Log-scale concentrated sampling near the wedge apex.

    Useful for problems with singularities at the tip.

    Parameters
    ----------
    apex_fraction : fraction of points clustered near the apex (default 0.9)
    """
    n_uni  = int(round(n_points * (1.0 - apex_fraction)))
    n_apex = n_points - n_uni

    k1, k2, k3 = random.split(key, 3)
    x_uni = interior(k1, n_uni, v1, v2, v3)

    log_y  = random.uniform(k2, (n_apex, 1), minval=jnp.log(1e-7), maxval=jnp.log(2.0))
    y_apex = jnp.exp(log_y)
    x_off  = random.uniform(k3, (n_apex, 1), minval=-0.5, maxval=0.5) * 0.5 * y_apex
    x_apex = jnp.hstack([v1[0] + x_off, y_apex])

    return jnp.concatenate([x_uni, x_apex], axis=0)
