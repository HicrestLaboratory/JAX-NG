"""Forward-mode jet propagation.

Computes u, Jacobian ∂u/∂x, Hessian ∂²u/∂x², and Laplacian Δu
*without* any second-order AD — by propagating (value, grad, Hessian)
analytically through each layer using the chain rule.

This is numerically identical to jax.jacobian / jax.hessian but avoids
the O(d²) memory overhead of reverse-over-forward AD for large networks.

Functions
---------
full        — returns (u, J, H): value + Jacobian + full Hessian
laplacian   — returns (u, Δu): value + scalar Laplacian  (no Hessian storage)
laplacian_periodic — same but with Fourier input embedding
"""
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import jit

from jax_ng.models.activations import get as get_act


@partial(jit, static_argnums=(2,))
def full(params: list, x, activation: str = "tanh") -> Tuple:
    """Value, Jacobian and full Hessian via forward-mode jet propagation.

    Returns
    -------
    u : ``(n_out,)``
    J : ``(n_out, d)``     — Jacobian
    H : ``(n_out, d, d)``  — Hessian
    """
    act, act_p, act_pp = get_act(activation)
    d = x.shape[0]
    z   = x
    dz  = jnp.eye(d)
    d2z = jnp.zeros((d, d, d))

    for W, b in params[:-1]:
        z_pre   = jnp.dot(W, z) + b
        dz_pre  = jnp.dot(W, dz)
        d2z_pre = jnp.einsum("ij,jkl->ikl", W, d2z)
        s_p  = act_p(z_pre)
        s_pp = act_pp(z_pre)
        z   = act(z_pre)
        dz  = s_p[:, None] * dz_pre
        d2z = (s_pp[:, None, None] * jnp.einsum("ij,ik->ijk", dz_pre, dz_pre)
               + s_p[:, None, None] * d2z_pre)

    W_out, b_out = params[-1]
    u = jnp.dot(W_out, z)   + b_out
    J = jnp.dot(W_out, dz)
    H = jnp.einsum("ij,jkl->ikl", W_out, d2z)
    return u, J, H


@partial(jit, static_argnums=(2,))
def laplacian(params: list, x, activation: str = "tanh") -> Tuple:
    """Value and scalar Laplacian — avoids storing the full Hessian.

    Returns
    -------
    u     : ``(n_out,)``
    lap_u : ``(n_out,)`` — Laplacian trace of each output
    """
    act, act_p, act_pp = get_act(activation)
    d = x.shape[0]
    z   = x
    dz  = jnp.eye(d)
    lap = jnp.zeros(d)

    for W, b in params[:-1]:
        z_pre   = jnp.dot(W, z) + b
        dz_pre  = jnp.dot(W, dz)
        lap_pre = jnp.dot(W, lap)
        s_p  = act_p(z_pre)
        s_pp = act_pp(z_pre)
        z   = act(z_pre)
        lap = s_p * lap_pre + s_pp * jnp.sum(dz_pre ** 2, axis=1)
        dz  = s_p[:, None] * dz_pre

    W_out, b_out = params[-1]
    u     = jnp.dot(W_out, z)   + b_out
    lap_u = jnp.dot(W_out, lap)
    return u, lap_u


@partial(jit, static_argnums=(3, 4))
def laplacian_periodic(params: list, x, periods, n_modes: int,
                       activation: str = "tanh") -> Tuple:
    """Value and Laplacian for a Fourier-embedding MLP.

    Builds embedding jets analytically then propagates through MLP layers.

    Returns
    -------
    u     : scalar
    lap_u : scalar
    """
    act, act_p, act_pp = get_act(activation)
    d      = x.shape[0]
    omegas = 2.0 * jnp.pi / jnp.array(periods)

    z_parts, dz_parts, lap_parts = [], [], []
    for i in range(d):
        w0     = omegas[i]
        k_vec  = jnp.arange(1, n_modes + 1, dtype=jnp.float64)
        ws     = w0 * k_vec
        angles = ws * x[i]
        sv, cv = jnp.sin(angles), jnp.cos(angles)

        feats   = jnp.stack([sv, cv], axis=1).reshape(-1)
        d_feats = jnp.stack([ws * cv, -ws * sv], axis=1).reshape(-1)
        lap_f   = jnp.stack([-(ws**2) * sv, -(ws**2) * cv], axis=1).reshape(-1)

        z_parts.append(feats)
        dz_parts.append(jnp.zeros((2 * n_modes, d)).at[:, i].set(d_feats))
        lap_parts.append(lap_f)

    z   = jnp.concatenate(z_parts)
    dz  = jnp.concatenate(dz_parts, axis=0)
    lap = jnp.concatenate(lap_parts)

    for W, b in params[:-1]:
        z_pre   = jnp.dot(W, z) + b
        dz_pre  = jnp.dot(W, dz)
        lap_pre = jnp.dot(W, lap)
        s_p  = act_p(z_pre)
        s_pp = act_pp(z_pre)
        z   = act(z_pre)
        lap = s_p * lap_pre + s_pp * jnp.sum(dz_pre ** 2, axis=1)
        dz  = s_p[:, None] * dz_pre

    W_out, b_out = params[-1]
    u     = (jnp.dot(W_out, z)   + b_out)[0]
    lap_u =  jnp.dot(W_out, lap)[0]
    return u, lap_u
