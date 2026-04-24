"""Stokes wedge-flow problem with pressure anchor."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import random


class StokesWedge:
    """Steady Stokes in a wedge. Network output is [u, v, p]."""

    def __init__(self, v1=None, v2=None, v3=None, anchor=None):
        self.v1 = jnp.array([0.5, 0.0]) if v1 is None else jnp.array(v1)
        self.v2 = jnp.array([0.0, 2.0]) if v2 is None else jnp.array(v2)
        self.v3 = jnp.array([1.0, 2.0]) if v3 is None else jnp.array(v3)
        self.anchor = jnp.array([0.0, 2.0]) if anchor is None else jnp.array(anchor)

    @staticmethod
    def _act(t):
        return jnp.tanh(t)

    @staticmethod
    def _act_p(t):
        th = jnp.tanh(t)
        return 1.0 - th * th

    @staticmethod
    def _act_pp(t):
        th = jnp.tanh(t)
        return -2.0 * th * (1.0 - th * th)

    def layer_sizes(self, width=64, depth=10):
        return [2] + [width] * depth + [3]

    def init_params(self, width, depth, key):
        sizes = self.layer_sizes(width=width, depth=depth)
        keys = random.split(key, len(sizes))
        params = []
        for m, n, k in zip(sizes[:-1], sizes[1:], keys):
            W = random.normal(k, (n, m)) * jnp.sqrt(2.0 / (m + n))
            b = jnp.zeros((n,))
            params.append((W, b))
        return params

    @partial(jax.jit, static_argnums=(0,))
    def derivative_propagation(self, params, x):
        z = x
        dz_dx = jnp.eye(x.shape[0])
        d2z_dxx = jnp.zeros((x.shape[0], x.shape[0], x.shape[0]))

        for W, b in params[:-1]:
            z_pre = W @ z + b
            dz_pre = W @ dz_dx
            d2z_pre = jnp.einsum("ij,jkl->ikl", W, d2z_dxx)
            z = self._act(z_pre)
            s_p = self._act_p(z_pre)
            s_pp = self._act_pp(z_pre)
            dz_dx = s_p[:, None] * dz_pre
            d2z_dxx = s_pp[:, None, None] * jnp.einsum("ij,ik->ijk", dz_pre, dz_pre) + s_p[
                :, None, None
            ] * d2z_pre

        Wf, bf = params[-1]
        z = Wf @ z + bf
        dz_dx = Wf @ dz_dx
        d2z_dxx = jnp.einsum("ij,jkl->ikl", Wf, d2z_dxx)
        return z, dz_dx, d2z_dxx

    @partial(jax.jit, static_argnums=(0,))
    def interior_res(self, params, x):
        _, jac, hess = self.derivative_propagation(params, x)
        u_x, v_y = jac[0, 0], jac[1, 1]
        p_x, p_y = jac[2, 0], jac[2, 1]
        lap_u = hess[0, 0, 0] + hess[0, 1, 1]
        lap_v = hess[1, 0, 0] + hess[1, 1, 1]
        return jnp.stack([p_x - lap_u, p_y - lap_v, u_x + v_y])

    @partial(jax.jit, static_argnums=(0,))
    def boundary_res(self, params, x):
        pred, _, _ = self.derivative_propagation(params, x)
        u_pred, v_pred = pred[0], pred[1]
        is_top = x[1] > 1.999
        u_lid = 16.0 * (x[0] ** 2) * ((1.0 - x[0]) ** 2)
        u_target = jnp.where(is_top, u_lid, 0.0)
        return jnp.stack([u_pred - u_target, v_pred])

    @partial(jax.jit, static_argnums=(0,))
    def anchor_res(self, params, x):
        pred, _, _ = self.derivative_propagation(params, x)
        return jnp.array([pred[2]])
