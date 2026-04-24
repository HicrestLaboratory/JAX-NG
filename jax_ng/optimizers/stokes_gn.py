"""Gauss-Newton solver with anchor residual support for Stokes wedge flow."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, random, vmap
from jax.flatten_util import ravel_pytree

from jax_ng.samplers.triangle import apex_heavy, wedge


class StokesJacobiGN:
    """Jacobi-preconditioned dual GN solver for Stokes (interior + boundary + pressure anchor)."""

    def __init__(self, interior_res_fn, boundary_res_fn, anchor_res_fn, anchor_point, ls_steps=None):
        self.interior_res_fn = interior_res_fn
        self.boundary_res_fn = boundary_res_fn
        self.anchor_res_fn = anchor_res_fn
        self.anchor_point = anchor_point
        self.ls_steps = ls_steps if ls_steps is not None else 0.5 ** jnp.linspace(0.0, 15.0, 16)

    def _build_J(self, f_params, x_int, x_bnd, unravel_fn):
        def int_row(x):
            def res(fp):
                return self.interior_res_fn(unravel_fn(fp), x)

            return res(f_params), jax.jacobian(res)(f_params)

        def bnd_row(x):
            def res(fp):
                return self.boundary_res_fn(unravel_fn(fp), x)

            return res(f_params), jax.jacobian(res)(f_params)

        def anc_row(x):
            def res(fp):
                return self.anchor_res_fn(unravel_fn(fp), x)

            return res(f_params), jax.jacobian(res)(f_params)

        r_i, J_i = vmap(int_row)(x_int)
        r_b, J_b = vmap(bnd_row)(x_bnd)
        r_a, J_a = anc_row(self.anchor_point)

        J_i_flat = J_i.reshape(-1, J_i.shape[-1])
        J_b_flat = J_b.reshape(-1, J_b.shape[-1])
        J_a_flat = J_a.reshape(-1, J_a.shape[-1])

        r_i_flat = r_i.reshape(-1)
        r_b_flat = r_b.reshape(-1)
        r_a_flat = r_a.reshape(-1)

        n_total = r_i_flat.shape[0] + r_b_flat.shape[0] + 1
        w_anchor = jnp.sqrt(n_total)
        J_a_flat = J_a_flat * w_anchor
        r_a_flat = r_a_flat * w_anchor

        J = jnp.concatenate([J_i_flat, J_b_flat, J_a_flat], axis=0)
        r = jnp.concatenate([r_i_flat, r_b_flat, r_a_flat], axis=0)
        return J, r

    @partial(jit, static_argnums=(0,))
    def _step_impl(self, params, x_int, x_bnd, damping):
        f_params, unravel_fn = ravel_pytree(params)
        J, r = self._build_J(f_params, x_int, x_bnd, unravel_fn)
        loss = 0.5 * jnp.mean(r ** 2)

        K = J @ J.T
        K = 0.5 * (K + K.T)
        scale = 1.0 / (jnp.sqrt(jnp.diag(K)) + 1e-16)
        K_tilde = K * scale[:, None] * scale[None, :]
        r_tilde = r * scale
        K_reg = K_tilde + damping * jnp.eye(K_tilde.shape[0])
        L = jnp.linalg.cholesky(K_reg)
        y = jax.scipy.linalg.cho_solve((L, True), r_tilde)
        w_dual = y * scale
        direction = J.T @ w_dual

        def evaluate(p_flat):
            p = unravel_fn(p_flat)
            ri = vmap(lambda x: self.interior_res_fn(p, x))(x_int).reshape(-1)
            rb = vmap(lambda x: self.boundary_res_fn(p, x))(x_bnd).reshape(-1)
            ra = self.anchor_res_fn(p, self.anchor_point).reshape(-1)
            n_total = ri.shape[0] + rb.shape[0] + 1
            ra = ra * jnp.sqrt(n_total)
            return 0.5 * jnp.mean(jnp.concatenate([ri, rb, ra]) ** 2)

        losses = vmap(lambda s: evaluate(f_params - s * direction))(self.ls_steps)
        best_idx = jnp.argmin(losses)
        new_params = unravel_fn(f_params - self.ls_steps[best_idx] * direction)
        return losses[best_idx], new_params

    def step(self, params, key, damping, use_apex_heavy, n_int, n_bnd, v1, v2, v3):
        k1, k2 = random.split(key)
        if use_apex_heavy:
            x_int = apex_heavy(k1, n_int, v1, v2, v3)
            _, x_bnd = wedge(k2, n_int, n_bnd, v1, v2, v3)
        else:
            x_int, x_bnd = wedge(k2, n_int, n_bnd, v1, v2, v3)
        return self._step_impl(params, x_int, x_bnd, damping)
