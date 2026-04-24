"""Windowed Gauss-Newton solver for residuals of the form r(params, p_past, w_idx, t, x)."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.flatten_util import ravel_pytree


class WindowedJacobiGN:
    """Jacobi-preconditioned GN with geometric-grid line search for time-windowed PINNs."""

    def __init__(self, pde_res_fn, ls_steps=None):
        self.pde_res_fn = pde_res_fn
        self.ls_steps = ls_steps if ls_steps is not None else 0.5 ** jnp.arange(10)

    def build_jacobian(self, f_params, t_col, x_col, unravel_fn, p_past, w_idx):
        def get_row(t, x):
            def scalar(fp):
                return self.pde_res_fn(unravel_fn(fp), p_past, w_idx, t, x)

            val, grad = jax.value_and_grad(scalar)(f_params)
            return val, grad

        r, J = vmap(get_row)(t_col, x_col)
        return J, r

    @partial(jit, static_argnums=(0,))
    def step(self, params, t_col, x_col, damping, p_past, w_idx):
        f_params, unravel_fn = ravel_pytree(params)
        J, r = self.build_jacobian(f_params, t_col, x_col, unravel_fn, p_past, w_idx)
        loss = 0.5 * jnp.mean(r ** 2)

        n_residuals, n_params = J.shape
        if n_params > n_residuals:
            K = J @ J.T
            K = 0.5 * (K + K.T)
            scale = 1.0 / (jnp.sqrt(jnp.diag(K)) + 1e-16)
            K_tilde = K * scale[:, None] * scale[None, :]
            r_tilde = r * scale
            K_reg = K_tilde + damping * jnp.eye(K_tilde.shape[0])
            L = jnp.linalg.cholesky(K_reg)
            y = jax.scipy.linalg.cho_solve((L, True), r_tilde)
            delta = J.T @ (y * scale)
        else:
            H = J.T @ J
            H = 0.5 * (H + H.T)
            scale = 1.0 / (jnp.sqrt(jnp.diag(H)) + 1e-16)
            H_tilde = H * scale[:, None] * scale[None, :]
            g_tilde = (J.T @ r) * scale
            H_reg = H_tilde + damping * jnp.eye(H_tilde.shape[0])
            L = jnp.linalg.cholesky(H_reg)
            y = jax.scipy.linalg.cho_solve((L, True), g_tilde)
            delta = y * scale

        def evaluate(p_flat):
            p = unravel_fn(p_flat)
            rp = vmap(lambda t, x: self.pde_res_fn(p, p_past, w_idx, t, x))(t_col, x_col)
            return 0.5 * jnp.mean(rp ** 2)

        losses = vmap(lambda lr: evaluate(f_params - lr * delta))(self.ls_steps)
        best_idx = jnp.argmin(losses)
        new_params = unravel_fn(f_params - self.ls_steps[best_idx] * delta)
        return new_params, loss
