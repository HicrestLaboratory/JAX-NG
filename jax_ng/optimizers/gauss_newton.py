"""Gauss-Newton optimizer with Jacobi preconditioning.

Supports:
  - dual solve  (residual space, cheap when N_res < N_params)
  - primal solve (parameter space, cheap when N_params < N_res)
  - Jacobi diagonal preconditioning in both modes
  - pluggable sampler and line-search
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.flatten_util import ravel_pytree


@dataclass
class SolveConfig:
    """Controls the normal-equation solve.

    mode    : ``"dual"``   O(N_res²)    — default, cheap when residuals < params
              ``"primal"`` O(N_params²) — cheap when params < residuals
    damping : Tikhonov / LM regularisation strength
    precond : ``"jacobi"`` (default) | ``"none"``
    """
    mode:    str   = "dual"
    damping: float = 1e-8
    precond: str   = "jacobi"


class GaussNewton:
    """Dense Jacobi-preconditioned Gauss-Newton.

    Parameters
    ----------
    interior_res_fn : ``(params, x) -> scalar or vector``
    boundary_res_fn : ``(params, x) -> scalar or vector``  or ``None``
    sampler_fn      : ``(key) -> (x_int, x_bnd)``
    linesearch_fn   : ``(loss_fn, params, direction, f0) -> (alpha, loss)``
    solve_config    : :class:`SolveConfig`
    int_res_dim     : number of interior residual components per point
    bnd_res_dim     : number of boundary residual components per point
    """

    def __init__(
        self,
        interior_res_fn: Callable,
        boundary_res_fn: Optional[Callable],
        sampler_fn:      Callable,
        linesearch_fn:   Callable,
        solve_config:    SolveConfig = None,
        int_res_dim:     int = 1,
        bnd_res_dim:     int = 1,
    ):
        self.interior_res_fn = interior_res_fn
        self.boundary_res_fn = boundary_res_fn
        self.sampler_fn      = sampler_fn
        self.linesearch_fn   = linesearch_fn
        self.cfg             = solve_config or SolveConfig()
        self.int_res_dim     = int_res_dim
        self.bnd_res_dim     = bnd_res_dim

    # ── public API ────────────────────────────────────────────────────────────
    def init(self, params):
        return {}   # GN is stateless

    def step(self, params, opt_state, key):
        loss, new_params = self._step_impl(params, key)
        return loss, new_params, opt_state

    # ── JIT-compiled core ─────────────────────────────────────────────────────
    @partial(jit, static_argnums=(0,))
    def _step_impl(self, params, key):
        f_params, unravel = ravel_pytree(params)
        x_int, x_bnd = self.sampler_fn(key)

        J, r    = self._build_J(f_params, x_int, x_bnd, unravel)
        loss    = 0.5 * jnp.mean(r ** 2)
        d_flat  = self._solve(J, r)
        direction = unravel(d_flat)

        def loss_fn(p):
            ri = vmap(lambda x: self.interior_res_fn(p, x))(x_int).reshape(-1)
            rb = (vmap(lambda x: self.boundary_res_fn(p, x))(x_bnd).reshape(-1)
                  if self.boundary_res_fn is not None and x_bnd.shape[0] > 0
                  else jnp.array([]))
            return 0.5 * jnp.mean(jnp.concatenate([ri, rb]) ** 2)

        alpha, _ = self.linesearch_fn(loss_fn, params, direction, loss)
        new_params = jax.tree_util.tree_map(lambda p, d: p - alpha * d, params, direction)
        return loss, new_params

    # ── Jacobian assembly ─────────────────────────────────────────────────────
    def _build_J(self, f_params, x_int, x_bnd, unravel):
        rows_J, rows_r = [], []

        # interior
        if self.int_res_dim == 1:
            def int_row(x):
                return jax.value_and_grad(
                    lambda fp: self.interior_res_fn(unravel(fp), x))(f_params)
            r_int, J_int = vmap(int_row)(x_int)
        else:
            def int_row(x):
                def res(fp): return self.interior_res_fn(unravel(fp), x)
                return res(f_params), jax.jacobian(res)(f_params)
            r_int, J_int = vmap(int_row)(x_int)

        rows_r.append(r_int.reshape(-1))
        rows_J.append(J_int.reshape(-1, J_int.shape[-1]))

        # boundary
        if self.boundary_res_fn is not None and x_bnd.shape[0] > 0:
            if self.bnd_res_dim == 1:
                def bnd_row(x):
                    return jax.value_and_grad(
                        lambda fp: self.boundary_res_fn(unravel(fp), x))(f_params)
                r_bnd, J_bnd = vmap(bnd_row)(x_bnd)
            else:
                def bnd_row(x):
                    def res(fp): return self.boundary_res_fn(unravel(fp), x)
                    return res(f_params), jax.jacobian(res)(f_params)
                r_bnd, J_bnd = vmap(bnd_row)(x_bnd)
            rows_r.append(r_bnd.reshape(-1))
            rows_J.append(J_bnd.reshape(-1, J_bnd.shape[-1]))

        return jnp.concatenate(rows_J), jnp.concatenate(rows_r)

    # ── Linear solve (dual or primal, with optional Jacobi preconditioning) ───
    def _solve(self, J, r):
        lam = self.cfg.damping

        if self.cfg.mode == "dual":
            K = 0.5 * (J @ J.T + (J @ J.T).T)
            if self.cfg.precond == "jacobi":
                s   = 1.0 / (jnp.sqrt(jnp.diag(K)) + 1e-16)
                K_  = K * s[:, None] * s[None, :]
                r_  = r * s
                L   = jnp.linalg.cholesky(K_ + lam * jnp.eye(K_.shape[0]))
                w   = jax.scipy.linalg.cho_solve((L, True), r_) * s
            else:
                L   = jnp.linalg.cholesky(K + lam * jnp.eye(K.shape[0]))
                w   = jax.scipy.linalg.cho_solve((L, True), r)
            return J.T @ w

        else:  # primal
            A = 0.5 * (J.T @ J + (J.T @ J).T)
            b = J.T @ r
            if self.cfg.precond == "jacobi":
                s   = 1.0 / (jnp.sqrt(jnp.diag(A)) + 1e-16)
                A_  = A * s[:, None] * s[None, :]
                L   = jnp.linalg.cholesky(A_ + lam * jnp.eye(A_.shape[0]))
                return jax.scipy.linalg.cho_solve((L, True), b * s) * s
            else:
                L   = jnp.linalg.cholesky(A + lam * jnp.eye(A.shape[0]))
                return jax.scipy.linalg.cho_solve((L, True), b)
