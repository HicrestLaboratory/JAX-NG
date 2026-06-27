"""Dual Gauss-Newton PCG optimizer with Nyström preconditioners."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular


@dataclass
class PCGConfig:
    maxiter: int = 500
    ew_gamma: float = 0.9
    ew_alpha: float = 1.5
    ew_eta_min: float = 1e-4
    ew_eta_max: float = 5e-1
    true_residual: bool = False


@dataclass
class ColumnNystromConfig:
    landmark_interior: int = 200
    landmark_boundary: int = 100
    rank: int = 0
    eig_rtol: float = 1e-12


@dataclass
class SketchNystromConfig:
    sketch_size: int = 800
    nu_factor: float = 1.0
    chol_jitter: float = 1e-14


@dataclass
class DualPCGNystromConfig:
    damping: float = 1e-8
    preconditioner: str = "column_nystrom"
    pcg: PCGConfig = field(default_factory=PCGConfig)
    column: ColumnNystromConfig = field(default_factory=ColumnNystromConfig)
    sketch: SketchNystromConfig = field(default_factory=SketchNystromConfig)


def eisenstat_walker_forcing(current_norm, previous_norm, gamma, alpha, eta_min, eta_max):
    """Return an Eisenstat-Walker relative PCG tolerance."""
    tiny = jnp.array(1e-300, dtype=current_norm.dtype)
    has_prev = jnp.isfinite(previous_norm) & (previous_norm > 0.0)
    ratio = current_norm / jnp.maximum(previous_norm, tiny)
    eta = gamma * ratio ** alpha
    eta = jnp.where(has_prev, eta, eta_max)
    return jnp.clip(eta, eta_min, eta_max)


def pcg_solve_matvec(matvec, b, tol_rel, maxiter, preconditioner_fn=None, true_residual=False):
    """Solve ``A x = b`` by left-preconditioned CG using only ``matvec``."""
    apply_m_inv = preconditioner_fn if preconditioner_fn is not None else (lambda x: x)
    tiny = jnp.array(1e-300, dtype=b.dtype)

    x0 = jnp.zeros_like(b)
    r0 = b
    z0 = apply_m_inv(r0)
    p0 = z0
    rho0 = jnp.vdot(r0, z0).real
    b_norm = jnp.maximum(jnp.linalg.norm(b), tiny)
    tol_abs = tol_rel * b_norm
    r_norm0 = jnp.linalg.norm(r0)

    def cond(state):
        i, x, r, z, p, rho, r_norm, converged, breakdown = state
        del x, r, z, p, rho, r_norm
        return (i < maxiter) & (~converged) & (~breakdown)

    def body(state):
        i, x, r, z, p, rho, r_norm, converged, breakdown = state
        del z, r_norm, converged, breakdown
        Ap = matvec(p)
        pAp = jnp.vdot(p, Ap).real
        bad = pAp <= 0.0
        alpha = rho / jnp.maximum(pAp, tiny)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        r_norm_new = jnp.linalg.norm(r_new)
        z_new = apply_m_inv(r_new)
        rho_new = jnp.vdot(r_new, z_new).real
        beta = rho_new / jnp.maximum(rho, tiny)
        p_new = z_new + beta * p
        conv_new = r_norm_new <= tol_abs
        breakdown_new = bad | (rho_new < 0.0) | (~jnp.isfinite(rho_new))
        return i + 1, x_new, r_new, z_new, p_new, rho_new, r_norm_new, conv_new, breakdown_new

    init = (
        jnp.array(0),
        x0,
        r0,
        z0,
        p0,
        rho0,
        r_norm0,
        r_norm0 <= tol_abs,
        (rho0 < 0.0) | (~jnp.isfinite(rho0)),
    )
    out = lax.while_loop(cond, body, init)
    i, x, r, z, p, rho, r_norm, converged, breakdown = out
    del r, z, p, rho

    relres = r_norm / b_norm
    if true_residual:
        relres = jnp.linalg.norm(b - matvec(x)) / b_norm
    info = {"iters": i, "relres": relres, "converged": converged, "breakdown": breakdown}
    return x, info


def build_column_nystrom_preconditioner(
    J,
    mu,
    key,
    n_int,
    n_bnd,
    int_res_dim,
    bnd_res_dim,
    cfg: ColumnNystromConfig,
):
    """Build the sampled-column Nyström inverse preconditioner."""
    key_i, key_b = random.split(key)
    pieces = []

    n_landmark_int = min(int(cfg.landmark_interior), int(n_int))
    if n_landmark_int > 0:
        pts_i = random.permutation(key_i, n_int)[:n_landmark_int]
        rows_i = (int_res_dim * pts_i[:, None] + jnp.arange(int_res_dim)[None, :]).reshape(-1)
        pieces.append(rows_i)

    n_landmark_bnd = min(int(cfg.landmark_boundary), int(n_bnd))
    if n_landmark_bnd > 0:
        pts_b = random.permutation(key_b, n_bnd)[:n_landmark_bnd]
        offset = int_res_dim * int(n_int)
        rows_b = (
            offset + bnd_res_dim * pts_b[:, None] + jnp.arange(bnd_res_dim)[None, :]
        ).reshape(-1)
        pieces.append(rows_b)

    if not pieces:
        return lambda z: z

    landmark_idx = jnp.concatenate(pieces).astype(jnp.int32)
    J_I = J[landmark_idx, :]
    C = J @ J_I.T
    W = J_I @ J_I.T
    W = 0.5 * (W + W.T)

    evals_w, Q_w = jnp.linalg.eigh(W)
    order = jnp.argsort(evals_w)[::-1]
    rank = landmark_idx.shape[0] if cfg.rank <= 0 else min(int(cfg.rank), landmark_idx.shape[0])
    order = order[:rank]
    evals = evals_w[order]
    Q = Q_w[:, order]

    max_eval = jnp.maximum(jnp.max(jnp.abs(evals_w)), jnp.array(1e-300, dtype=J.dtype))
    floor = cfg.eig_rtol * max_eval
    evals_safe = jnp.maximum(evals, floor)

    B = C @ (Q / jnp.sqrt(evals_safe)[None, :])
    U, svals, _ = jnp.linalg.svd(B, full_matrices=False)
    lambdas = jnp.maximum(svals ** 2, 0.0)
    mu_safe = jnp.maximum(mu, jnp.array(1e-300, dtype=J.dtype))

    def apply(z):
        Utz = U.T @ z
        Uz = U @ Utz
        captured = U @ (Utz / (lambdas + mu_safe))
        complement = (z - Uz) / mu_safe
        return captured + complement

    return apply


def _build_sketch_nystrom_factors(J, mu, key, cfg: SketchNystromConfig):
    """Return ``B, L_R`` for the Gaussian-sketch Nyström preconditioner."""
    m = J.shape[0]
    ell = min(int(cfg.sketch_size), m)
    ell = max(ell, 1)
    dtype = J.dtype
    mu_safe = jnp.maximum(mu, jnp.array(1e-300, dtype=dtype))

    Omega = random.normal(key, (m, ell), dtype=dtype)
    Y = J @ (J.T @ Omega)
    eps = jnp.array(jnp.finfo(dtype).eps, dtype=dtype)
    nu = jnp.array(cfg.nu_factor, dtype=dtype) * eps * jnp.maximum(
        jnp.linalg.norm(Y), jnp.array(1.0, dtype=dtype)
    )
    Y_nu = Y + nu * Omega

    S = Omega.T @ Y_nu
    S = 0.5 * (S + S.T)
    avg_diag = jnp.maximum(jnp.trace(S) / ell, jnp.array(1.0, dtype=dtype))
    S_reg = S + cfg.chol_jitter * avg_diag * jnp.eye(ell, dtype=dtype)
    C = jnp.linalg.cholesky(S_reg)

    B = solve_triangular(C, Y_nu.T, lower=True).T
    R = B.T @ B + mu_safe * jnp.eye(ell, dtype=dtype)
    R = 0.5 * (R + R.T)
    avg_diag_R = jnp.maximum(jnp.trace(R) / ell, jnp.array(1.0, dtype=dtype))
    R_reg = R + cfg.chol_jitter * avg_diag_R * jnp.eye(ell, dtype=dtype)
    L_R = jnp.linalg.cholesky(R_reg)
    return B, L_R, mu_safe


def build_sketch_nystrom_preconditioner(J, mu, key, cfg: SketchNystromConfig):
    """Build the Gaussian-sketch Nyström Woodbury inverse preconditioner."""
    B, L_R, mu_safe = _build_sketch_nystrom_factors(J, mu, key, cfg)

    def apply(z):
        Bt_z = B.T @ z
        tmp = solve_triangular(L_R, Bt_z, lower=True)
        Rinv_Bt_z = solve_triangular(L_R.T, tmp, lower=False)
        return z / mu_safe - (B @ Rinv_Bt_z) / mu_safe

    return apply


def build_exact_debug_preconditioner(J, mu):
    """Build an exact dense inverse preconditioner for small debugging cases."""
    K = J @ J.T
    K = 0.5 * (K + K.T)
    dtype = J.dtype
    mu_safe = jnp.maximum(mu, jnp.array(1e-300, dtype=dtype))
    L = jnp.linalg.cholesky(K + mu_safe * jnp.eye(K.shape[0], dtype=dtype))

    def apply(z):
        tmp = solve_triangular(L, z, lower=True)
        return solve_triangular(L.T, tmp, lower=False)

    return apply


class DualPCGNystrom:
    """Dual damped Gauss-Newton solved by PCG in residual space."""

    def __init__(
        self,
        interior_res_fn: Callable,
        boundary_res_fn: Optional[Callable],
        sampler_fn: Callable,
        linesearch_fn: Callable,
        solve_config: DualPCGNystromConfig = None,
        int_res_dim: int = 1,
        bnd_res_dim: int = 1,
    ):
        self.interior_res_fn = interior_res_fn
        self.boundary_res_fn = boundary_res_fn
        self.sampler_fn = sampler_fn
        self.linesearch_fn = linesearch_fn
        self.cfg = solve_config or DualPCGNystromConfig()
        self.int_res_dim = int_res_dim
        self.bnd_res_dim = bnd_res_dim

    def init(self, params):
        del params
        return {"previous_res_norm": jnp.array(jnp.inf)}

    def step(self, params, opt_state, key):
        previous_res_norm = opt_state.get("previous_res_norm", jnp.array(jnp.inf))
        loss, new_params, new_prev_norm, info = self._step_impl(params, key, previous_res_norm)
        return loss, new_params, {"previous_res_norm": new_prev_norm, "info": info}

    @partial(jit, static_argnums=(0,))
    def _step_impl(self, params, key, previous_res_norm):
        f_params, unravel_fn = ravel_pytree(params)
        key_sample, key_prec = random.split(key)

        x_int, x_bnd = self.sampler_fn(key_sample)
        J, r = self._build_J(f_params, x_int, x_bnd, unravel_fn)

        loss = 0.5 * jnp.mean(r ** 2)
        current_res_norm = jnp.linalg.norm(r)
        mu = jnp.minimum(loss, self.cfg.damping)
        mu = jnp.maximum(mu, jnp.array(1e-300, dtype=r.dtype))

        eta = eisenstat_walker_forcing(
            current_res_norm,
            previous_res_norm,
            self.cfg.pcg.ew_gamma,
            self.cfg.pcg.ew_alpha,
            self.cfg.pcg.ew_eta_min,
            self.cfg.pcg.ew_eta_max,
        )

        matvec = lambda v: J @ (J.T @ v) + mu * v
        M_inv = self._build_preconditioner(J, mu, key_prec, x_int.shape[0], x_bnd.shape[0])
        y, pcg_info = pcg_solve_matvec(
            matvec,
            r,
            eta,
            self.cfg.pcg.maxiter,
            M_inv,
            self.cfg.pcg.true_residual,
        )

        direction_flat = J.T @ y
        direction = unravel_fn(direction_flat)

        def loss_fn(p):
            ri = vmap(lambda x: self.interior_res_fn(p, x))(x_int).reshape(-1)
            rb = (
                vmap(lambda x: self.boundary_res_fn(p, x))(x_bnd).reshape(-1)
                if self.boundary_res_fn is not None and x_bnd.shape[0] > 0
                else jnp.array([], dtype=ri.dtype)
            )
            return 0.5 * jnp.mean(jnp.concatenate([ri, rb]) ** 2)

        alpha, _ = self.linesearch_fn(loss_fn, params, direction, loss)
        new_params = jax.tree_util.tree_map(lambda p, d: p - alpha * d, params, direction)

        info = {
            "eta": eta,
            "mu": mu,
            "pcg_iters": pcg_info["iters"],
            "pcg_relres": pcg_info["relres"],
            "pcg_converged": pcg_info["converged"],
            "pcg_breakdown": pcg_info["breakdown"],
        }
        return loss, new_params, current_res_norm, info

    def _build_J(self, f_params, x_int, x_bnd, unravel):
        rows_J, rows_r = [], []

        if self.int_res_dim == 1:
            def int_row(x):
                return jax.value_and_grad(lambda fp: self.interior_res_fn(unravel(fp), x))(f_params)

            r_int, J_int = vmap(int_row)(x_int)
        else:
            def int_row(x):
                def res(fp):
                    return self.interior_res_fn(unravel(fp), x)

                return res(f_params), jax.jacobian(res)(f_params)

            r_int, J_int = vmap(int_row)(x_int)

        rows_r.append(r_int.reshape(-1))
        rows_J.append(J_int.reshape(-1, J_int.shape[-1]))

        if self.boundary_res_fn is not None and x_bnd.shape[0] > 0:
            if self.bnd_res_dim == 1:
                def bnd_row(x):
                    return jax.value_and_grad(lambda fp: self.boundary_res_fn(unravel(fp), x))(f_params)

                r_bnd, J_bnd = vmap(bnd_row)(x_bnd)
            else:
                def bnd_row(x):
                    def res(fp):
                        return self.boundary_res_fn(unravel(fp), x)

                    return res(f_params), jax.jacobian(res)(f_params)

                r_bnd, J_bnd = vmap(bnd_row)(x_bnd)

            rows_r.append(r_bnd.reshape(-1))
            rows_J.append(J_bnd.reshape(-1, J_bnd.shape[-1]))

        return jnp.concatenate(rows_J), jnp.concatenate(rows_r)

    def _build_preconditioner(self, J, mu, key, n_int, n_bnd):
        mode = self.cfg.preconditioner
        if mode == "none":
            return lambda z: z
        if mode == "column_nystrom":
            return build_column_nystrom_preconditioner(
                J,
                mu,
                key,
                n_int,
                n_bnd,
                self.int_res_dim,
                self.bnd_res_dim,
                self.cfg.column,
            )
        if mode == "sketch_nystrom":
            return build_sketch_nystrom_preconditioner(J, mu, key, self.cfg.sketch)
        if mode == "exact":
            return build_exact_debug_preconditioner(J, mu)
        raise ValueError(
            "DualPCGNystromConfig.preconditioner must be one of "
            "'none', 'column_nystrom', 'sketch_nystrom', or 'exact'"
        )
