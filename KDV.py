#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windowed KS Solver approach applied to KdV.

PDE (periodic in x on [-1, 1]):
  u_t + ETA * u * u_x + MU2 * u_xxx = 0

Key design choices (ported from KS1d.py):
  - Hard Initial Condition Constraint via Ansatz
  - Fourier Embedding for periodicity
  - Dual / Gauss-Newton optimizer with line search (JacobiGNSolver)
  - Windowed time marching: each window stores params for the hard-IC ansatz
  - JET-based automatic differentiation for PDE residuals
"""

import os
import sys
import timeit
import argparse
import numpy as np
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from jax.flatten_util import ravel_pytree
from functools import partial
from jax.experimental.jet import jet

jax.config.update("jax_enable_x64", True)

from jax_ng import utils


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

class Config:
    # Domain: periodic on [-1, 1]  (period L = 2)
    X_MIN   = -1.0
    X_MAX   =  1.0
    L       =  2.0          # period length

    # KdV parameters
    ETA  = 1.0
    MU2  = 0.022 ** 2       # coefficient of u_xxx

    # Windowing
    DT_WINDOW  = 0.2       # duration of one time window
    N_WINDOWS  = 5        # total windows  →  T_final = 1.0
    TOTAL_T    = DT_WINDOW * N_WINDOWS

    # Architecture
    NUM_MODES = 10            # Fourier embedding modes
    WIDTH     = 30
    DEPTH     = 4

    # Training
    N_COLLOCATION = 30000
    ITERATIONS    = 2000
    PRINT_EVERY   = 500
    DAMPING       = 1e-12
    SEED          = 42

    # Validation data
    DATA_PATH = "kdv.mat"


cfg = Config()


# ==============================================================================
# 2. INITIAL CONDITION
# ==============================================================================

def initial_condition_analytic(x):
    """u0(x) = cos(pi * x)  — standard KdV benchmark IC."""
    return jnp.cos(jnp.pi * x)


# ==============================================================================
# 3. NETWORK  (same architecture as KS1d)
# ==============================================================================

def init_params(layers, key):
    keys = random.split(key, len(layers))
    params = []
    for m, n, k in zip(layers[:-1], layers[1:], keys):
        w_k, b_k = random.split(k)
        scale = jnp.sqrt(2.0 / (m + n))
        W = random.normal(w_k, (n, m)) * scale
        b = jnp.zeros((n,))
        params.append((W, b))
    return params


def periodic_embedding(x, L, num_modes):
    """Fourier features for a function with period L."""
    omega = 2.0 * jnp.pi / L
    feats = []
    for k in range(1, num_modes + 1):
        feats.append(jnp.sin(k * omega * x))
        feats.append(jnp.cos(k * omega * x))
    return jnp.concatenate([jnp.atleast_1d(f) for f in feats])


def base_network(params, t, x):
    """
    Network phi(t, x).
    t is normalised to [-1, 1] within the window [0, DT_WINDOW].
    x is embedded with Fourier features.
    """
    t_norm = t * (2.0 / cfg.DT_WINDOW) - 1.0
    x_emb  = periodic_embedding(x, cfg.L, cfg.NUM_MODES)
    h = jnp.concatenate([jnp.atleast_1d(t_norm), x_emb])
    for W, b in params[:-1]:
        h = jnp.sin(jnp.dot(W, h) + b)
    W_final, b_final = params[-1]
    return (jnp.dot(W_final, h) + b_final)[0]


# ==============================================================================
# 4. HARD ANSATZ  (exact IC enforcement via stacked past parameters)
# ==============================================================================

def hard_ansatz(params, p_past, w_idx, t, x):
    """
    u(t, x) = u_ic(x)
              + sum_{j < w_idx} [phi_j(DT, x) - phi_j(0, x)]    (past windows)
              + phi_current(t, x) - phi_current(0, x)            (this window)

    By construction u(0, x) = u_ic(x) for the first window, and
    u(DT, x) of window j is used as IC of window j+1 automatically.
    """
    phi_t = base_network(params, t, x)
    phi_0 = base_network(params, 0.0, x)
    u_ic  = initial_condition_analytic(x)

    def eval_past_delta(p):
        return base_network(p, cfg.DT_WINDOW, x) - base_network(p, 0.0, x)

    deltas = jax.vmap(eval_past_delta)(p_past)
    mask   = (jnp.arange(cfg.N_WINDOWS) < w_idx).astype(jnp.float64)
    u_ic   = u_ic + jnp.sum(mask * deltas)

    return phi_t - phi_0 + u_ic


# ==============================================================================
# 5. PHYSICS  (JET-based derivatives, KdV residual)
# ==============================================================================

def derivs_jet(params, p_past, w_idx, t, x):
    """Compute u, u_t, u_x, u_xxx via jet (forward-mode Taylor)."""
    # --- spatial derivatives ---
    g_x = lambda xx: hard_ansatz(params, p_past, w_idx, t, xx)
    u_val, (ux, uxx, uxxx) = jet(g_x, (x,), ((1.0, 0.0, 0.0),))

    # --- time derivative ---
    g_t = lambda tt: hard_ansatz(params, p_past, w_idx, tt, x)
    _, (ut,) = jet(g_t, (t,), ((1.0,),))

    return u_val, ut, ux, uxx, uxxx


def pde_residual(params, p_past, w_idx, t, x):
    """
    KdV residual:  u_t + ETA * u * u_x + MU2 * u_xxx = 0
    """
    u, ut, ux, _uxx, uxxx = derivs_jet(params, p_past, w_idx, t, x)
    return ut + cfg.ETA * u * ux + cfg.MU2 * uxxx


# ==============================================================================
# 6. OPTIMIZER  (JacobiGNSolver — identical structure to KS1d)
# ==============================================================================

class JacobiGNSolver:
    def __init__(self, pde_res_fn, ls_steps):
        self.pde_res_fn = pde_res_fn
        self.ls_steps   = ls_steps

    def build_jacobian(self, f_params, t_col, x_col, unravel_fn, p_past, w_idx):
        def get_pde_row(t, x):
            def scalar(p):
                return self.pde_res_fn(unravel_fn(p), p_past, w_idx, t, x)
            val, grad = jax.value_and_grad(scalar)(f_params)
            return val, grad

        r, J = vmap(get_pde_row)(t_col, x_col)
        return J, r

    @partial(jit, static_argnums=(0,))
    def step(self, params, t_col, x_col, damping, p_past, w_idx):
        f_params, unravel_fn = ravel_pytree(params)
        J, r = self.build_jacobian(f_params, t_col, x_col, unravel_fn, p_past, w_idx)
        loss = 0.5 * jnp.mean(r ** 2)

        n_residuals, n_params = J.shape  # J is (N_col, N_params)

        if n_params > n_residuals:
            # ----------------------------------------------------------
            # DUAL solve  (Gram matrix, size N_col × N_col)
            # Preferred when parameter space is larger than residual space
            # ----------------------------------------------------------
            K = jnp.dot(J, J.T, precision=jax.lax.Precision.HIGHEST)
            K = 0.5 * (K + K.T)

            # Diagonal pre-conditioning
            diag    = jnp.diag(K)
            scale   = 1.0 / (jnp.sqrt(diag) + 1e-16)
            K_tilde = K * scale[:, None] * scale[None, :]
            r_tilde = r * scale

            # Regularised Cholesky solve
            K_reg  = K_tilde + damping * jnp.eye(K_tilde.shape[0])
            L      = jnp.linalg.cholesky(K_reg)
            y      = jax.scipy.linalg.cho_solve((L, True), r_tilde)

            w_dual = y * scale
            delta  = J.T @ w_dual        # parameter update direction

        else:
            # ----------------------------------------------------------
            # PRIMAL solve  (Gauss-Newton Hessian, size N_params × N_params)
            # Preferred when residual space is larger than parameter space
            # ----------------------------------------------------------
            H = jnp.dot(J.T, J, precision=jax.lax.Precision.HIGHEST)
            H = 0.5 * (H + H.T)

            # Diagonal pre-conditioning
            diag    = jnp.diag(H)
            scale   = 1.0 / (jnp.sqrt(diag) + 1e-16)
            H_tilde = H * scale[:, None] * scale[None, :]
            g_tilde = (J.T @ r) * scale  # scaled gradient

            # Regularised Cholesky solve
            H_reg  = H_tilde + damping * jnp.eye(H_tilde.shape[0])
            L      = jnp.linalg.cholesky(H_reg)
            y      = jax.scipy.linalg.cho_solve((L, True), g_tilde)

            delta  = y * scale           # parameter update direction

        # Armijo-style line search
        def evaluate(p_flat):
            p  = unravel_fn(p_flat)
            rp = vmap(lambda t, x: self.pde_res_fn(p, p_past, w_idx, t, x))(t_col, x_col)
            return 0.5 * jnp.mean(rp ** 2)

        def check(lr):
            return evaluate(f_params - lr * delta)

        lrs       = self.ls_steps
        losses    = vmap(check)(lrs)
        best_idx  = jnp.argmin(losses)
        new_params = unravel_fn(f_params - lrs[best_idx] * delta)
        return new_params, loss


# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="KdV — Gauss-Newton Windowed PINN")
    parser.add_argument("--seed",       type=int, default=cfg.SEED,       help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results_kdv_gn", help="Output directory")
    parser.add_argument("--data_path",  type=str, default=cfg.DATA_PATH,  help="Path to kdv.mat")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load reference data
    # ------------------------------------------------------------------
    print(f"--- Loading reference data from {args.data_path} ---")
    try:
        data   = scipy.io.loadmat(args.data_path)
        t_full = data["t"].flatten()
        x_full = data["x"].flatten()
        u_full = data["usol"]          # shape: (N_t, N_x)  [time-major]
    except FileNotFoundError:
        print(f"File '{args.data_path}' not found — validation will be skipped.")
        t_full = x_full = u_full = None

    # ------------------------------------------------------------------
    # Initialise network for window 0
    # ------------------------------------------------------------------
    key         = random.PRNGKey(args.seed)
    layer_sizes = [1 + 2 * cfg.NUM_MODES] + [cfg.WIDTH] * cfg.DEPTH + [1]
    params      = init_params(layer_sizes, key)

    # Pre-allocate stacked past-parameter array (zeros, one slot per window)
    stacked_past_params = jax.tree_util.tree_map(
        lambda leaf: jnp.zeros((cfg.N_WINDOWS,) + leaf.shape),
        params,
    )

    solver = JacobiGNSolver(pde_residual, ls_steps=0.5 ** jnp.arange(10))

    print(f"\n--- Starting Hard-Constrained Windowed GN Training (KdV) ---")
    print(f"Windows: {cfg.N_WINDOWS} | DT_window: {cfg.DT_WINDOW} | "
          f"Iters/Win: {cfg.ITERATIONS} | Damping: {cfg.DAMPING}")
    print(f"ETA={cfg.ETA}  MU2={cfg.MU2:.6f}  Domain=[{cfg.X_MIN},{cfg.X_MAX}]")

    stitched_preds  = []
    stitched_times  = []
    stitched_truth  = []
    history         = []

    @jit
    def predict_window(p, p_past, w_idx, t_grid, x_grid):
        """Evaluate the hard ansatz on a 2-D meshgrid (t_grid, x_grid)."""
        return vmap(vmap(lambda t, x: hard_ansatz(p, p_past, w_idx, t, x)))(t_grid, x_grid)

    # ------------------------------------------------------------------
    # Warm-up / compile (window 0 dry run)
    # ------------------------------------------------------------------
    print("Compiling solver ...", end="", flush=True)
    key, k1, k2 = random.split(key, 3)
    _t = random.uniform(k1, (cfg.N_COLLOCATION,),
                        minval=0.0, maxval=cfg.DT_WINDOW)
    _x = random.uniform(k2, (cfg.N_COLLOCATION,),
                        minval=cfg.X_MIN, maxval=cfg.X_MAX)
    _ = solver.step(params, _t, _x, cfg.DAMPING,
                    stacked_past_params, jnp.array(0, dtype=jnp.int32))
    print(" Done.")

    # ------------------------------------------------------------------
    # Window loop
    # ------------------------------------------------------------------
    for w in range(cfg.N_WINDOWS):
        win_t_start = w * cfg.DT_WINDOW
        win_t_end   = (w + 1) * cfg.DT_WINDOW
        print(f"\n>>> Window {w}: t ∈ [{win_t_start:.4f}, {win_t_end:.4f}]")

        w_idx_array = jnp.array(w, dtype=jnp.int32)

        # ---- Slice reference data for this window --------------------
        if u_full is not None:
            eps  = 1e-8
            mask = (t_full >= win_t_start - eps) & (t_full <= win_t_end + eps)
            t_val_global = t_full[mask]
            u_val_slice  = u_full[mask, :]                 # (Nt_win, Nx)
            t_val_local  = t_val_global - win_t_start
            T_mesh, X_mesh = jnp.meshgrid(t_val_local, x_full, indexing="ij")
        else:
            t_val_global = u_val_slice = t_val_local = T_mesh = X_mesh = None

        start_time = timeit.default_timer()

        # ---- Training loop ------------------------------------------
        for i in range(1, cfg.ITERATIONS + 1):
            key, k1, k2 = random.split(key, 3)
            t_col = random.uniform(k1, (cfg.N_COLLOCATION,),
                                   minval=0.0, maxval=cfg.DT_WINDOW)
            x_col = random.uniform(k2, (cfg.N_COLLOCATION,),
                                   minval=cfg.X_MIN, maxval=cfg.X_MAX)

            params, loss = solver.step(
                params, t_col, x_col, cfg.DAMPING,
                stacked_past_params, w_idx_array,
            )

            if i % cfg.PRINT_EVERY == 0:
                elapsed = timeit.default_timer() - start_time
                history.append({"iter": w * cfg.ITERATIONS + i, "loss": float(loss)})
                if T_mesh is not None:
                    u_pred_val = predict_window(
                        params, stacked_past_params, w_idx_array, T_mesh, X_mesh
                    )
                    rel_err = (jnp.linalg.norm(u_pred_val - u_val_slice)
                               / (jnp.linalg.norm(u_val_slice) + 1e-30))
                    print(f"  Iter {i:5d} | PDE Loss: {loss:.4e} | "
                          f"Rel L2: {rel_err:.6e} | Time: {elapsed:.2f}s")
                else:
                    print(f"  Iter {i:5d} | PDE Loss: {loss:.4e} | Time: {elapsed:.2f}s")

        # ---- Store predictions for stitching -------------------------
        if T_mesh is not None:
            u_final = predict_window(
                params, stacked_past_params, w_idx_array, T_mesh, X_mesh
            )
            stitched_preds.append(u_final)
            stitched_truth.append(u_val_slice)
            stitched_times.append(t_val_global)

        # ---- Push current params into the stacked past array ---------
        stacked_past_params = jax.tree_util.tree_map(
            lambda stacked, current: stacked.at[w].set(current),
            stacked_past_params, params,
        )

        # jax_ng: checkpoint after each window
        utils.save_checkpoint(params, os.path.join(args.output_dir, f"params_window{w}.pkl"))

    # ------------------------------------------------------------------
    # 8. POST-PROCESSING & VISUALISATION
    # ------------------------------------------------------------------
    if stitched_preds:
        print("\n--- Stitching and plotting ---")
        t_raw    = np.concatenate(stitched_times)
        u_raw    = np.concatenate(stitched_preds,  axis=0)
        u_tr_raw = np.concatenate(stitched_truth, axis=0)

        t_plot, uid = np.unique(t_raw, return_index=True)
        u_plot      = np.array(u_raw[uid])
        u_truth     = np.array(u_tr_raw[uid])

        diff_map     = np.abs(u_plot - u_truth)
        total_rel_l2 = (np.linalg.norm(diff_map) /
                        (np.linalg.norm(u_truth) + 1e-30))
        print(f"Total Stitched Rel L2 Error: {total_rel_l2:.6e}")

        x_ref = x_full if x_full is not None else np.linspace(cfg.X_MIN, cfg.X_MAX, u_plot.shape[1])

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        im0 = axes[0].pcolormesh(t_plot, x_ref, u_truth.T, cmap="viridis", shading="auto")
        axes[0].set_title("Ground Truth (MATLAB)")
        axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].pcolormesh(t_plot, x_ref, u_plot.T, cmap="viridis", shading="auto")
        axes[1].set_title(f"GN Prediction\nGlobal Rel L2: {total_rel_l2:.4e}")
        axes[1].set_xlabel("t")
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].pcolormesh(t_plot, x_ref, diff_map.T, cmap="inferno", shading="auto")
        axes[2].set_title("Absolute Error")
        axes[2].set_xlabel("t")
        plt.colorbar(im2, ax=axes[2])

        # Mark window boundaries
        for w_line in range(1, cfg.N_WINDOWS):
            axes[2].axvline(w_line * cfg.DT_WINDOW, color="white",
                            linestyle="--", alpha=0.2, linewidth=0.5)

        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, "kdv_gn_window_result.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Saved figure → {fig_path}")
        plt.close(fig)

        # Save arrays
        npz_path = os.path.join(args.output_dir, f"kdv_gn_seed_{args.seed}.npz")
        np.savez(npz_path,
                 t_plot=t_plot,
                 x_ref=x_ref,
                 u_pred=u_plot,
                 u_truth=u_truth,
                 global_rel_l2=total_rel_l2)
        print(f"Saved results  → {npz_path}")

        # jax_ng: save loss history and plot
        utils.save_history(history, os.path.join(args.output_dir, "history.pkl"))
        utils.plot_history(history, save_path=os.path.join(args.output_dir, "loss_curve.png"))

    else:
        print("No stitched predictions to plot (no reference data loaded).")

    print("\n--- Done. ---")


if __name__ == "__main__":
    main()
