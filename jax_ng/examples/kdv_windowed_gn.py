"""KdV windowed training with hard-IC ansatz and Jacobi GN."""
import argparse
import os
import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random, vmap

from jax_ng import optimizers, problems

jax.config.update("jax_enable_x64", True)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    default_data_path = repo_root / "data" / "kdv.mat"
    default_output_dir = repo_root / "runs" / "kdv_gn"

    parser = argparse.ArgumentParser(description="KdV windowed GN example")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--n_collocation", type=int, default=30000)
    parser.add_argument("--print_every", type=int, default=500)
    parser.add_argument("--data_path", type=str, default=str(default_data_path))
    parser.add_argument("--output_dir", type=str, default=str(default_output_dir))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pde = problems.KdVWindowed(dt_window=0.2, n_windows=5, width=30, depth=4)

    key = random.PRNGKey(args.seed)
    params = pde.init_params(key)
    stacked_past = pde.init_stacked_params(params)
    solver = optimizers.WindowedJacobiGN(pde.pde_residual, ls_steps=0.5 ** jnp.arange(10))

    t_full = x_full = u_full = None
    try:
        import scipy.io

        data = scipy.io.loadmat(args.data_path)
        t_full = data["t"].flatten()
        x_full = data["x"].flatten()
        u_full = data["usol"]
        print(f"Loaded validation data: {args.data_path}")
    except Exception:
        print(f"Validation data not loaded from: {args.data_path}")

    @jax.jit
    def predict_window(p, p_past, w_idx, t_grid, x_grid):
        return vmap(vmap(lambda t, x: pde.hard_ansatz(p, p_past, w_idx, t, x)))(t_grid, x_grid)

    for w in range(pde.n_windows):
        win_t_start = w * pde.dt_window
        win_t_end = (w + 1) * pde.dt_window
        print(f"\nWindow {w}: [{win_t_start:.4f}, {win_t_end:.4f}]")
        w_idx = jnp.array(w, dtype=jnp.int32)

        t_val_local = u_val_slice = T_mesh = X_mesh = None
        if u_full is not None:
            eps = 1e-8
            mask = (t_full >= win_t_start - eps) & (t_full <= win_t_end + eps)
            t_val_global = t_full[mask]
            u_val_slice = u_full[mask, :]
            t_val_local = t_val_global - win_t_start
            T_mesh, X_mesh = jnp.meshgrid(t_val_local, x_full, indexing="ij")

        start = timeit.default_timer()
        for i in range(1, args.iterations + 1):
            key, k1, k2 = random.split(key, 3)
            t_col = random.uniform(k1, (args.n_collocation,), minval=0.0, maxval=pde.dt_window)
            x_col = random.uniform(k2, (args.n_collocation,), minval=pde.x_min, maxval=pde.x_max)
            params, loss = solver.step(params, t_col, x_col, 1e-8, stacked_past, w_idx)

            if i % args.print_every == 0:
                elapsed = timeit.default_timer() - start
                if T_mesh is not None:
                    u_pred = predict_window(params, stacked_past, w_idx, T_mesh, X_mesh)
                    rel = jnp.linalg.norm(u_pred - u_val_slice) / (jnp.linalg.norm(u_val_slice) + 1e-30)
                    print(f"Iter {i:5d} | PDE loss {loss:.4e} | Rel L2 {rel:.6e} | {elapsed:.1f}s")
                else:
                    print(f"Iter {i:5d} | PDE loss {loss:.4e} | {elapsed:.1f}s")

        stacked_past = jax.tree_util.tree_map(lambda stk, cur: stk.at[w].set(cur), stacked_past, params)

    out_path = os.path.join(args.output_dir, "final_params_windowed.pkl")
    import pickle

    with open(out_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
