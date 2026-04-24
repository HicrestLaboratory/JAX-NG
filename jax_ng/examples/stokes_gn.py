"""Stokes wedge training with multi-stage GN and pressure anchor."""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random, vmap

from jax_ng import optimizers, problems

jax.config.update("jax_enable_x64", True)


def read_reference_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        data = jnp.array([[float(v) for v in row] for row in reader])
    return data


def main():
    repo_root = Path(__file__).resolve().parents[2]
    default_data_path = repo_root / "data" / "st_flow.csv"
    default_output_dir = repo_root / "runs" / "stokes_gn"

    parser = argparse.ArgumentParser(description="Stokes wedge GN example")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--phase1_iters", type=int, default=500)
    parser.add_argument("--phase2_iters", type=int, default=1000)
    parser.add_argument("--batch_int", type=int, default=1000)
    parser.add_argument("--batch_bnd", type=int, default=1000)
    parser.add_argument("--phase1_damping", type=float, default=1e-12)
    parser.add_argument("--phase2_damping", type=float, default=5e-9)
    parser.add_argument("--phase2_tol", type=float, default=5e-19)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--data_path", type=str, default=str(default_data_path))
    parser.add_argument("--output_dir", type=str, default=str(default_output_dir))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pde = problems.StokesWedge()
    key = random.PRNGKey(args.seed)
    params = pde.init_params(width=64, depth=10, key=key)
    solver = optimizers.StokesJacobiGN(
        pde.interior_res,
        pde.boundary_res,
        pde.anchor_res,
        pde.anchor,
        ls_steps=0.5 ** jnp.linspace(0.0, 15.0, 16),
    )

    ref = read_reference_csv(args.data_path)
    X_val = U_val = V_val = P_val = None
    if ref is not None:
        X_val = ref[:, 3:5]
        U_val = ref[:, 0:1]
        V_val = ref[:, 1:2]
        P_val = ref[:, 2:3]
        print(f"Loaded validation data: {args.data_path}")
    else:
        print(f"Validation data not loaded from: {args.data_path}")

    total_iters = args.phase1_iters + args.phase2_iters
    history = []
    start = timeit.default_timer()
    for i in range(1, total_iters + 1):
        if i <= args.phase1_iters:
            damping = args.phase1_damping
            use_apex = False
        else:
            damping = args.phase2_damping
            use_apex = True

        key, sk = random.split(key)
        loss, params = solver.step(
            params,
            sk,
            damping,
            use_apex,
            args.batch_int,
            args.batch_bnd,
            pde.v1,
            pde.v2,
            pde.v3,
        )

        err_u = err_v = err_p = 0.0
        if X_val is not None and i % args.checkpoint_every == 0:
            preds = vmap(lambda x: pde.derivative_propagation(params, x)[0])(X_val)
            up, vp, pp = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
            err_u = jnp.linalg.norm(up - U_val) / (jnp.linalg.norm(U_val) + 1e-30)
            err_v = jnp.linalg.norm(vp - V_val) / (jnp.linalg.norm(V_val) + 1e-30)
            err_p = jnp.linalg.norm(pp - P_val) / (jnp.linalg.norm(P_val) + 1e-30)

        history.append([i, float(loss), float(err_u), float(err_v), float(err_p)])
        if i % args.checkpoint_every == 0:
            elapsed = timeit.default_timer() - start
            print(
                f"Iter {i:4d} | loss {loss:.4e} | ErrU {err_u:.2e} | ErrV {err_v:.2e} | ErrP {err_p:.2e} | {elapsed:.1f}s"
            )
            with open(os.path.join(args.output_dir, f"checkpoint_{i:05d}.pkl"), "wb") as f:
                pickle.dump(params, f)

        if i > args.phase1_iters and float(loss) < args.phase2_tol:
            print(f"Early stop at iter {i} (loss < {args.phase2_tol})")
            break

    with open(os.path.join(args.output_dir, "final_params.pkl"), "wb") as f:
        pickle.dump(params, f)
    with open(os.path.join(args.output_dir, "history.pkl"), "wb") as f:
        pickle.dump(history, f)
    print("Saved final params and history.")


if __name__ == "__main__":
    main()
