"""Beltrami 3D solver with Gauss-Newton optimizer."""
from __future__ import annotations

import argparse
import os
import pickle
import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random, vmap

from jax_ng import linesearch, optimizers, problems, samplers

jax.config.update("jax_enable_x64", True)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Beltrami solver with Gauss-Newton optimizer")
    parser.add_argument("--LM", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--n_interior", type=int, default=1000)
    parser.add_argument("--n_boundary", type=int, default=500)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--re", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--time_limit", type=float, default=3000.0)
    parser.add_argument("--output_dir", type=str, default=str(repo_root / "runs" / "beltrami_gn"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    box = ((0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    pde = problems.Beltrami3D(Re=args.re)
    params = pde.init_params(width=args.width, depth=args.depth, key=random.PRNGKey(args.seed))
    sampler = lambda key: samplers.uniform_box(key, args.n_interior, args.n_boundary, box)
    ls = linesearch.build("grid_search", base=0.5, n_steps=16, start_exp=0.0)

    opt = optimizers.GaussNewton(
        interior_res_fn=pde.interior_res,
        boundary_res_fn=pde.boundary_res,
        sampler_fn=sampler,
        linesearch_fn=ls,
        solve_config=optimizers.SolveConfig(mode="auto", damping=args.LM, precond="jacobi"),
        int_res_dim=4,
        bnd_res_dim=3,
    )

    key = random.PRNGKey(args.seed)
    key, eval_key = random.split(key)
    x_eval, _ = sampler(eval_key)
    u_star_eval = vmap(pde.exact_velocity)(x_eval)
    true_norm = jnp.linalg.norm(u_star_eval)

    history = {"iterations": [], "relative_l2_errors": [], "losses": [], "times": []}
    state = opt.init(params)

    print("--- Starting Beltrami GN ---")
    start = timeit.default_timer()
    for iteration in range(args.max_iters):
        elapsed_total = timeit.default_timer() - start
        if elapsed_total > args.time_limit:
            print("Time budget exceeded, stopping.")
            break

        key, sk = random.split(key)
        loss, params, state = opt.step(params, state, sk)

        if iteration % args.log_every == 0:
            pred = vmap(lambda x: pde.boundary_res(params, x) + pde.exact_velocity(x))(x_eval)
            rel_l2 = jnp.linalg.norm(pred - u_star_eval) / (true_norm + 1e-30)
            history["iterations"].append(iteration)
            history["relative_l2_errors"].append(float(rel_l2))
            history["losses"].append(float(loss))
            history["times"].append(float(elapsed_total))
            print(f"Iter {iteration:5d} | Loss: {loss:.3e} | Rel L2 Err: {rel_l2:.3e} | Total: {elapsed_total:.1f}s")

    out = {
        "config": vars(args),
        "history": history,
        "params": params,
        "total_time": float(timeit.default_timer() - start),
    }
    file_name = os.path.join(args.output_dir, f"gn_beltrami_seed_{args.seed}.pkl")
    with open(file_name, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved results to {file_name}")


if __name__ == "__main__":
    main()
