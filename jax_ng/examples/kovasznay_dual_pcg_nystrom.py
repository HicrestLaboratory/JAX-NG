"""Kovasznay flow with dual GN-PCG and Nyström preconditioning."""
import argparse

import jax
jax.config.update("jax_enable_x64", True)
from jax import random, vmap

from jax_ng import linesearch, models, optimizers, problems, samplers, utils


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-interior", type=int, default=400)
    parser.add_argument("--n-boundary", type=int, default=100)
    parser.add_argument("--eval-points", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-dir", default="./runs/kovasznay_dual_pcg_nystrom")
    parser.add_argument("--damping", type=float, default=1e-8)
    parser.add_argument(
        "--preconditioner",
        choices=("none", "column_nystrom", "sketch_nystrom", "exact"),
        default="column_nystrom",
    )
    parser.add_argument("--pcg-maxiter", type=int, default=500)
    parser.add_argument("--ew-gamma", type=float, default=0.9)
    parser.add_argument("--ew-alpha", type=float, default=1.5)
    parser.add_argument("--ew-eta-min", type=float, default=1e-4)
    parser.add_argument("--ew-eta-max", type=float, default=5e-1)
    parser.add_argument("--true-residual", action="store_true")
    parser.add_argument("--landmark-interior", type=int, default=200)
    parser.add_argument("--landmark-boundary", type=int, default=100)
    parser.add_argument("--nystrom-rank", type=int, default=0)
    parser.add_argument("--nystrom-eig-rtol", type=float, default=1e-12)
    parser.add_argument("--sketch-size", type=int, default=800)
    parser.add_argument("--sketch-nu-factor", type=float, default=1.0)
    parser.add_argument("--sketch-chol-jitter", type=float, default=1e-14)
    return parser.parse_args()


def main():
    args = parse_args()
    key = random.PRNGKey(args.seed)
    pde = problems.Kovasznay(Re=40.0)
    params = pde.init_params(width=args.width, depth=args.depth, key=key)
    box = ((-0.5, 1.0), (-0.5, 1.5))

    sampler = lambda k: samplers.uniform_box(k, args.n_interior, args.n_boundary, box)
    ls = linesearch.build("grid_search", n_steps=16)
    cfg = optimizers.DualPCGNystromConfig(
        damping=args.damping,
        preconditioner=args.preconditioner,
        pcg=optimizers.PCGConfig(
            maxiter=args.pcg_maxiter,
            ew_gamma=args.ew_gamma,
            ew_alpha=args.ew_alpha,
            ew_eta_min=args.ew_eta_min,
            ew_eta_max=args.ew_eta_max,
            true_residual=args.true_residual,
        ),
        column=optimizers.ColumnNystromConfig(
            landmark_interior=args.landmark_interior,
            landmark_boundary=args.landmark_boundary,
            rank=args.nystrom_rank,
            eig_rtol=args.nystrom_eig_rtol,
        ),
        sketch=optimizers.SketchNystromConfig(
            sketch_size=args.sketch_size,
            nu_factor=args.sketch_nu_factor,
            chol_jitter=args.sketch_chol_jitter,
        ),
    )
    opt = optimizers.DualPCGNystrom(
        interior_res_fn=pde.interior_res,
        boundary_res_fn=pde.boundary_res,
        sampler_fn=sampler,
        linesearch_fn=ls,
        solve_config=cfg,
        int_res_dim=3,
        bnd_res_dim=2,
    )

    key, eval_key = random.split(key)
    x_eval, _ = samplers.uniform_box(eval_key, args.eval_points, 0, box)
    uv_true = vmap(pde.exact_uv)(x_eval)

    def eval_fn(p):
        preds = vmap(lambda x: models.jet_full(p, x)[0][0:2])(x_eval)
        return {"rel_l2_uv": utils.rel_l2(preds, uv_true)}

    trainer = utils.Trainer(
        opt,
        n_iters=args.max_iters,
        eval_fn=eval_fn,
        log_interval=args.log_interval,
        time_limit=args.time_limit,
        save_dir=args.save_dir,
    )
    trainer.run(params, key)


if __name__ == "__main__":
    main()
