"""Helmholtz — Gauss-Newton + Jacobi preconditioning."""
import jax; jax.config.update("jax_enable_x64", True)
from jax import random, vmap
from jax_ng import problems, models, samplers, linesearch, optimizers, utils

pde    = problems.Helmholtz(A1=1.0, A2=4.0, k_val=50.0, periods=[2.0, 2.0], n_modes=1)
params = pde.init_params(width=30, depth=4, key=random.PRNGKey(0))
box     = tuple([(-1.0, 1.0)] * 2)
sampler = lambda key: samplers.uniform_box(key, 2000, 500, box)
ls      = linesearch.build("grid_search", n_steps=16)
opt = optimizers.GaussNewton(
    interior_res_fn=pde.interior_res, boundary_res_fn=pde.boundary_res,
    sampler_fn=sampler, linesearch_fn=ls,
    solve_config=optimizers.SolveConfig(mode="dual", damping=1e-12),
)
x_eval = random.uniform(random.PRNGKey(1), (10_000, 2), minval=-1., maxval=1.)
u_true = vmap(pde.exact_u)(x_eval)
def eval_fn(p):
    preds = vmap(lambda x: models.jet_laplacian_periodic(p, x, pde.periods, pde.n_modes)[0])(x_eval)
    return {"rel_l2": utils.rel_l2(preds, u_true), "l_inf": utils.l_inf(preds, u_true)}
trainer = utils.Trainer(opt, n_iters=500, eval_fn=eval_fn, log_interval=50, save_dir="./runs/helmholtz_gn")
params, history = trainer.run(params, random.PRNGKey(0))
