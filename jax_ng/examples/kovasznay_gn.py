"""Kovasznay flow — vector-residual GN (3 interior + 2 boundary components)."""
import jax; jax.config.update("jax_enable_x64", True)
from jax import random, vmap
from jax_ng import problems, models, samplers, linesearch, optimizers, utils

pde    = problems.Kovasznay(Re=40.0)
params = pde.init_params(width=50, depth=4, key=random.PRNGKey(0))
box     = tuple([(-0.5, 1.0), (-0.5, 1.5)])
sampler = lambda key: samplers.uniform_box(key, 400, 100, box)
ls      = linesearch.build("grid_search", n_steps=16)
opt = optimizers.GaussNewton(
    interior_res_fn=pde.interior_res, boundary_res_fn=pde.boundary_res,
    sampler_fn=sampler, linesearch_fn=ls,
    solve_config=optimizers.SolveConfig(mode="dual", damping=1e-12),
    int_res_dim=3, bnd_res_dim=2,
)
x_eval, _ = samplers.uniform_box(random.PRNGKey(1), 5000, 0, box)
uv_true   = vmap(pde.exact_uv)(x_eval)
def eval_fn(p):
    preds = vmap(lambda x: models.jet_full(p, x)[0][0:2])(x_eval)
    return {"rel_l2_uv": utils.rel_l2(preds, uv_true)}
trainer = utils.Trainer(opt, n_iters=5000, eval_fn=eval_fn, log_interval=50,
                         time_limit=120, save_dir="./runs/kovasznay_gn")
params, history = trainer.run(params, random.PRNGKey(0))
