import pytest
import jax
jax.config.update("jax_enable_x64", True)

from jax import random
from jax_ng import problems, samplers, linesearch, optimizers
from jax_ng.optimizers import SolveConfig

BOX = ((-1.0, 1.0), (-1.0, 1.0))
KEY = random.PRNGKey(0)


@pytest.fixture
def helmholtz_setup():
    pde     = problems.Helmholtz(A1=1., A2=1., k_val=1., n_modes=1)
    params  = pde.init_params(16, 2, KEY)
    sampler = lambda key: samplers.uniform_box(key, 50, 20, BOX)
    ls      = linesearch.build("grid_search", n_steps=8)
    opt     = optimizers.GaussNewton(
        pde.interior_res, pde.boundary_res, sampler, ls,
        SolveConfig(damping=1e-8),
    )
    return opt, params
