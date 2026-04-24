"""Tests for jax_ng.optimizers."""
import pytest
import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)

from jax_ng import linesearch, optimizers, problems, samplers
from jax_ng.optimizers import SolveConfig


KEY = random.PRNGKey(0)
BOX = ((-1.0, 1.0), (-1.0, 1.0))


class TestGaussNewtonModes:
    def test_step_finite_dual(self, helmholtz_setup):
        opt, params = helmholtz_setup
        opt.cfg = SolveConfig(mode="dual", damping=1e-6)
        loss, _, _ = opt.step(params, opt.init(params), KEY)
        assert jnp.isfinite(loss)

    def test_step_finite_primal(self, helmholtz_setup):
        opt, params = helmholtz_setup
        opt.cfg = SolveConfig(mode="primal", damping=1e-6)
        loss, _, _ = opt.step(params, opt.init(params), KEY)
        assert jnp.isfinite(loss)

    def test_step_finite_auto(self, helmholtz_setup):
        opt, params = helmholtz_setup
        opt.cfg = SolveConfig(mode="auto", damping=1e-6)
        loss, _, _ = opt.step(params, opt.init(params), KEY)
        assert jnp.isfinite(loss)

    def test_auto_selects_primal_when_residuals_exceed_params(self):
        pde = problems.Helmholtz(A1=1.0, A2=1.0, k_val=1.0, n_modes=1)
        params = pde.init_params(width=2, depth=1, key=KEY)
        sampler = lambda key: samplers.uniform_box(key, 128, 0, BOX)
        ls = linesearch.build("grid_search", n_steps=8)
        opt = optimizers.GaussNewton(
            pde.interior_res,
            pde.boundary_res,
            sampler,
            ls,
            SolveConfig(mode="auto", damping=1e-6),
        )
        loss, _, _ = opt.step(params, opt.init(params), KEY)
        assert jnp.isfinite(loss)

    def test_invalid_mode_raises(self, helmholtz_setup):
        opt, params = helmholtz_setup
        opt.cfg = SolveConfig(mode="invalid", damping=1e-6)
        with pytest.raises(ValueError):
            opt.step(params, opt.init(params), KEY)
