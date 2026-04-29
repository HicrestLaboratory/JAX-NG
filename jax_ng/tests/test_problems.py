"""Tests for jax_ng.problems."""
import pytest
import jax
import jax.numpy as jnp
from jax import random, vmap

jax.config.update("jax_enable_x64", True)

from jax_ng.problems.helmholtz    import Helmholtz
from jax_ng.problems.kovasznay    import Kovasznay
from jax_ng.problems.kdv          import KdVWindowed
from jax_ng.problems.ks1d         import KS1DWindowed
from jax_ng.problems.stokes_wedge import StokesWedge
from jax_ng.problems.beltrami     import Beltrami3D


class TestHelmholtz:
    @pytest.fixture
    def pde(self):    return Helmholtz(A1=1.0, A2=1.0, k_val=1.0, n_modes=1)
    @pytest.fixture
    def params(self, pde): return pde.init_params(16, 2, random.PRNGKey(0))
    @pytest.fixture
    def x(self):      return jnp.array([0.3, -0.5])

    def test_exact_u(self, pde, x):
        import math
        expected = math.sin(math.pi * 0.3) * math.sin(math.pi * -0.5)
        assert jnp.allclose(pde.exact_u(x), jnp.array(expected), atol=1e-10)

    def test_interior_res_scalar(self, pde, params, x):
        r = pde.interior_res(params, x)
        assert r.shape == () and jnp.isfinite(r)

    def test_boundary_res_scalar(self, pde, params, x):
        r = pde.boundary_res(params, x)
        assert r.shape == () and jnp.isfinite(r)

    def test_vmap_interior(self, pde, params):
        xs = random.uniform(random.PRNGKey(0), (20, 2), minval=-1., maxval=1.)
        rs = vmap(lambda xi: pde.interior_res(params, xi))(xs)
        assert rs.shape == (20,) and jnp.all(jnp.isfinite(rs))

    def test_layer_sizes(self, pde):
        sz = pde.layer_sizes(32, 3)
        assert sz[0] == 4 and sz[-1] == 1 and len(sz) == 5


class TestKovasznay:
    @pytest.fixture
    def pde(self):    return Kovasznay(Re=40.0)
    @pytest.fixture
    def params(self, pde): return pde.init_params(16, 2, random.PRNGKey(0))
    @pytest.fixture
    def x(self):      return jnp.array([0.2, 0.3])

    def test_exact_uv_shape(self, pde, x):
        assert pde.exact_uv(x).shape == (2,)

    def test_interior_res_shape(self, pde, params, x):
        r = pde.interior_res(params, x)
        assert r.shape == (3,) and jnp.all(jnp.isfinite(r))

    def test_boundary_res_shape(self, pde, params, x):
        r = pde.boundary_res(params, x)
        assert r.shape == (2,) and jnp.all(jnp.isfinite(r))

    def test_lambda_negative(self, pde):
        assert float(pde.lam) < 0.0

    def test_layer_sizes(self, pde):
        sz = pde.layer_sizes(50, 4)
        assert sz[0] == 2 and sz[-1] == 3


class TestKdVWindowed:
    def test_residual_finite(self):
        pde = KdVWindowed(n_windows=2, width=16, depth=2, num_modes=2)
        params = pde.init_params(random.PRNGKey(0))
        stacked = pde.init_stacked_params(params)
        r = pde.pde_residual(params, stacked, jnp.array(0, dtype=jnp.int32), 0.1, 0.2)
        assert jnp.isfinite(r)


class TestKS1DWindowed:
    def test_residual_finite(self):
        pde = KS1DWindowed(n_windows=2, width=16, depth=2, num_modes=2)
        params = pde.init_params(random.PRNGKey(0))
        stacked = pde.init_stacked_params(params)
        r = pde.pde_residual(params, stacked, jnp.array(0, dtype=jnp.int32), 0.01, 1.0)
        assert jnp.isfinite(r)


class TestStokesWedge:
    def test_residual_shapes(self):
        pde = StokesWedge()
        params = pde.init_params(width=16, depth=2, key=random.PRNGKey(0))
        x = jnp.array([0.5, 1.0])
        r_int = pde.interior_res(params, x)
        r_bnd = pde.boundary_res(params, x)
        r_anc = pde.anchor_res(params, pde.anchor)
        assert r_int.shape == (3,)
        assert r_bnd.shape == (2,)
        assert r_anc.shape == (1,)


class TestBeltrami3D:
    def test_shapes(self):
        pde = Beltrami3D(Re=1.0)
        params = pde.init_params(width=16, depth=2, key=random.PRNGKey(0))
        x = jnp.array([0.2, -0.1, 0.3, -0.4])
        r_int = pde.interior_res(params, x)
        r_bnd = pde.boundary_res(params, x)
        uvw = pde.exact_velocity(x)
        assert r_int.shape == (4,)
        assert r_bnd.shape == (3,)
        assert uvw.shape == (3,)
