"""Tests for jax_ng.models (activations, init, mlp, jets)."""
import pytest
import jax
import jax.numpy as jnp
from jax import random, vmap

jax.config.update("jax_enable_x64", True)

from jax_ng.models import activations, init, jets
import jax_ng.models as mlp_mod
from jax_ng import models


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def params_1out():
    return init.glorot([2, 16, 16, 1], random.PRNGKey(0))

@pytest.fixture
def params_3out():
    return init.glorot([2, 16, 16, 3], random.PRNGKey(1))

@pytest.fixture
def params_periodic():
    return init.glorot([4, 16, 16, 1], random.PRNGKey(2))   # 2 dims * 2 * 1 mode

@pytest.fixture
def x():
    return jnp.array([0.3, -0.7])


# ── activations ───────────────────────────────────────────────────────────────
class TestActivations:
    @pytest.mark.parametrize("t", [-2.0, 0.0, 1.5])
    def test_tanh_p_finite_difference(self, t):
        t_a = jnp.array(t)
        fd  = (activations.tanh(t_a + 1e-6) - activations.tanh(t_a - 1e-6)) / 2e-6
        assert jnp.allclose(activations.tanh_p(t_a), fd, atol=1e-5)

    @pytest.mark.parametrize("t", [-1.0, 0.0, 2.0])
    def test_tanh_pp_finite_difference(self, t):
        t_a = jnp.array(t)
        fd  = (activations.tanh_p(t_a + 1e-5) - activations.tanh_p(t_a - 1e-5)) / 2e-5
        assert jnp.allclose(activations.tanh_pp(t_a), fd, atol=1e-4)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError):
            activations.get("relu99")


# ── init ──────────────────────────────────────────────────────────────────────
class TestInit:
    def test_glorot_shapes(self, params_1out):
        W, b = params_1out[0]
        assert W.shape == (16, 2) and b.shape == (16,)

    def test_glorot_no_nan(self, params_3out):
        for W, b in params_3out:
            assert not jnp.any(jnp.isnan(W))

    def test_glorot_bias_shapes(self):
        p = init.glorot_bias([2, 8, 1], random.PRNGKey(0))
        assert p[0][1].shape == (8,)


# ── mlp ───────────────────────────────────────────────────────────────────────
class TestMLP:
    def test_output_shape(self, params_1out, x):
        assert models.mlp(params_1out, x).shape == (1,)

    def test_output_shape_3(self, params_3out, x):
        assert models.mlp(params_3out, x).shape == (3,)

    def test_vmap(self, params_1out):
        xs   = jnp.ones((10, 2))
        outs = vmap(lambda xi: models.mlp(params_1out, xi))(xs)
        assert outs.shape == (10, 1)

    def test_periodic_embed_shape(self, x):
        emb = models.periodic_embedding(x, [2.0, 2.0], n_modes=3)
        assert emb.shape == (12,)   # 2 dims * 2 * 3 modes

    def test_layer_sizes(self):
        assert models.layer_sizes(2, 32, 3, 1) == [2, 32, 32, 32, 1]

    def test_periodic_input_dim(self):
        assert models.periodic_input_dim(2, 1) == 4
        assert models.periodic_input_dim(2, 5) == 20


# ── jets ──────────────────────────────────────────────────────────────────────
class TestJets:
    def test_full_shapes(self, params_3out, x):
        u, J, H = jets.full(params_3out, x)
        assert u.shape == (3,) and J.shape == (3, 2) and H.shape == (3, 2, 2)

    def test_full_jacobian_matches_jax(self, params_1out, x):
        _, J_jet, _ = jets.full(params_1out, x)
        J_auto      = jax.jacobian(lambda xi: models.mlp(params_1out, xi))(x)
        assert jnp.allclose(J_jet, J_auto, atol=1e-10)

    def test_full_hessian_matches_jax(self, params_1out, x):
        _, _, H_jet = jets.full(params_1out, x)
        H_auto      = jax.hessian(lambda xi: models.mlp(params_1out, xi)[0])(x)
        assert jnp.allclose(H_jet[0], H_auto, atol=1e-9)

    def test_laplacian_matches_hessian_trace(self, params_1out, x):
        _, _, H   = jets.full(params_1out, x)
        _, lap    = jets.laplacian(params_1out, x)
        assert jnp.allclose(lap[0], jnp.trace(H[0]), atol=1e-10)

    def test_laplacian_periodic_shape(self, params_periodic, x):
        u, lap = jets.laplacian_periodic(params_periodic, x, [2.0, 2.0], 1)
        assert u.shape == () and lap.shape == ()

    def test_laplacian_periodic_vs_fd(self, params_periodic, x):
        eps    = 1e-5
        u0, _  = jets.laplacian_periodic(params_periodic, x, [2.0, 2.0], 1)
        lap_fd = sum(
            (jets.laplacian_periodic(params_periodic, x.at[i].add(eps),  [2.,2.], 1)[0]
           - 2 * u0
           + jets.laplacian_periodic(params_periodic, x.at[i].add(-eps), [2.,2.], 1)[0])
            / eps**2 for i in range(2))
        _, lap_jet = jets.laplacian_periodic(params_periodic, x, [2.0, 2.0], 1)
        assert jnp.allclose(lap_jet, lap_fd, atol=1e-4)
