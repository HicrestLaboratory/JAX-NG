"""Tests for jax_ng.samplers."""
import pytest
import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)

from jax_ng.samplers import box as box_mod, triangle as tri_mod

BOX = ((-1.0, 1.0), (-1.0, 1.0))
V1  = jnp.array([0.5, 0.0])
V2  = jnp.array([0.0, 2.0])
V3  = jnp.array([1.0, 2.0])


class TestUniformBox:
    def test_interior_shape(self):
        x_int, _ = box_mod.uniform_box(random.PRNGKey(0), 100, 0, BOX)
        assert x_int.shape == (100, 2)

    def test_boundary_shape(self):
        _, x_bnd = box_mod.uniform_box(random.PRNGKey(0), 0, 50, BOX)
        assert x_bnd.shape == (50, 2)

    def test_interior_in_bounds(self):
        x_int, _ = box_mod.uniform_box(random.PRNGKey(0), 500, 0, BOX)
        assert jnp.all(x_int >= -1.0) and jnp.all(x_int <= 1.0)

    def test_boundary_on_face(self):
        _, x_bnd = box_mod.uniform_box(random.PRNGKey(0), 0, 200, BOX)
        on_face  = (jnp.isclose(x_bnd[:, 0], -1.) | jnp.isclose(x_bnd[:, 0], 1.) |
                    jnp.isclose(x_bnd[:, 1], -1.) | jnp.isclose(x_bnd[:, 1], 1.))
        assert jnp.all(on_face)

    def test_zero_boundary_shape(self):
        _, x_bnd = box_mod.uniform_box(random.PRNGKey(0), 100, 0, BOX)
        assert x_bnd.shape == (0, 2)

    def test_deterministic(self):
        x1, _ = box_mod.uniform_box(random.PRNGKey(7), 50, 0, BOX)
        x2, _ = box_mod.uniform_box(random.PRNGKey(7), 50, 0, BOX)
        assert jnp.allclose(x1, x2)


class TestTriangle:
    def test_interior_shape(self):
        x = tri_mod.interior(random.PRNGKey(0), 200, V1, V2, V3)
        assert x.shape == (200, 2)

    def test_boundary_shape(self):
        x = tri_mod.boundary(random.PRNGKey(0), 90, V1, V2, V3)
        assert x.shape == (90, 2)

    def test_wedge_shapes(self):
        x_int, x_bnd = tri_mod.wedge(random.PRNGKey(0), 300, 150, V1, V2, V3)
        assert x_int.shape == (300, 2) and x_bnd.shape == (150, 2)

    def test_interior_in_bounding_box(self):
        x = tri_mod.interior(random.PRNGKey(0), 500, V1, V2, V3)
        assert jnp.all(x[:, 0] >= 0.0) and jnp.all(x[:, 0] <= 1.0)
        assert jnp.all(x[:, 1] >= 0.0) and jnp.all(x[:, 1] <= 2.0)


class TestApexHeavy:
    def test_shape(self):
        x = tri_mod.apex_heavy(random.PRNGKey(0), 400, V1, V2, V3)
        assert x.shape == (400, 2)

    def test_more_near_apex_than_uniform(self):
        x_apex = tri_mod.apex_heavy(random.PRNGKey(0), 1000, V1, V2, V3, apex_fraction=0.9)
        x_uni  = tri_mod.interior(random.PRNGKey(1), 1000, V1, V2, V3)
        assert jnp.sum(x_apex[:, 1] < 0.1) > jnp.sum(x_uni[:, 1] < 0.1)
