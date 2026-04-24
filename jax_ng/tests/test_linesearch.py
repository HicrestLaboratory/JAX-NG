"""Tests for jax_ng.linesearch."""
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax_ng.linesearch import grid, backtrack, fixed, build

# Simple quadratic: f(x) = 0.5 ||x - x*||^2, min at x* = [1, 2]
# gradient at x0=[0,0]: x0 - x* = [-1,-2]
# descent direction = -grad = [1, 2]  →  x0 - alpha*grad reaches x* at alpha=1
x_star    = jnp.array([1.0, 2.0])
x0        = jnp.array([0.0, 0.0])
direction = x0 - x_star      # gradient direction; step = x0 - alpha*direction

def loss_fn(x):
    return 0.5 * jnp.sum((x - x_star) ** 2)

f0 = loss_fn(x0)   # = 2.5


class TestGridSearch:
    def test_reduces_loss(self):
        _, best_loss = grid.grid_search(loss_fn, x0, direction, f0)
        assert best_loss < f0

    def test_near_optimal_alpha(self):
        alpha, _ = grid.grid_search(loss_fn, x0, direction, f0, n_steps=20)
        assert abs(float(alpha) - 1.0) < 0.1

    def test_custom_base(self):
        _, loss = grid.grid_search(loss_fn, x0, direction, f0, base=0.7, n_steps=15)
        assert loss < f0


class TestArmijo:
    def test_reduces_loss(self):
        _, loss = backtrack.armijo(loss_fn, x0, direction, f0)
        assert float(loss) < float(f0)

    def test_positive_alpha(self):
        alpha, _ = backtrack.armijo(loss_fn, x0, direction, f0)
        assert float(alpha) > 0


class TestWolfe:
    def test_reduces_loss(self):
        _, loss = backtrack.wolfe(loss_fn, x0, direction, f0)
        assert float(loss) < float(f0)


class TestFixed:
    def test_uses_given_alpha(self):
        alpha, _ = fixed.fixed_step(loss_fn, x0, direction, f0, alpha=0.5)
        assert float(alpha) == pytest.approx(0.5)


class TestBuildFactory:
    def test_grid(self):
        ls       = build("grid_search", n_steps=8)
        _, loss  = ls(loss_fn, x0, direction, f0)
        assert float(loss) < float(f0)

    def test_armijo(self):
        ls       = build("armijo")
        alpha, _ = ls(loss_fn, x0, direction, f0)
        assert float(alpha) > 0

    def test_fixed(self):
        ls       = build("fixed", alpha=0.25)
        alpha, _ = ls(loss_fn, x0, direction, f0)
        assert float(alpha) == pytest.approx(0.25)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            build("nonexistent")
