import jax
import jax.numpy as jnp
from jax import random

from jax_ng.optimizers.dual_pcg_nystrom import (
    ColumnNystromConfig,
    SketchNystromConfig,
    _build_sketch_nystrom_factors,
    build_column_nystrom_preconditioner,
    build_sketch_nystrom_preconditioner,
    pcg_solve_matvec,
)


def test_pcg_matches_dense_dual_solve():
    key = random.PRNGKey(0)
    J = random.normal(key, (9, 5), dtype=jnp.float64)
    b = random.normal(random.PRNGKey(1), (9,), dtype=jnp.float64)
    mu = jnp.array(1e-3, dtype=jnp.float64)

    matvec = lambda v: J @ (J.T @ v) + mu * v
    x_pcg, info = pcg_solve_matvec(matvec, b, 1e-10, 200, None, true_residual=True)
    K = J @ J.T
    x_dense = jnp.linalg.solve(K + mu * jnp.eye(K.shape[0], dtype=J.dtype), b)

    assert bool(info["converged"])
    assert jnp.linalg.norm(x_pcg - x_dense) / jnp.linalg.norm(x_dense) < 1e-8


def test_column_nystrom_full_landmarks_is_effective_preconditioner():
    key = random.PRNGKey(2)
    J = random.normal(key, (6, 10), dtype=jnp.float64)
    b = random.normal(random.PRNGKey(3), (6,), dtype=jnp.float64)
    mu = jnp.array(1e-4, dtype=jnp.float64)
    cfg = ColumnNystromConfig(landmark_interior=6, landmark_boundary=0, rank=0)
    m_inv = build_column_nystrom_preconditioner(
        J, mu, random.PRNGKey(4), 6, 0, 1, 1, cfg
    )

    K_mu = J @ J.T + mu * jnp.eye(J.shape[0], dtype=J.dtype)
    z = random.normal(random.PRNGKey(5), (6,), dtype=jnp.float64)
    assert jnp.linalg.norm(K_mu @ m_inv(z) - z) / jnp.linalg.norm(z) < 1e-8

    matvec = lambda v: J @ (J.T @ v) + mu * v
    _, info = pcg_solve_matvec(matvec, b, 1e-10, 20, m_inv, true_residual=True)
    assert bool(info["converged"])
    assert int(info["iters"]) <= 2


def test_sketch_nystrom_woodbury_orientation_matches_explicit_inverse():
    key = random.PRNGKey(6)
    J = random.normal(key, (7, 4), dtype=jnp.float64)
    mu = jnp.array(1e-3, dtype=jnp.float64)
    cfg = SketchNystromConfig(sketch_size=5, chol_jitter=1e-14)
    B, _, _ = _build_sketch_nystrom_factors(J, mu, random.PRNGKey(7), cfg)
    apply = build_sketch_nystrom_preconditioner(J, mu, random.PRNGKey(7), cfg)

    z = random.normal(random.PRNGKey(8), (7,), dtype=jnp.float64)
    P = B @ B.T + mu * jnp.eye(B.shape[0], dtype=B.dtype)
    rel = jnp.linalg.norm(P @ apply(z) - z) / jnp.linalg.norm(z)
    assert rel < 1e-7


def test_dual_pcg_nystrom_imported_from_public_namespace():
    from jax_ng import optimizers

    assert optimizers.DualPCGNystrom is not None
    assert optimizers.DualPCGNystromConfig is not None
