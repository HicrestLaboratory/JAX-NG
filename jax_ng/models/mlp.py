"""Plain MLP and periodic-embedding forward passes."""
from typing import Callable
import jax.numpy as jnp
from jax_ng.models.activations import tanh


def forward(params, x, activation: Callable = tanh):
    h = x
    for W, b in params[:-1]:
        h = activation(jnp.dot(W, h) + b)
    W, b = params[-1]
    return jnp.dot(W, h) + b


def periodic_embed(x, periods, n_modes: int = 1):
    omegas = 2.0 * jnp.pi / jnp.array(periods)
    parts = []
    for i, w in enumerate(omegas):
        for k in range(1, n_modes + 1):
            a = k * w * x[i]
            parts += [jnp.sin(a), jnp.cos(a)]
    return jnp.stack(parts)


def layer_sizes(input_dim: int, width: int, depth: int, output_dim: int):
    return [input_dim] + [width] * depth + [output_dim]


def periodic_input_dim(raw_dim: int, n_modes: int) -> int:
    return 2 * raw_dim * n_modes
