"""Parameter initialisation strategies."""
from jax import random
import jax.numpy as jnp


def glorot(sizes: list, key) -> list:
    """Glorot-uniform weights, zero biases.

    Parameters
    ----------
    sizes : layer widths including input and output, e.g. ``[2, 64, 64, 3]``

    Returns
    -------
    params : list of ``(W, b)`` tuples
    """
    keys = random.split(key, len(sizes))
    layers = []
    for m, n, k in zip(sizes[:-1], sizes[1:], keys):
        stddev = jnp.sqrt(2.0 / (m + n))
        W = stddev * random.normal(k, (n, m))
        b = jnp.zeros(n)
        layers.append((W, b))
    return layers


def glorot_bias(sizes: list, key) -> list:
    """Glorot weights *and* Glorot-initialised biases."""
    keys = random.split(key, len(sizes))
    layers = []
    for m, n, k in zip(sizes[:-1], sizes[1:], keys):
        wk, bk = random.split(k)
        stddev = jnp.sqrt(2.0 / (m + n))
        W = stddev * random.normal(wk, (n, m))
        b = stddev * random.normal(bk, (n,))
        layers.append((W, b))
    return layers
