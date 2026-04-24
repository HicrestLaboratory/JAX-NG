"""Activation functions and their first/second derivatives.

Used by jet propagation — we need analytic derivatives to propagate
Jacobians and Laplacians without second-order AD.
"""
import jax.numpy as jnp


def tanh(t):
    return jnp.tanh(t)

def tanh_p(t):
    return 1.0 - jnp.square(jnp.tanh(t))

def tanh_pp(t):
    th = jnp.tanh(t)
    return -2.0 * th * (1.0 - jnp.square(th))


def swish(t):
    import jax
    return t * jax.nn.sigmoid(t)

def swish_p(t):
    import jax
    s = jax.nn.sigmoid(t)
    return s + t * s * (1.0 - s)

def swish_pp(t):
    import jax
    s  = jax.nn.sigmoid(t)
    sp = s * (1.0 - s)
    return 2.0 * sp + t * sp * (1.0 - 2.0 * s)


_REGISTRY = {
    "tanh":  (tanh,  tanh_p,  tanh_pp),
    "swish": (swish, swish_p, swish_pp),
}


def get(name: str = "tanh"):
    """Return ``(act, act', act'')`` triple for the named activation.

    Parameters
    ----------
    name : ``"tanh"`` (default) or ``"swish"``
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_REGISTRY)}")
    return _REGISTRY[name]
