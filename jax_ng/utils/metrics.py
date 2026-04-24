"""Error metrics."""
import jax.numpy as jnp
from jax import vmap


def rel_l2(pred, true):
    """Relative L2 error: ``||pred - true|| / ||true||``."""
    return jnp.linalg.norm(pred - true) / (jnp.linalg.norm(true) + 1e-16)


def l_inf(pred, true):
    """L∞ error: ``max |pred - true|``."""
    return jnp.max(jnp.abs(pred - true))


def eval_errors(params, model_fn, x_eval, u_true):
    """Compute relative L2 and L∞ on an evaluation set.

    Parameters
    ----------
    model_fn : ``(params, x) -> value`` per point
    x_eval   : ``(N, d)``
    u_true   : ``(N, ...)``

    Returns
    -------
    err_l2   : float
    err_linf : float
    """
    preds = vmap(lambda x: model_fn(params, x))(x_eval)
    return float(rel_l2(preds, u_true)), float(l_inf(preds, u_true))
