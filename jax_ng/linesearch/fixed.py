"""Fixed / constant step size — no search at all.

Useful for fixed-step updates where the step is controlled by a
norm constraint rather than a line-search.
"""
import jax


def fixed_step(loss_fn, params, direction, current_loss=None, *, alpha: float = 1.0):
    """Always returns ``alpha``.  Evaluates the loss at the new point.

    Returns
    -------
    alpha : float
    loss  : float
    """
    p_new = jax.tree_util.tree_map(lambda p, d: p - alpha * d, params, direction)
    return alpha, loss_fn(p_new)
