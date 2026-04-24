"""Grid line-search: evaluate a fixed geometric schedule and pick the best step.

JIT-friendly — the candidate grid is static, so this compiles cleanly
inside a ``@jit``-decorated optimizer step.
"""
import jax
import jax.numpy as jnp
from jax import vmap


def grid_search(loss_fn, params, direction, current_loss=None,
                *, base: float = 0.5, n_steps: int = 16, start_exp: float = 0.0):
    """Evaluate ``alpha = base^k`` for k in ``linspace(start_exp, n_steps-1)``
    and return the ``(alpha, loss)`` pair with the smallest loss.

    Parameters
    ----------
    base      : geometric ratio (default 0.5)
    n_steps   : number of candidate steps
    start_exp : exponent for the largest step (default 0 → alpha_max = 1)
    """
    alphas = base ** jnp.linspace(start_exp, float(n_steps - 1), n_steps)

    def eval_alpha(a):
        p_new = jax.tree_util.tree_map(lambda p, d: p - a * d, params, direction)
        return loss_fn(p_new)

    losses  = vmap(eval_alpha)(alphas)
    best    = jnp.argmin(losses)
    return alphas[best], losses[best]
