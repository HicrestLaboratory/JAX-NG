"""Backtracking line-searches: Armijo (sufficient decrease) and weak Wolfe."""
import jax
import jax.numpy as jnp


def _eval(loss_fn, params, direction, alpha):
    p_new = jax.tree_util.tree_map(lambda p, d: p - alpha * d, params, direction)
    return loss_fn(p_new)


def _fd_directional(loss_fn, params, direction, f0, eps=1e-7):
    """Finite-difference approximation of the directional derivative."""
    return (_eval(loss_fn, params, direction, eps) - f0) / eps


def armijo(loss_fn, params, direction, current_loss,
           *, alpha_init: float = 1.0, rho: float = 0.5,
           c1: float = 1e-4, max_iter: int = 20):
    """Armijo / sufficient-decrease backtracking.

    Parameters
    ----------
    alpha_init : starting step size
    rho        : reduction factor per trial
    c1         : sufficient-decrease constant
    max_iter   : maximum halvings before giving up
    """
    slope = _fd_directional(loss_fn, params, direction, current_loss)
    alpha = alpha_init
    for _ in range(max_iter):
        f_new = _eval(loss_fn, params, direction, alpha)
        if f_new <= current_loss + c1 * alpha * slope:
            return alpha, f_new
        alpha *= rho
    f_final = _eval(loss_fn, params, direction, alpha)
    return alpha, f_final


def wolfe(loss_fn, params, direction, current_loss,
          *, alpha_init: float = 1.0, rho: float = 0.5,
          c1: float = 1e-4, c2: float = 0.9, max_iter: int = 20):
    """Weak Wolfe conditions: Armijo + curvature.

    Parameters
    ----------
    c2 : curvature constant  (0 < c1 < c2 < 1)
    """
    eps   = 1e-7
    slope = _fd_directional(loss_fn, params, direction, current_loss)
    alpha = alpha_init
    for _ in range(max_iter):
        f_new = _eval(loss_fn, params, direction, alpha)
        if f_new > current_loss + c1 * alpha * slope:
            alpha *= rho
            continue
        new_slope = (_eval(loss_fn, params, direction, alpha - eps) - f_new) / eps
        if new_slope < c2 * slope:
            alpha *= rho
            continue
        return alpha, f_new
    f_final = _eval(loss_fn, params, direction, alpha)
    return alpha, f_final
