"""Windowed KdV problem with hard initial-condition ansatz."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
from jax.experimental.jet import jet


class KdVWindowed:
    """KdV on x in [-1, 1] with periodic boundary and time-windowed hard IC."""

    def __init__(
        self,
        x_min=-1.0,
        x_max=1.0,
        dt_window=1.0,
        n_windows=5,
        eta=1.0,
        mu2=0.022 ** 2,
        num_modes=10,
        width=128,
        depth=4,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.L = x_max - x_min
        self.dt_window = dt_window
        self.n_windows = n_windows
        self.eta = eta
        self.mu2 = mu2
        self.num_modes = num_modes
        self.width = width
        self.depth = depth

    def initial_condition(self, x):
        return jnp.cos(jnp.pi * x)

    def layer_sizes(self):
        return [1 + 2 * self.num_modes] + [self.width] * self.depth + [1]

    def init_params(self, key):
        sizes = self.layer_sizes()
        keys = random.split(key, len(sizes))
        params = []
        for m, n, k in zip(sizes[:-1], sizes[1:], keys):
            w_k, _ = random.split(k)
            scale = jnp.sqrt(2.0 / (m + n))
            W = random.normal(w_k, (n, m)) * scale
            b = jnp.zeros((n,))
            params.append((W, b))
        return params

    def init_stacked_params(self, params):
        return jax.tree_util.tree_map(
            lambda leaf: jnp.zeros((self.n_windows,) + leaf.shape),
            params,
        )

    def periodic_embedding(self, x):
        omega = 2.0 * jnp.pi / self.L
        feats = []
        for k in range(1, self.num_modes + 1):
            feats.append(jnp.sin(k * omega * x))
            feats.append(jnp.cos(k * omega * x))
        return jnp.concatenate([jnp.atleast_1d(f) for f in feats])

    def base_network(self, params, t, x):
        t_norm = t * (2.0 / self.dt_window) - 1.0
        x_emb = self.periodic_embedding(x)
        h = jnp.concatenate([jnp.atleast_1d(t_norm), x_emb])
        for W, b in params[:-1]:
            h = jnp.sin(jnp.dot(W, h) + b)
        Wf, bf = params[-1]
        return (jnp.dot(Wf, h) + bf)[0]

    def hard_ansatz(self, params, p_past, w_idx, t, x):
        phi_t = self.base_network(params, t, x)
        phi_0 = self.base_network(params, 0.0, x)
        u_ic = self.initial_condition(x)

        def eval_past_delta(p):
            return self.base_network(p, self.dt_window, x) - self.base_network(p, 0.0, x)

        deltas = jax.vmap(eval_past_delta)(p_past)
        mask = (jnp.arange(self.n_windows) < w_idx).astype(jnp.float64)
        u_ic = u_ic + jnp.sum(mask * deltas)
        return phi_t - phi_0 + u_ic

    def derivs_jet(self, params, p_past, w_idx, t, x):
        g_x = lambda xx: self.hard_ansatz(params, p_past, w_idx, t, xx)
        u_val, (u_x, _u_xx, u_xxx) = jet(g_x, (x,), ((1.0, 0.0, 0.0),))
        g_t = lambda tt: self.hard_ansatz(params, p_past, w_idx, tt, x)
        _, (u_t,) = jet(g_t, (t,), ((1.0,),))
        return u_val, u_t, u_x, u_xxx

    def pde_residual(self, params, p_past, w_idx, t, x):
        u, u_t, u_x, u_xxx = self.derivs_jet(params, p_past, w_idx, t, x)
        return u_t + self.eta * u * u_x + self.mu2 * u_xxx
