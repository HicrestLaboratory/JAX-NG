"""3D Beltrami flow in space-time: input (t, x, y, z), output (u, v, w, p)."""
from __future__ import annotations

import jax.numpy as jnp
import jax_ng.models as M


class Beltrami3D:
    """Beltrami benchmark with incompressible Navier-Stokes residuals."""

    def __init__(self, a=1.0, d=1.0, Re=1.0, activation="tanh"):
        self.a = a
        self.d = d
        self.Re = Re
        self.activation = activation

    def exact_velocity(self, txyz):
        t, x, y, z = txyz[0], txyz[1], txyz[2], txyz[3]
        a, d = self.a, self.d
        decay = jnp.exp(-(d ** 2) * t)
        u = -a * (jnp.exp(a * x) * jnp.sin(a * y + d * z) + jnp.exp(a * z) * jnp.cos(a * x + d * y)) * decay
        v = -a * (jnp.exp(a * y) * jnp.sin(a * z + d * x) + jnp.exp(a * x) * jnp.cos(a * y + d * z)) * decay
        w = -a * (jnp.exp(a * z) * jnp.sin(a * x + d * y) + jnp.exp(a * y) * jnp.cos(a * z + d * x)) * decay
        return jnp.array([u, v, w])

    def interior_res(self, params, x):
        """Returns [continuity, mom_x, mom_y, mom_z]."""
        uvwp, J, H = M.jet_full(params, x, self.activation)
        u, v, w = uvwp[0], uvwp[1], uvwp[2]

        u_t, u_x, u_y, u_z = J[0, 0], J[0, 1], J[0, 2], J[0, 3]
        v_t, v_x, v_y, v_z = J[1, 0], J[1, 1], J[1, 2], J[1, 3]
        w_t, w_x, w_y, w_z = J[2, 0], J[2, 1], J[2, 2], J[2, 3]
        p_x, p_y, p_z = J[3, 1], J[3, 2], J[3, 3]

        lap_u = H[0, 1, 1] + H[0, 2, 2] + H[0, 3, 3]
        lap_v = H[1, 1, 1] + H[1, 2, 2] + H[1, 3, 3]
        lap_w = H[2, 1, 1] + H[2, 2, 2] + H[2, 3, 3]

        continuity = u_x + v_y + w_z
        mom_x = u_t + (u * u_x + v * u_y + w * u_z) + p_x - (1.0 / self.Re) * lap_u
        mom_y = v_t + (u * v_x + v * v_y + w * v_z) + p_y - (1.0 / self.Re) * lap_v
        mom_z = w_t + (u * w_x + v * w_y + w * w_z) + p_z - (1.0 / self.Re) * lap_w

        return jnp.stack([continuity, mom_x, mom_y, mom_z])

    def boundary_res(self, params, x):
        uvwp = M.mlp(params, x)
        return uvwp[0:3] - self.exact_velocity(x)

    def layer_sizes(self, width, depth):
        return M.layer_sizes(4, width, depth, 4)

    def init_params(self, width, depth, key):
        return M.glorot_init_bias(self.layer_sizes(width, depth), key)
