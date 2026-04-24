"""Helmholtz equation on [-1,1]^2.
PDE:  Δu + k²u = f,  exact: u* = sin(A1 π x)sin(A2 π y)
"""
import jax.numpy as jnp
import jax_ng.models as M


class Helmholtz:
    def __init__(self, A1=1.0, A2=4.0, k_val=50.0, periods=None, n_modes=1, activation="tanh"):
        self.A1=A1; self.A2=A2; self.k_val=k_val
        self.periods=periods or [2.0,2.0]; self.n_modes=n_modes; self.activation=activation

    def exact_u(self, x):
        return jnp.sin(self.A1*jnp.pi*x[0]) * jnp.sin(self.A2*jnp.pi*x[1])

    def forcing(self, x):
        u = self.exact_u(x)
        return -(jnp.pi**2*(self.A1**2+self.A2**2))*u + self.k_val**2*u

    def interior_res(self, params, x):
        u, lap_u = M.jet_laplacian_periodic(params, x, self.periods, self.n_modes, self.activation)
        return lap_u + self.k_val**2*u - self.forcing(x)

    def boundary_res(self, params, x):
        u, _ = M.jet_laplacian_periodic(params, x, self.periods, self.n_modes, self.activation)
        return u - self.exact_u(x)

    def layer_sizes(self, width, depth):
        return M.layer_sizes(M.periodic_input_dim(2, self.n_modes), width, depth, 1)

    def init_params(self, width, depth, key):
        return M.glorot_init(self.layer_sizes(width, depth), key)
