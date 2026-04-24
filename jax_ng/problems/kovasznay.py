"""2-D Kovasznay flow. Output: [u, v, p]."""
import jax.numpy as jnp
import jax_ng.models as M


class Kovasznay:
    def __init__(self, Re=40.0, activation="tanh"):
        self.Re=Re; self.activation=activation
        nu=1.0/Re
        self.lam = 1.0/(2.0*nu) - jnp.sqrt(1.0/(4.0*nu**2) + 4.0*jnp.pi**2)

    def exact_uv(self, x):
        l=self.lam
        return jnp.array([
            1.0 - jnp.exp(l*x[0])*jnp.cos(2.0*jnp.pi*x[1]),
            (l/(2.0*jnp.pi))*jnp.exp(l*x[0])*jnp.sin(2.0*jnp.pi*x[1]),
        ])

    def interior_res(self, params, x):
        uvp, J, H = M.jet_full(params, x, self.activation)
        u,v = uvp[0],uvp[1]; nu=1.0/self.Re
        return jnp.stack([
            J[0,0]+J[1,1],
            u*J[0,0]+v*J[0,1]+J[2,0]-nu*(H[0,0,0]+H[0,1,1]),
            u*J[1,0]+v*J[1,1]+J[2,1]-nu*(H[1,0,0]+H[1,1,1]),
        ])

    def boundary_res(self, params, x):
        uvp,_,_ = M.jet_full(params, x, self.activation)
        return uvp[0:2] - self.exact_uv(x)

    def layer_sizes(self, width, depth):
        return M.layer_sizes(2, width, depth, 3)

    def init_params(self, width, depth, key):
        return M.glorot_init_bias(self.layer_sizes(width, depth), key)
