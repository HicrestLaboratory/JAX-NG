"""jax_ng.samplers
==================
Collocation-point samplers.

Submodules
----------
box       uniform_box — hyper-rectangle interior + face boundary
triangle  interior, boundary, wedge, apex_heavy — triangular domains
"""
from jax_ng.samplers.box      import uniform_box
from jax_ng.samplers.triangle import interior as triangle_interior
from jax_ng.samplers.triangle import boundary as triangle_boundary
from jax_ng.samplers.triangle import wedge, apex_heavy

__all__ = [
    "uniform_box",
    "triangle_interior",
    "triangle_boundary",
    "wedge",
    "apex_heavy",
]
