"""jax_ng.problems exports."""

from jax_ng.problems.helmholtz import Helmholtz
from jax_ng.problems.kovasznay import Kovasznay
from jax_ng.problems.kdv import KdVWindowed
from jax_ng.problems.ks1d import KS1DWindowed
from jax_ng.problems.stokes_wedge import StokesWedge
from jax_ng.problems.beltrami import Beltrami3D

__all__ = ["Helmholtz", "Kovasznay", "KdVWindowed", "KS1DWindowed", "StokesWedge", "Beltrami3D"]
