from setuptools import setup, find_packages

setup(
    name             = "JAX-NG",
    version          = "0.1.0",
    description      = "JAX-NG: modular JAX framework for second-order PINN optimizers",
    python_requires  = ">=3.9",
    packages         = find_packages(),
    install_requires = ["jax>=0.4.1", "jaxlib>=0.4.1", "numpy>=1.23"],
    extras_require   = {"plot": ["matplotlib>=3.5"], "dev": ["pytest"]},
)
