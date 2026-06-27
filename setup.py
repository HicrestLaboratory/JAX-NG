from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"

setup(
    name             = "jax-ng",
    version          = "0.1.0",
    description      = "JAX-NG: modular JAX framework for second-order PINN optimizers",
    long_description = README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type = "text/markdown",
    url              = "https://github.com/HicrestLaboratory/JAX-NG",
    author           = "Anas Jnini, Flavio Vella",
    license          = "MIT",
    license_files    = ["LICENSE"],
    python_requires  = ">=3.9",
    packages         = find_packages(exclude=("runs", "runs.*")),
    install_requires = ["jax>=0.4.1", "jaxlib>=0.4.1", "numpy>=1.23"],
    extras_require   = {"plot": ["matplotlib>=3.5"], "dev": ["pytest"]},
    classifiers      = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
