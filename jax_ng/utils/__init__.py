"""jax_ng.utils
===============
Training utilities, metrics, checkpointing, and plotting.

Submodules
----------
metrics     rel_l2, l_inf, eval_errors
checkpoint  save, load, save_history, load_history
plotting    plot_history
trainer     Trainer
"""
from jax_ng.utils.metrics    import rel_l2, l_inf, eval_errors
from jax_ng.utils.checkpoint import (save as save_checkpoint,
                                      load as load_checkpoint,
                                      save_history, load_history)
from jax_ng.utils.plotting   import plot_history
from jax_ng.utils.trainer    import Trainer

__all__ = [
    "rel_l2", "l_inf", "eval_errors",
    "save_checkpoint", "load_checkpoint",
    "save_history", "load_history",
    "plot_history",
    "Trainer",
]
