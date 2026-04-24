"""MultiStageGN — phase-based training schedule wrapper.

Wraps any GaussNewton instance and applies a list of PhaseConfig objects
sequentially, overriding damping per phase and supporting early stopping.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional

import jax

from jax_ng.optimizers.gauss_newton import GaussNewton, SolveConfig


@dataclass
class PhaseConfig:
    """Configuration for a single training phase.

    Parameters
    ----------
    n_iters        : number of iterations in this phase
    damping        : Tikhonov damping override for this phase
    early_stop_tol : stop early if loss drops below this value (optional)
    label          : human-readable name for logging
    """
    n_iters:        int
    damping:        float
    early_stop_tol: Optional[float] = None
    label:          str             = "phase"


class MultiStageGN:
    """Run a GaussNewton optimizer through a sequence of phases.

    Each phase can have its own damping and early-stop threshold.

    Parameters
    ----------
    optimizer : a :class:`~jax_ng.optimizers.gauss_newton.GaussNewton` instance
    phases    : list of :class:`PhaseConfig`

    Example
    -------
    >>> phases = [
    ...     PhaseConfig(n_iters=2000, damping=1e-11, label="warmup"),
    ...     PhaseConfig(n_iters=1000, damping=5e-9, early_stop_tol=5e-16, label="refine"),
    ... ]
    >>> solver = MultiStageGN(optimizer, phases)
    >>> params, history = solver.train(params, key)
    """

    def __init__(self, optimizer: GaussNewton, phases: List[PhaseConfig]):
        self.optimizer = optimizer
        self.phases    = phases

    def train(
        self,
        params,
        key,
        callback:      Optional[Callable] = None,
        checkpoint_fn: Optional[Callable] = None,
    ):
        """Run all phases sequentially.

        Parameters
        ----------
        callback      : called as ``callback(phase_idx, local_iter, loss, params)``
        checkpoint_fn : called as ``checkpoint_fn(global_iter, params)``

        Returns
        -------
        params  : final parameter pytree
        history : list of ``(global_iter, phase_label, loss)`` tuples
        """
        opt_state = self.optimizer.init(params)
        history   = []
        global_i  = 0
        saved_cfg = self.optimizer.cfg

        for phase_i, phase in enumerate(self.phases):
            # override damping for this phase
            self.optimizer.cfg = SolveConfig(
                mode    = saved_cfg.mode,
                damping = phase.damping,
                precond = saved_cfg.precond,
            )

            for local_i in range(phase.n_iters):
                key, sk = jax.random.split(key)
                loss, params, opt_state = self.optimizer.step(params, opt_state, sk)

                global_i += 1
                history.append((global_i, phase.label, float(loss)))

                if callback is not None:
                    callback(phase_i, local_i, float(loss), params)
                if checkpoint_fn is not None:
                    checkpoint_fn(global_i, params)

                if phase.early_stop_tol is not None and float(loss) < phase.early_stop_tol:
                    print(f"[{phase.label}] early stop at iter {local_i + 1} "
                          f"(loss={loss:.2e} < {phase.early_stop_tol:.0e})")
                    break

        self.optimizer.cfg = saved_cfg   # restore
        return params, history
