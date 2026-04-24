"""Trainer — minimal training loop compatible with all jax_ng optimizers."""
from __future__ import annotations
import os
import timeit
from typing import Callable, Optional

import jax

from jax_ng.utils.checkpoint import save as save_checkpoint, save_history
from jax_ng.utils.plotting   import plot_history


class Trainer:
    """Run an optimizer for ``n_iters`` steps with logging and checkpointing.

    Parameters
    ----------
    optimizer        : any jax_ng optimizer (GaussNewton, MultiStageGN, …)
    n_iters          : total training iterations
    eval_fn          : ``(params) -> dict`` called every ``log_interval`` steps
    log_interval     : how often to log and call ``eval_fn``
    save_dir         : directory for checkpoints and history (``None`` = no save)
    checkpoint_every : save a checkpoint every N iterations
    time_limit       : stop after this many seconds (``None`` = unlimited)
    early_stop_tol   : stop if loss drops below this value
    verbose          : print log lines

    Example
    -------
    >>> trainer = Trainer(opt, n_iters=500, eval_fn=my_eval, log_interval=50)
    >>> params, history = trainer.run(params, key)
    """

    def __init__(
        self,
        optimizer,
        n_iters:          int      = 1000,
        eval_fn:          Optional[Callable] = None,
        log_interval:     int      = 50,
        save_dir:         Optional[str] = None,
        checkpoint_every: int      = 500,
        time_limit:       Optional[float] = None,
        early_stop_tol:   Optional[float] = None,
        verbose:          bool     = True,
    ):
        self.optimizer        = optimizer
        self.n_iters          = n_iters
        self.eval_fn          = eval_fn
        self.log_interval     = log_interval
        self.save_dir         = save_dir
        self.checkpoint_every = checkpoint_every
        self.time_limit       = time_limit
        self.early_stop_tol   = early_stop_tol
        self.verbose          = verbose

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def run(self, params, key):
        """Execute the training loop.

        Returns
        -------
        params  : final parameter pytree
        history : list of dicts recorded at each log step
        """
        opt_state = self.optimizer.init(params)
        history   = []
        t0        = timeit.default_timer()

        for i in range(1, self.n_iters + 1):
            if self.time_limit and (timeit.default_timer() - t0) > self.time_limit:
                if self.verbose:
                    print(f"Time limit reached at iter {i}.")
                break

            key, sk = jax.random.split(key)
            loss, params, opt_state = self.optimizer.step(params, opt_state, sk)
            loss_f = float(loss)

            if i % self.log_interval == 0:
                record = {"iter": i, "loss": loss_f}
                if self.eval_fn is not None:
                    metrics = self.eval_fn(params)
                    if isinstance(metrics, dict):
                        record.update(metrics)
                    else:
                        record["metric"] = float(metrics)
                history.append(record)

                if self.verbose:
                    elapsed = timeit.default_timer() - t0
                    msg = f"Iter {i:5d} | Loss: {loss_f:.3e} | Time: {elapsed:.1f}s"
                    for k, v in record.items():
                        if k not in ("iter", "loss"):
                            msg += f" | {k}: {float(v):.3e}"
                    print(msg)

            if self.save_dir and i % self.checkpoint_every == 0:
                save_checkpoint(params,
                                os.path.join(self.save_dir, f"ckpt_{i:06d}.pkl"))

            if self.early_stop_tol and loss_f < self.early_stop_tol:
                if self.verbose:
                    print(f"Early stop at iter {i} (loss={loss_f:.2e}).")
                break

        if self.save_dir:
            save_checkpoint(params, os.path.join(self.save_dir, "final_params.pkl"))
            save_history(history, os.path.join(self.save_dir, "history.pkl"))
            plot_history(history,
                         save_path=os.path.join(self.save_dir, "training_plot.png"))

        return params, history
