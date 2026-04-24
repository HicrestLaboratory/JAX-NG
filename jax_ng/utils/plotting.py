"""Training curve plotting (optional — requires matplotlib)."""
from typing import List, Optional


def plot_history(history: List[dict], save_path: Optional[str] = None,
                 show: bool = False) -> None:
    """Plot loss and any numeric metrics from a history list.

    Parameters
    ----------
    history   : list of dicts with keys ``"iter"``, ``"loss"``, and optional metrics
    save_path : if given, save figure to this path
    show      : call ``plt.show()`` if True
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    iters       = [r["iter"]  for r in history]
    losses      = [r["loss"]  for r in history]
    metric_keys = [k for k in history[0] if k not in ("iter", "loss")]

    ncols = 1 + len(metric_keys)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    axes[0].semilogy(iters, losses)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].grid(True, alpha=0.3)

    for ax, key in zip(axes[1:], metric_keys):
        vals = [r.get(key, float("nan")) for r in history]
        ax.semilogy(iters, vals, color="darkorange")
        ax.set_title(key)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
