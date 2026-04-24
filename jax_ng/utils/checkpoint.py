"""Checkpoint and history persistence."""
import pickle


def save(params, path: str) -> None:
    """Pickle ``params`` to ``path``."""
    with open(path, "wb") as f:
        pickle.dump(params, f)


def load(path: str):
    """Load pickled params from ``path``."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_history(history: list, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(history, f)


def load_history(path: str) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)
