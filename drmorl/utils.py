"""Small helpers: random streams, empirical risk measures, logging, I/O."""
import os
import pickle
import time

import numpy as np


def make_rng(seed, *keys):
    """Independent, reproducible random stream identified by integer keys."""
    return np.random.default_rng([int(seed)] + [int(k) for k in keys])


def mean_cvar_empirical(Z, alpha, kappa):
    """(1-kappa) * mean + kappa * CVaR_alpha along the last axis.

    Z holds equally weighted atoms.  CVaR is the average of the upper
    (1-alpha) tail with fractional weight on the boundary atom (exact).
    """
    Z = np.asarray(Z, dtype=float)
    K = Z.shape[-1]
    mean = Z.mean(-1)
    if kappa == 0.0:
        return mean
    zs = np.sort(Z, axis=-1)
    k = np.arange(1, K + 1)
    wts = np.clip(k / K - alpha, 0.0, 1.0 / K)
    cv = (zs * wts).sum(-1) / (1.0 - alpha)
    return (1.0 - kappa) * mean + kappa * cv


def cvar(x, level):
    """Empirical CVaR of a 1-d sample at the given level."""
    return float(mean_cvar_empirical(np.asarray(x, dtype=float), level, 1.0))


class Logger:
    """Prints to stdout and appends to a log file."""

    def __init__(self, path=None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, msg):
        print(msg, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)
