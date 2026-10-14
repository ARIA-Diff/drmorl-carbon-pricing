"""Primal-dual calibration of the budget multiplier, Algorithm 1 (middle loop).

The multiplier follows the projected supergradient rule (Proposition 3(ii))

    nu <- [ nu + beta_nu * ( G_hat - delta ) ]_+ ,   G_hat = E[(C_T - B)_+]

where G_hat is a Monte Carlo estimate under the empirical distribution of the
shock record.  As in Section 5.3 the multiplier is obtained for the stochastic
benchmark (SP) and then used by all methods.
"""
import numpy as np

from .evaluate import simulate
from .fqi import train_policy
from .methods import make_method
from .utils import make_rng


def calibrate_multiplier(econ, cfg, sm, w, log=print, method="SP"):
    dc = cfg["dual"]
    spec = make_method(method, cfg)
    seed = cfg["seed"]
    xi = sm.resample((dc["n_sim"], econ.T), make_rng(seed, 300))   # fixed paths: empirical distribution
    nu = float(dc["nu0"])
    hist = []
    for k in range(int(dc["K_nu"])):
        pol = train_policy(econ, spec, sm, w, nu, cfg, seed)
        CE = simulate(econ, pol, xi)["E"].sum(1)
        G = float(np.maximum(CE - econ.B, 0.0).mean())
        V = float((CE > econ.B).mean())
        hist.append((k, nu, float(CE.mean()), V))
        log("  dual k=%d nu=%.3f CE=%.1f V=%.3f" % (k, nu, CE.mean(), V))
        if k < int(dc["K_nu"]) - 1:
            nu = max(0.0, nu + dc["beta"] * (G - econ.delta))
    return nu, hist
