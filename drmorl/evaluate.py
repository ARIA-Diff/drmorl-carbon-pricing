"""Out-of-sample evaluation of a pricing policy, Section 5.4 (performance measures)."""
import numpy as np

from .shocks import draw_shocks, scenario_spec
from .utils import cvar, make_rng


def scenario_shocks(cfg, names, seed, n=None):
    """Common random numbers: one fixed (n, T, 4) shock array per scenario, shared by all methods."""
    n = n or cfg["eval"]["n_traj"]
    T = cfg["economy"]["T"]
    out = {}
    for k, name in enumerate(names):
        out[name] = draw_shocks(cfg, scenario_spec(cfg, name), n, T, make_rng(seed, 200 + k))
    return out


def simulate(econ, policy, xi):
    """Roll the feedback policy forward on the shock paths xi of shape (n, T, 4)."""
    n, T, _ = xi.shape
    s = econ.initial_state(n)
    keys = ("E", "Lam", "mu", "loss")
    sim = {k: np.empty((n, T)) for k in keys}
    sim["p"] = np.empty((n, T))
    sim["c"] = np.empty((n, T, 3))
    for t in range(T):
        a = policy(t, s)
        s, out = econ.step(t, s, a, xi[:, t])
        sim["p"][:, t] = a
        for k in keys:
            sim[k][:, t] = out[k]
        sim["c"][:, t] = out["c"]
    return sim


def metrics(econ, cfg, sim, w=None):
    """CE, PVL, VOL, J, violation probability, overshoot and their CVaR counterparts."""
    w = econ.w if w is None else np.asarray(w, dtype=float)
    lvl = cfg["eval"]["cvar_level"]
    T = econ.T
    disc = econ.gamma ** np.arange(T)
    CE = sim["E"].sum(1)
    PVL = (sim["loss"] * disc).sum(1)
    J = ((sim["c"] @ w) * disc).sum(1)
    dp = np.abs(np.diff(sim["p"], axis=1, prepend=econ.p_init))
    marks = [int(y) - cfg["economy"]["start_year"] for y in cfg["eval"]["year_marks"]]
    m = dict(
        CE=CE.mean(), PVL=PVL.mean(), PVLc=cvar(PVL, lvl), VOL=dp.mean(),
        J=J.mean(), Jc=cvar(J, lvl),
        V=float((CE > econ.B).mean()), X=float(np.maximum(CE - econ.B, 0.0).mean()),
        CEc=cvar(CE, lvl),
        ab=sim["mu"].mean(), lossGDP=100.0 * sim["Lam"].mean(),
    )
    for y, t in zip(cfg["eval"]["year_marks"], marks):
        m["P%d" % (y % 100)] = float(np.median(sim["p"][:, t]))
    return m


def path_record(sim, econ):
    """Per-trajectory arrays kept for the figures (single precision)."""
    disc = econ.gamma ** np.arange(econ.T)
    return dict(p=sim["p"].astype(np.float32), E=sim["E"].astype(np.float32),
                Lam=sim["Lam"].astype(np.float32),
                CE=sim["E"].sum(1).astype(np.float32),
                PVL=(sim["loss"] * disc).sum(1).astype(np.float32))
