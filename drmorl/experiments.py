"""Experiment parts of Section 6: main, pareto, sens, sample.

Every part stores its raw output in ``<out>/<part>.pkl``; tables are produced from
these files by ``s03_tables.py``.
"""
import os

import numpy as np

from .dual import calibrate_multiplier
from .dynamics import CarbonEconomy
from .evaluate import metrics, path_record, scenario_shocks, simulate
from .fqi import train_policy
from .methods import METHOD_ORDER, make_method
from .shocks import ShockModel, draw_record, draw_shocks, scenario_spec
from .utils import load_pickle, make_rng, save_pickle

SCEN = ["S1", "S2", "S3", "S4", "S5"]


def setup(cfg, N=None, rec_key=0):
    """Economy, shock record and the learners' shock model."""
    econ = CarbonEconomy(cfg)
    N = int(N or cfg["shocks"]["N"])
    record = draw_record(cfg, N, make_rng(cfg["seed"], 100, N, rec_key))
    return econ, ShockModel(cfg, record)


def _line(name, scn, m):
    return ("%-7s %s CE=%.1f PVL=%.3f PVLc=%.3f VOL=%.2f J=%.3f Jc=%.3f V=%.3f X=%.2f CEc=%.1f "
            "P30=%.1f P40=%.1f P54=%.1f ab=%.3f lossGDP=%.3f"
            % (name, scn, m["CE"], m["PVL"], m["PVLc"], m["VOL"], m["J"], m["Jc"], m["V"], m["X"],
               m["CEc"], m["P30"], m["P40"], m["P54"], m["ab"], m["lossGDP"]))


def _get_nu(cfg, out_dir, econ, sm, log):
    path = os.path.join(out_dir, "main.pkl")
    if os.path.exists(path):
        return float(load_pickle(path)["nu_bar"])
    log("[nu] main.pkl not found: calibrating the budget multiplier")
    return calibrate_multiplier(econ, cfg, sm, econ.w, log)[0]


# ---------------------------------------------------------------------- main
def policy_map(econ, pol, cfg):
    """Feedback rule in the map year as a function of the cumulative-emission gap and the
    deviation of emission intensity from trend (state otherwise on the BAU trend)."""
    pm = cfg["policy_map"]
    t = int(pm["year"]) - cfg["economy"]["start_year"]
    gap = np.linspace(*pm["gap"][:2], int(pm["gap"][2]))
    dev = np.linspace(*pm["dev"][:2], int(pm["dev"][2]))
    G, D = np.meshgrid(gap, dev, indexing="ij")
    s = np.empty(G.shape + (5,))
    s[..., 0] = pm["p_prev"]
    s[..., 1] = econ.sigma_bar[t] * np.exp(D)
    s[..., 2] = econ.Y_bar[t]
    s[..., 3] = econ.Pb_bar[t]
    s[..., 4] = econ.B * t / econ.T + G
    return dict(gap=gap, dev=dev, price=pol(t, s))


def run_main(cfg, out_dir, log):
    econ, sm = setup(cfg)
    w = econ.w
    nu_bar, hist = calibrate_multiplier(econ, cfg, sm, w, log)
    log("[main] nu_bar=%.4f" % nu_bar)

    xi = scenario_shocks(cfg, SCEN, cfg["seed"])
    res, paths, r2, theta, times = {}, {}, {}, {}, {}
    pols = {}
    for name in METHOD_ORDER:
        spec = make_method(name, cfg)
        pol = train_policy(econ, spec, sm, w, nu_bar, cfg, cfg["seed"])
        pols[name] = pol
        res[name], paths[name] = {}, {}
        for scn in SCEN:
            sim = simulate(econ, pol, xi[scn])
            res[name][scn] = metrics(econ, cfg, sim)
            paths[name][scn] = path_record(sim, econ)
            log("[main] " + _line(name, scn, res[name][scn]))
        r2[name], theta[name], times[name] = pol.r2, pol.theta, pol.train_time

    rng = make_rng(cfg["seed"], 400)
    density = {s: draw_shocks(cfg, scenario_spec(cfg, s), 2000, econ.T, rng)
               .reshape(-1, 4).astype(np.float32) for s in SCEN}
    bau = econ.sigma_bar[:econ.T] * econ.Y_bar[:econ.T]        # BAU emissions, no abatement
    out = dict(cfg=cfg, record=sm.record, nu_bar=nu_bar, hist=hist, res=res, paths=paths,
               r2=r2, theta=theta, time=times, bau=bau, density=density,
               policy_map={n: policy_map(econ, pols[n], cfg) for n in ("SP", "DRMORL")})
    save_pickle(out, os.path.join(out_dir, "main.pkl"))

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        for name in METHOD_ORDER:
            for scn in SCEN:
                fh.write(_line(name, scn, res[name][scn]) + "\n")
        fh.write("nu_bar=%.4f hist=%s\ntime %s\n" % (nu_bar, hist, times))
    return out


# -------------------------------------------------------------------- pareto
def run_pareto(cfg, out_dir, log):
    """Weight sweep without budget multiplier (nu = 0): w = (w1, 0.9 - w1, w3)."""
    pc = cfg["pareto"]
    econ, sm = setup(cfg)
    xi = scenario_shocks(cfg, SCEN, cfg["seed"])
    res = {}
    for name in pc["methods"]:
        spec = make_method(name, cfg)
        res[name] = {}
        for w1 in pc["w1"]:
            w = np.array([w1, 1.0 - pc["w3"] - w1, pc["w3"]])
            pol = train_policy(econ, spec, sm, w, 0.0, cfg, cfg["seed"])
            res[name][w1] = {}
            txt = []
            for scn in pc["scenarios"]:
                m = metrics(econ, cfg, simulate(econ, pol, xi[scn]), w)
                res[name][w1][scn] = {k: m[k] for k in ("CE", "PVL", "VOL", "V", "X", "J")}
                txt.append("%s:%.1f/%.2f" % (scn, m["CE"], m["PVL"]))
            log("[pareto] %s w1=%.2f %s" % (name, w1, " ".join(txt)))
    out = dict(cfg=cfg, res=res)
    save_pickle(out, os.path.join(out_dir, "pareto.pkl"))
    return out


# ---------------------------------------------------------------------- sens
def run_sens(cfg, out_dir, log):
    """DR-MORL under alternative radius eps, CVaR level alpha and CVaR weight kappa."""
    econ, sm = setup(cfg)
    nu = _get_nu(cfg, out_dir, econ, sm, log)
    xi = scenario_shocks(cfg, SCEN, cfg["seed"])
    grids = dict(eps=cfg["sens"]["eps"], alpha=cfg["sens"]["alpha"], kappa=cfg["sens"]["kappa"])
    res = {}
    for par, vals in grids.items():
        res[par] = {}
        for v in vals:
            spec = make_method("DRMORL", cfg, **{par: float(v)})
            pol = train_policy(econ, spec, sm, econ.w, nu, cfg, cfg["seed"])
            res[par][v] = {s: metrics(econ, cfg, simulate(econ, pol, xi[s])) for s in SCEN}
            log("[sens] %s=%s " % (par, v) + " ".join(
                "%s:J%.2f/X%.2f" % (s, res[par][v][s]["J"], res[par][v][s]["X"]) for s in SCEN))
    out = dict(cfg=cfg, nu_bar=nu, res=res)
    save_pickle(out, os.path.join(out_dir, "sens.pkl"))
    return out


# -------------------------------------------------------------------- sample
def run_sample(cfg, out_dir, log):
    """(a) length N of the shock record; (b) number of training states per stage (n_s)."""
    sc = cfg["sample"]
    econ0, sm0 = setup(cfg)
    nu = _get_nu(cfg, out_dir, econ0, sm0, log)
    xi = scenario_shocks(cfg, ["S1"], cfg["seed"])["S1"]
    eps0 = cfg["risk"]["eps"]

    by_N = {}
    for N in sc["N"]:
        by_N[N] = {m: [] for m in sc["methods"]}
        M = max(cfg["shocks"]["M"], N + int(sc["extra_atoms"]))
        for rec in range(int(sc["n_records"])):
            econ, sm = setup(cfg, N=N, rec_key=rec)
            for name in sc["methods"]:
                over = dict(eps=eps0 * np.sqrt(sc["eps_ref_N"] / N), M=M) if name == "DRMORL" else {}
                spec = make_method(name, cfg, **over)
                pol = train_policy(econ, spec, sm, econ.w, nu, cfg, cfg["seed"], rep=rec)
                m = metrics(econ, cfg, simulate(econ, pol, xi))
                by_N[N][name].append([m["J"], m["Jc"], m["X"], m["V"]])
        for name in sc["methods"]:
            by_N[N][name] = np.asarray(by_N[N][name])           # (records, [J, Jc, X, V])
            log("[sample] N=%d %s %s" % (N, name, np.round(by_N[N][name].mean(0), 3)))

    by_ns = {}
    econ, sm = setup(cfg)
    for n_s in sc["n_s"]:
        by_ns[n_s] = {m: [] for m in sc["methods"]}
        for seed_k in range(int(sc["n_seeds"])):
            for name in sc["methods"]:
                spec = make_method(name, cfg)
                pol = train_policy(econ, spec, sm, econ.w, nu, cfg, cfg["seed"], n_s=n_s, rep=seed_k)
                m = metrics(econ, cfg, simulate(econ, pol, xi))
                by_ns[n_s][name].append([m["J"], m["Jc"], pol.train_time])
        for name in sc["methods"]:
            by_ns[n_s][name] = np.asarray(by_ns[n_s][name])      # (seeds, [J, Jc, time])
            log("[sample] n_s=%d %s %s" % (n_s, name, np.round(by_ns[n_s][name].mean(0), 3)))

    out = dict(cfg=cfg, nu_bar=nu, by_N=by_N, by_ns=by_ns)
    save_pickle(out, os.path.join(out_dir, "sample.pkl"))
    return out


PARTS = dict(main=run_main, pareto=run_pareto, sens=run_sens, sample=run_sample)
