"""Write the LaTeX tables of the manuscript from the stored experiment output.

    python s03_tables.py [--results ../results] [--out ../paper/tables]

tab_main.tex, tab_scen.tex, tab_ablation.tex  <- main.pkl
tab_eps.tex                                   <- sens.pkl
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drmorl.methods import METHOD_LABEL, METHOD_ORDER   # noqa: E402
from drmorl.utils import load_pickle                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SC = ["S1", "S2", "S3", "S4", "S5"]


def bf(txt, best):
    return r"\textbf{%s}" % txt if best else txt


def tab_main(res, times):
    cols = [("CE", "%.1f"), ("PVL", "%.2f"), ("PVLc", "%.2f"), ("VOL", "%.2f"),
            ("J", "%.2f"), ("V", "%.3f"), ("X", "%.2f")]
    best = {k: min(res[m]["S1"][k] for m in METHOD_ORDER) for k, _ in cols}
    rows = []
    for m in METHOD_ORDER:
        cells = [bf(f % res[m]["S1"][k], res[m]["S1"][k] == best[k]) for k, f in cols]
        rows.append(" & ".join([METHOD_LABEL[m]] + cells + ["%.1f" % times[m]]) + r" \\")
    return rows


def tab_scen(res):
    rows = [r"\multicolumn{6}{l}{\emph{Violation probability / expected overshoot}} \\"]
    bv = {s: min(res[m][s]["V"] for m in METHOD_ORDER) for s in SC}
    bx = {s: min(res[m][s]["X"] for m in METHOD_ORDER) for s in SC}
    for m in METHOD_ORDER:
        cells = ["%s / %s" % (bf("%.3f" % res[m][s]["V"], res[m][s]["V"] == bv[s]),
                              bf("%.2f" % res[m][s]["X"], res[m][s]["X"] == bx[s])) for s in SC]
        rows.append(" & ".join([METHOD_LABEL[m]] + cells) + r" \\")
    rows += [r"\midrule",
             r"\multicolumn{6}{l}{\emph{Scalarised cost $J$ / CVaR$_{0.95}$ of cumulative emissions}} \\"]
    for m in METHOD_ORDER:
        cells = ["%.2f / %.1f" % (res[m][s]["J"], res[m][s]["CEc"]) for s in SC]
        rows.append(" & ".join([METHOD_LABEL[m]] + cells) + r" \\")
    return rows


def tab_ablation(res):
    """Percentage change of the scenario-averaged (S2-S5) measures relative to SP."""
    sc = SC[1:]
    avg = lambda m, k: np.mean([res[m][s][k] for s in sc])
    rows = []
    for m, w, c in (("CVaR", "--", r"\checkmark"), ("DR", r"\checkmark", "--"),
                    ("DRMORL", r"\checkmark", r"\checkmark")):
        d = [100.0 * (avg(m, k) / avg("SP", k) - 1.0) for k in ("J", "PVL", "V", "X", "CEc")]
        rows.append(" & ".join([METHOD_LABEL[m], w, c] + ["%+.1f" % x for x in d]) + r" \\")
    return rows


def tab_eps(sens):
    sc = ["S3", "S4", "S5"]
    rows = []
    for e, r in sorted(sens["res"]["eps"].items()):
        rows.append(" & ".join([
            ("%g" % e), "%.2f" % r["S1"]["J"], "%.2f" % r["S1"]["PVL"], "%.3f" % r["S1"]["V"],
            "%.3f" % np.mean([r[s]["V"] for s in sc]), "%.2f" % np.mean([r[s]["X"] for s in sc]),
            "%.2f" % np.mean([r[s]["J"] for s in SC])]) + r" \\")
    return rows


def write(path, caption, label, colspec, header, rows, mid_after=None):
    body = [r"\begin{table}[H]", r"\begin{adjustwidth}{-\extralength}{0cm}",
            r"\caption{%s\label{%s}}" % (caption, label),
            r"\begin{tabularx}{\fulllength}{%s}" % colspec, r"\toprule", header, r"\midrule"]
    body += rows + [r"\bottomrule", r"\end{tabularx}", r"\end{adjustwidth}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(body) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "paper", "tables"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    main_ = load_pickle(os.path.join(a.results, "main.pkl"))
    res = main_["res"]

    write(os.path.join(a.out, "tab_main.tex"),
          r"Out-of-sample performance under the nominal process (S1). CE: mean cumulative emissions (Gt); "
          r"PVL: present value of output losses (trillion USD); VOL: mean absolute price change (USD/t per year); "
          r"$J$: discounted scalarised cost; Viol.: budget violation probability; Exc.: expected overshoot (Gt); "
          r"Time: training time (s). Best values in bold.", "tab:main", "lCCCCCCCC",
          r"\textbf{Method} & \textbf{CE} & \textbf{PVL} & \textbf{CVaR(PVL)} & \textbf{VOL} & "
          r"\boldmath$J$ & \textbf{Viol.} & \textbf{Exc.} & \textbf{Time} \\",
          tab_main(res, main_["time"]))

    write(os.path.join(a.out, "tab_scen.tex"),
          r"Robustness across uncertainty structures. Upper panel: budget violation probability / expected "
          r"overshoot (Gt). Lower panel: mean scalarised cost $J$ / CVaR$_{0.95}$ of cumulative emissions (Gt). "
          r"Best values in bold.", "tab:scen_res", "lCCCCC",
          r"\textbf{Method} & \textbf{S1} & \textbf{S2} & \textbf{S3} & \textbf{S4} & \textbf{S5} \\",
          tab_scen(res))

    write(os.path.join(a.out, "tab_ablation.tex"),
          r"Ablation study: percentage change relative to SP, averaged over the misspecified scenarios S2--S5.",
          "tab:ablation", "lCCCCCCC",
          r"\textbf{Variant} & \textbf{Wasserstein} & \textbf{CVaR} & \boldmath$\Delta J$ & "
          r"\boldmath$\Delta$\textbf{PVL} & \boldmath$\Delta$\textbf{Viol.} & \boldmath$\Delta$\textbf{Exc.} & "
          r"\boldmath$\Delta$\textbf{CVaR(CE)} \\", tab_ablation(res))

    sens_path = os.path.join(a.results, "sens.pkl")
    if os.path.exists(sens_path):
        write(os.path.join(a.out, "tab_eps.tex"),
              r"Effect of the Wasserstein radius $\varepsilon$ ($\alpha=0.9$, $\kappa=0.5$).", "tab:eps", "lCCCCCC",
              r"\boldmath$\varepsilon$ & \boldmath$J$ \textbf{(S1)} & \textbf{PVL (S1)} & \textbf{Viol. (S1)} & "
              r"\textbf{Viol. (S3--S5)} & \textbf{Exc. (S3--S5)} & \boldmath$J$ \textbf{(all)} \\",
              tab_eps(load_pickle(sens_path)))


if __name__ == "__main__":
    main()
