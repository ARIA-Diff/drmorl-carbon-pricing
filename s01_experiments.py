"""Run the numerical experiments of the manuscript.

    python s01_experiments.py                       # all parts, paper settings
    python s01_experiments.py --parts main sens     # selected parts
    python s01_experiments.py --fast                # reduced sizes (smoke test only)

Parts: main (Sections 6.1-6.5, 6.9), pareto (6.6), sens (6.7), sample (6.8).
``sens`` and ``sample`` reuse the budget multiplier stored by ``main``.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drmorl.config import load_config            # noqa: E402
from drmorl.experiments import PARTS              # noqa: E402
from drmorl.utils import Logger                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=list(PARTS))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "results"))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, fast=args.fast, seed=args.seed)
    out_dir = os.path.abspath(args.out)
    log = Logger(os.path.join(out_dir, "exp_log.txt"))
    log("==== run %s parts=%s fast=%s budget=%.1f ====" % (
        time.strftime("%Y-%m-%d %H:%M:%S"), args.parts, args.fast, cfg["economy"]["budget"]))
    for part in [p for p in PARTS if p in args.parts]:      # keep main first
        t0 = time.time()
        PARTS[part](cfg, out_dir, log)
        log("---- part %s done in %.0fs" % (part, time.time() - t0))


if __name__ == "__main__":
    main()
