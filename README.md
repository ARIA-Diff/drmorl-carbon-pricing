# DR-MORL for adaptive carbon pricing — code

Implementation of the method and experiments of the manuscript
*A Distributionally Robust Multi-Objective Reinforcement Learning Model for Adaptive Carbon Pricing under Economic and Environmental Uncertainty*.
No plotting or verification scripts are included.

## Layout

| File | Content (manuscript reference) |
|---|---|
| `config/default.yaml` | All parameters (Table 1, Table 2, Section 5) |
| `drmorl/dynamics.py` | State, action set, abatement, costs, transition, terminal cost (Eqs. 1–8, 11) |
| `drmorl/shocks.py` | Shock process, scenarios S1–S5, candidate support, standardised transport cost (Eq. 9, Sec. 5.2) |
| `drmorl/risk.py` | Worst-case mean–CVaR over the Wasserstein ball via the dual of Lemma 1 (Eq. 20) |
| `drmorl/fqi.py` | 31-feature model-informed fitted Q-iteration, ridge fit, greedy rule (Eqs. 18–22, Algorithm 1) |
| `drmorl/dual.py` | Projected supergradient update of the budget multiplier (Prop. 3, Algorithm 1) |
| `drmorl/methods.py` | DET, SP, RL, CVaR-RL, DR-RL, DR-MORL (Sec. 5.3) |
| `drmorl/evaluate.py` | Simulation with common random numbers and performance measures (Sec. 5.4) |
| `drmorl/experiments.py` | Parts `main`, `pareto`, `sens`, `sample` |
| `s01_experiments.py` | Command-line entry point |
| `s03_tables.py` | Writes `paper/tables/*.tex` from the stored results |

## Usage

```
pip install -r requirements.txt
python s01_experiments.py                  # all parts -> ../results/*.pkl, exp_log.txt, summary.txt
python s01_experiments.py --parts main     # main must run first (it stores the multiplier nu_bar)
python s03_tables.py                       # -> ../paper/tables
```

`--fast` shrinks sample sizes for a smoke test; its numbers are not those of the paper.

## Parts

* `main` — multiplier calibration on SP, six methods × S1–S5, stage-wise R², price/emission paths, density samples, feedback-rule map for 2035 (`main.pkl`, `summary.txt`).
* `pareto` — weight sweep w = (w1, 0.9 − w1, 0.1), ν = 0, DET/SP/RL/DR-MORL on S1, S3, S4 (`pareto.pkl`).
* `sens` — DR-MORL over ε, α, κ on S1–S5 (`sens.pkl`).
* `sample` — record length N ∈ {10,…,80} with ε_N = 0.1·√(30/N) and the number of training states per stage (`sample.pkl`).

## Choices not reported in the manuscript (assumed, editable in the YAML)

Master seed; growth-innovation s.d. inside a regime (1.0 pp, recession drop centred so the nominal mean is zero); bounded support (clipping) of the shocks; ridge parameter ς = 1e‑5; η grid = 8 empirical quantiles in [0.5, 1]; λ grid = 14 log-spaced multiples (10^-2…10^1) of the target spread; exploratory states from a uniformly random price revision with shocks resampled from the record (fitted Gaussian for SP); 2000 trajectories for the multiplier estimate; M = max(60, N+30) in the record-length experiment; model-free RL regresses on the features without the known nominal-cost term. Results therefore reproduce the design of the paper, not necessarily its digits.
