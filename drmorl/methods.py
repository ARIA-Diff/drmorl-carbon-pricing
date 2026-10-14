"""Definition of the compared methods, Section 5.3.

All methods share the structural model, the weights, the feature map, the action
grid, the number of training states and the budget multiplier; they differ only in
how uncertainty enters the Bellman targets.

    DET     certainty equivalent: shocks replaced by their empirical mean
    SP      risk-neutral SDP under a fitted Gaussian, 60 sampled scenarios per stage
    RL      model-free risk-neutral FQI, one observed transition per training sample
    CVaR    mean-CVaR under the empirical distribution (eps = 0, kappa = 0.5)
    DR      Wasserstein-robust risk-neutral targets (eps = 0.1, kappa = 0)
    DRMORL  Wasserstein-robust mean-CVaR targets (eps = 0.1, alpha = 0.9, kappa = 0.5)
"""
from dataclasses import dataclass, replace

METHOD_ORDER = ["DET", "SP", "RL", "CVaR", "DR", "DRMORL"]
METHOD_LABEL = {"DET": "DET", "SP": "SP", "RL": "RL", "CVaR": "CVaR-RL",
                "DR": "DR-RL", "DRMORL": "DR-MORL"}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    support: str          # 'mean' | 'gaussian' | 'single' | 'augmented'
    eps: float            # Wasserstein radius
    alpha: float          # CVaR level
    kappa: float          # CVaR weight
    offset: bool          # Q = known nominal stage cost + learned residual (model-informed)
    M: int                # number of candidate atoms (augmented support)


def make_method(name, cfg, **over):
    r = cfg["risk"]
    M = cfg["shocks"]["M"]
    base = {
        "DET": MethodSpec("DET", "mean", 0.0, r["alpha"], 0.0, True, 1),
        "SP": MethodSpec("SP", "gaussian", 0.0, r["alpha"], 0.0, True, cfg["algo"]["sp_scenarios"]),
        "RL": MethodSpec("RL", "single", 0.0, r["alpha"], 0.0, False, 1),
        "CVaR": MethodSpec("CVaR", "augmented", 0.0, r["alpha"], r["kappa"], True, M),
        "DR": MethodSpec("DR", "augmented", r["eps"], r["alpha"], 0.0, True, M),
        "DRMORL": MethodSpec("DRMORL", "augmented", r["eps"], r["alpha"], r["kappa"], True, M),
    }[name]
    return replace(base, **over) if over else base
