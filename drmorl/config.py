"""Configuration loading."""
import copy
import os

import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config", "default.yaml")


def deep_update(base, new):
    """Recursively merge ``new`` into ``base`` (returns a new dict)."""
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path=None, fast=False, seed=None):
    with open(path or DEFAULT_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    fast_over = cfg.pop("fast", {})
    if fast:
        cfg = deep_update(cfg, fast_over)
    if seed is not None:
        cfg["seed"] = int(seed)
    cfg["fast_mode"] = bool(fast)
    return cfg
