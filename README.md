# DR-MORL for adaptive carbon pricing

Code for a distributionally robust multi-objective reinforcement learning model of carbon pricing (Wasserstein ambiguity set, mean–CVaR risk, fitted Q-iteration).

```
pip install -r requirements.txt
python s01_experiments.py              # all parts -> ../results
python s01_experiments.py --parts main # run main first; sens and sample reuse its multiplier
python s03_tables.py                   # LaTeX tables from the results
```

- `config/default.yaml`: all parameters
- `drmorl/`: dynamics, shocks, robust targets, fitted Q-iteration, multiplier update, benchmarks, evaluation, experiments
- Parts: `main`, `pareto`, `sens`, `sample`; `--fast` runs a reduced smoke test

Parameters not specified in the paper are marked "assumed" in the YAML.
