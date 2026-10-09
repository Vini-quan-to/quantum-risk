# QuantumRisk

## 1. Project overview

QuantumRisk investigates a portfolio-selection problem using a Quadratic Unconstrained Binary Optimization (QUBO) model and the Quantum Approximate Optimization Algorithm (QAOA).

The portfolio contains exactly four equally weighted ETFs. The objective is to minimize historical portfolio variance. A classical exhaustive search provides a reference solution.

## 2. Data and methodology

- Assets: SPY, QQQ, IWM, EFA, EEM, TLT, GLD, and USO.
- Data: historical daily asset returns stored in `daily_returns.csv`.
- Portfolio constraint: select exactly four assets.
- Portfolio weights: 25% per selected asset.
- Risk objective: minimize estimated daily portfolio variance.
- Quantum method: QAOA using Qiskit and a local Aer simulator.
- Classical baseline: exhaustive enumeration of feasible portfolios.

For a selected-asset vector x and covariance matrix Sigma, the objective is x^T Sigma x / 16, subject to sum(x_i) = 4 and x_i in {0, 1}.

## 3. QUBO validation

- Feasible portfolios checked: 70.
- The detailed validation is available in `qubo_validation.csv`.
- The QUBO was validated against direct portfolio-variance calculations in the eight-asset model.

## 4. Quantum versus classical risk

| method                           | assets             |   qaoa_runs |   mean_daily_return |   annualized_return_estimate |   annualized_volatility |    VaR_95 |   CVaR_95 |   max_drawdown |
|:---------------------------------|:-------------------|------------:|--------------------:|-----------------------------:|------------------------:|----------:|----------:|---------------:|
| QAOA minimum-variance portfolio  | EFA, GLD, SPY, TLT |           5 |         0.000701948 |                     0.176891 |                0.113539 | 0.0110622 | 0.0160488 |      0.0953289 |
| Classical minimum-CVaR portfolio | EFA, SPY, TLT, USO |           0 |         0.000688617 |                     0.173531 |                0.115615 | 0.0104048 | 0.0152496 |      0.120029  |

## 5. QAOA Aer experiment

- Exact optimum found in 5 of 5 recorded runs.
- Mean recorded QAOA runtime: 0.5027 seconds.

Full results: `qaoa_aer_results.csv`.

## 6. Scalability experiment

|   asset_count |   feasible_combinations | matched_exact_optimum   |   relative_variance_gap |   qaoa_runtime_seconds |   classical_runtime_seconds |
|--------------:|------------------------:|:------------------------|------------------------:|-----------------------:|----------------------------:|
|             5 |                       5 | True                    |                       0 |               0.359838 |                   9.83e-05  |
|             6 |                      15 | True                    |                       0 |               0.371346 |                   0.0002588 |
|             7 |                      35 | True                    |                       0 |               0.387355 |                   0.0005258 |
|             8 |                      70 | True                    |                       0 |               0.720785 |                   0.0008476 |

These are small pilot instances. The results do not establish quantum advantage; the classical search is expected to be very effective at these problem sizes.

## 7. Visualizations

- `figures/risk_comparison.png`
- `figures/return_volatility.png`
- `figures/qaoa_reliability.png`

## 8. Limitations

- Results depend on the historical sample and covariance estimate.
- Historical risk metrics do not guarantee future performance.
- The portfolio uses equal weights rather than optimized continuous weights.
- QAOA results depend on circuit depth, optimizer settings, sampling, and penalty strength.
- Small simulated instances do not demonstrate quantum speedup.

## 9. Conclusion

QuantumRisk demonstrates an end-to-end workflow for encoding a constrained portfolio-selection problem as QUBO, solving it with QAOA, validating the objective against classical enumeration, and comparing historical portfolio risk. Further experiments with larger instances, repeated seeds, and stronger classical baselines are required before drawing conclusions about scalability or quantum advantage.

## Reproducibility

Run the project scripts from the repository root using the project's Python virtual environment. The `data/` directory contains the experiment outputs and `figures/` contains the generated plots.
