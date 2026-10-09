# QuantumRisk: Classical and QAOA Benchmark

## 1. Experiment setup

- Assets considered: 8 ETFs.
- Portfolio constraint: select exactly 4 assets.
- Feasible portfolios evaluated classically: 70.
- Portfolio weights: equal weight (25% per selected asset).
- Quantum algorithm: QAOA using Qiskit Aer simulation.
- QAOA repetitions: 5.

## 2. Exact classical minimum-variance solution

- Selected assets: EFA, GLD, SPY, TLT.
- Daily variance: 5.115502115e-05.
- QAOA feasible runs: 5/5.
- QAOA runs matching the exact optimum: 5/5 (100.0%).
- Mean reported QAOA runtime: 0.5027 seconds.

## 3. Minimum-variance versus minimum-CVaR

| Metric | QAOA minimum-variance | Classical minimum-CVaR |
|---|---:|---:|
| Assets | EFA, GLD, SPY, TLT | EFA, SPY, TLT, USO |
| Annualized return estimate | 17.6891% | 17.3531% |
| Annualized volatility | 11.3539% | 11.5615% |
| Historical VaR 95% | 1.1062% | 1.0405% |
| Historical CVaR 95% | 1.6049% | 1.5250% |
| Maximum drawdown | 9.5329% | 12.0029% |

The minimum-CVaR portfolio's measured CVaR is 0.0799 percentage points lower than that of the minimum-variance portfolio under the shared risk engine.

## 4. Interpretation

- The classical exhaustive search provides the exact minimum-variance reference for this eight-asset, four-asset selection problem.
- QAOA found that exact portfolio in the recorded runs.
- Minimum variance and minimum CVaR are different objectives; their optimal portfolios need not match.
- The return figures are historical annualized estimates, not forecasts or guarantees.

## 5. Limitations

- QAOA was executed on a classical simulator, not quantum hardware.
- These results do not demonstrate quantum advantage.
- Five runs are a small sample and do not establish general reliability.
- Runtime measurements depend on the computer, software, optimizer settings, and measurement shots.
- The experiment uses historical data and equal-weight portfolios.
- The current QAOA experiment optimizes the variance-based QUBO; it does not directly optimize CVaR.

## 6. Conclusion

QuantumRisk demonstrates a working QUBO formulation, a QAOA simulation using Qiskit Aer, classical exhaustive benchmarking, and historical portfolio-risk evaluation. The next research step is to test larger problem instances and compare solution quality, runtime, and resource use under clearly controlled conditions.
