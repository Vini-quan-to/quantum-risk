
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from qiskit_aer import AerSimulator
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qubo_model import build_qubo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "data" / "quantum_portfolio_results.csv"

NUMBER_OF_ASSETS_TO_SELECT = 4
SHOTS = 1024


def main():
    print("=" * 60)
    print("QUANTUMRISK: QAOA PORTFOLIO OPTIMIZATION")
    print("=" * 60)

    # 1. Build the QUBO from the existing model.
    original_problem, qubo, assets, covariance = build_qubo()

    print(f"\nAssets: {assets}")
    print(f"Required selections: {NUMBER_OF_ASSETS_TO_SELECT}")

    # 2. Configure the local quantum simulator.
    simulator = AerSimulator()

    # 3. Configure QAOA.
    qaoa = QAOA(
        sampler=None,
        optimizer=COBYLA(maxiter=50),
        reps=1,
        initial_point=np.array([0.5, 0.5]),
    )

    # Qiskit Algorithms versions may differ in how the sampler
    # is configured. Use the Qiskit-compatible sampler if needed.
    from qiskit.primitives import StatevectorSampler

    sampler = StatevectorSampler(default_shots=SHOTS)

    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=50),
        reps=1,
        initial_point=np.array([0.5, 0.5]),
    )

    # 4. Solve the QUBO.
    solver = MinimumEigenOptimizer(qaoa)

    print("\nRunning QAOA. Please wait...")
    result = solver.solve(qubo)

    print("\nQAOA optimization completed.")
    print("Best measured solution:")
    print(result)

    # 5. Decode the best solution.
    bit_values = np.rint(result.x).astype(int)
    selected_assets = [
        assets[i]
        for i, bit in enumerate(bit_values)
        if bit == 1
    ]

    print("\nSelected assets:", selected_assets)
    print("Number selected:", len(selected_assets))
    print("QUBO objective value:", result.fval)

    if len(selected_assets) != NUMBER_OF_ASSETS_TO_SELECT:
        print(
            "\nWARNING: The best solution does not select exactly "
            "four assets. We will inspect feasibility and penalties."
        )

    # 6. Evaluate historical risk for the selected portfolio.
    returns_path = PROJECT_ROOT / "data" / "daily_returns.csv"
    returns = pd.read_csv(returns_path)

    date_columns = [
        column
        for column in returns.columns
        if str(column).strip().lower()
        in ("date", "datetime", "timestamp")
    ]
    returns = returns.drop(columns=date_columns)
    returns = returns.select_dtypes(include=[np.number])
    returns = returns.dropna(axis=0, how="any")

    portfolio_returns = returns[selected_assets].mean(axis=1)
    losses = -portfolio_returns.to_numpy()

    var_95 = np.quantile(losses, 0.95)
    cvar_95 = losses[losses >= var_95].mean()

    daily_mean = portfolio_returns.mean()
    daily_volatility = portfolio_returns.std(ddof=1)

    # Approximate annualization using 252 trading days.
    annualized_return = daily_mean * 252
    annualized_volatility = daily_volatility * np.sqrt(252)

    running_peak = (1 + portfolio_returns).cumprod().cummax()
    wealth = (1 + portfolio_returns).cumprod()
    drawdowns = 1 - wealth / running_peak
    maximum_drawdown = drawdowns.max()

    print("\nHistorical portfolio evaluation:")
    print(f"Mean daily return:       {daily_mean:.6%}")
    print(f"Annualized mean return:  {annualized_return:.6%}")
    print(f"Annualized volatility:   {annualized_volatility:.6%}")
    print(f"Historical VaR 95%:      {var_95:.6%}")
    print(f"Historical CVaR 95%:     {cvar_95:.6%}")
    print(f"Maximum drawdown:        {maximum_drawdown:.6%}")

    # 7. Save the result.
    output = pd.DataFrame(
        {
            "asset": selected_assets,
            "weight": [
                1 / len(selected_assets)
                for _ in selected_assets
            ],
        }
    )

    output.to_csv(RESULTS_FILE, index=False)

    print(f"\nSelected assets saved to: {RESULTS_FILE}")
    print("\nQuantumRisk run finished.")


if __name__ == "__main__":
    main()
