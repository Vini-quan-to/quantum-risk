
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from qubo_model import build_qubo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETURNS_FILE = PROJECT_ROOT / "data" / "daily_returns.csv"

ASSETS_TO_SELECT = 4
TRADING_DAYS = 252


def load_returns():
    data = pd.read_csv(RETURNS_FILE)

    date_columns = [
        column
        for column in data.columns
        if str(column).strip().lower()
        in ("date", "datetime", "timestamp")
    ]

    data = data.drop(columns=date_columns)
    data = data.select_dtypes(include=[np.number])
    data = data.dropna(axis=0, how="any")

    return data


def main():
    print("=" * 65)
    print("QUANTUMRISK: QUBO MATHEMATICAL VALIDATION")
    print("=" * 65)

    # Build the exact same model used by QAOA.
    original_problem, qubo, assets, covariance = build_qubo()

    returns = load_returns()
    assets = list(returns.columns)

    # Respect the variable order in the converted QUBO.
    qubo_variables = [variable.name for variable in qubo.variables]

    if set(qubo_variables) != set(assets):
        raise ValueError(
            "QUBO variables differ from the asset columns. "
            f"QUBO: {qubo_variables}; assets: {assets}"
        )

    covariance = returns.cov().to_numpy()
    asset_indices = {asset: i for i, asset in enumerate(assets)}

    results = []

    # Enumerate every feasible four-asset portfolio.
    for selected in combinations(assets, ASSETS_TO_SELECT):
        selected_set = set(selected)

        # Binary vector in the original asset order.
        x = np.array(
            [1 if asset in selected_set else 0 for asset in assets],
            dtype=float,
        )

        # Binary vector in the QUBO's variable order.
        qubo_x = np.array(
            [
                1 if variable in selected_set else 0
                for variable in qubo_variables
            ],
            dtype=float,
        )

        # Direct covariance-based portfolio variance.
        # Equal weight = 1/4 for each selected asset.
        direct_variance = float(
            x @ covariance @ x / ASSETS_TO_SELECT**2
        )

        # Energy evaluated using Qiskit's converted QUBO.
        qubo_energy = float(qubo.objective.evaluate(qubo_x))

        results.append(
            {
                "assets": ", ".join(selected),
                "direct_variance": direct_variance,
                "qubo_energy": qubo_energy,
                "difference": qubo_energy - direct_variance,
            }
        )

    results_df = pd.DataFrame(results)

    # Compare the ranking of all feasible portfolios.
    variance_ranking = results_df["direct_variance"].rank(
        method="min"
    )
    qubo_ranking = results_df["qubo_energy"].rank(
        method="min"
    )

    max_abs_difference = results_df["difference"].abs().max()

    # A constant energy offset is allowed because it doesn't change
    # the ranking or the minimizing portfolio.
    energy_offsets = (
        results_df["qubo_energy"]
        - results_df["direct_variance"]
    )
    offset_spread = energy_offsets.max() - energy_offsets.min()

    variance_winner = results_df.loc[
        results_df["direct_variance"].idxmin()
    ]
    qubo_winner = results_df.loc[
        results_df["qubo_energy"].idxmin()
    ]

    same_winner = set(variance_winner["assets"].split(", ")) == set(
        qubo_winner["assets"].split(", ")
    )

    same_ranking = variance_ranking.equals(qubo_ranking)

    print(f"\nAssets: {len(assets)}")
    print(f"Feasible portfolios checked: {len(results_df)}")
    print(f"Expected combinations: {len(list(combinations(assets, 4)))}")

    print("\n--- MINIMUM BY DIRECT VARIANCE ---")
    print(f"Assets: {variance_winner['assets']}")
    print(f"Daily variance: {variance_winner['direct_variance']:.12g}")

    print("\n--- MINIMUM BY QUBO ENERGY ---")
    print(f"Assets: {qubo_winner['assets']}")
    print(f"QUBO energy: {qubo_winner['qubo_energy']:.12g}")

    print("\n--- VALIDATION CHECKS ---")
    print(
        "Maximum absolute energy difference: "
        f"{max_abs_difference:.12g}"
    )
    print(
        "Spread of (QUBO energy - variance): "
        f"{offset_spread:.12g}"
    )
    print(f"Same minimizing portfolio: {same_winner}")
    print(f"Same complete ranking: {same_ranking}")

    # Save the detailed results for inspection.
    output_path = PROJECT_ROOT / "data" / "qubo_validation.csv"
    results_df.sort_values("direct_variance").to_csv(
        output_path, index=False
    )

    print(f"\nDetailed results saved to: {output_path}")

    if offset_spread < 1e-10 and same_winner:
        print("\nPASS: QUBO energy matches variance up to a constant offset.")
    elif same_winner:
        print(
            "\nPARTIAL PASS: Both objectives select the same winner, "
            "but inspect the energy differences and ranking."
        )
    else:
        print(
            "\nCHECK REQUIRED: The QUBO and direct variance objectives "
            "select different portfolios."
        )


if __name__ == "__main__":
    main()

