
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETURNS_FILE = PROJECT_ROOT / "data" / "daily_returns.csv"

NUMBER_OF_ASSETS_TO_SELECT = 4


# --------------------------------------------------
# Build a configurable QUBO model
# --------------------------------------------------

def build_qubo(
    asset_limit=None,
    number_of_assets_to_select=NUMBER_OF_ASSETS_TO_SELECT,
):
    """
    Build an equal-weight minimum-variance portfolio QUBO.

    Parameters
    ----------
    asset_limit : int or None
        Number of assets to include from daily_returns.csv.
        None uses all available assets.

    number_of_assets_to_select : int
        Exact number of assets to select.

    Returns
    -------
    problem, qubo, asset_names, covariance
    """

    if not RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Returns file not found: {RETURNS_FILE}"
        )

    if not isinstance(number_of_assets_to_select, int):
        raise TypeError(
            "number_of_assets_to_select must be an integer."
        )

    if number_of_assets_to_select < 1:
        raise ValueError(
            "At least one asset must be selected."
        )

    if asset_limit is not None:
        if not isinstance(asset_limit, int):
            raise TypeError("asset_limit must be an integer or None.")

        if asset_limit < 1:
            raise ValueError("asset_limit must be positive.")

    # Load returns and exclude date/time columns.
    data = pd.read_csv(RETURNS_FILE)

    date_columns = [
        column
        for column in data.columns
        if str(column).strip().lower()
        in ("date", "datetime", "timestamp")
    ]

    data = data.drop(columns=date_columns)

    returns = data.select_dtypes(include=[np.number])
    returns = returns.dropna(axis=1, how="all")
    returns = returns.dropna(axis=0, how="any")

    if returns.empty:
        raise ValueError("No valid numerical return data found.")

    # Restrict the asset universe for controlled experiments.
    if asset_limit is not None:
        if asset_limit > returns.shape[1]:
            raise ValueError(
                f"Requested {asset_limit} assets, but only "
                f"{returns.shape[1]} numerical asset columns exist."
            )

        returns = returns.iloc[:, :asset_limit]

    asset_names = list(returns.columns)
    number_of_assets = len(asset_names)

    if number_of_assets < number_of_assets_to_select:
        raise ValueError(
            f"Cannot select {number_of_assets_to_select} assets "
            f"from only {number_of_assets} available assets."
        )

    # --------------------------------------------------
    # Estimate the covariance matrix
    # --------------------------------------------------

    covariance = returns.cov().to_numpy(dtype=float)

    if not np.isfinite(covariance).all():
        raise ValueError(
            "Covariance matrix contains invalid values."
        )

    # Binary variable x[i] is 1 when asset i is selected.
    #
    # Equal-weight portfolio variance:
    #
    #     variance = x.T @ covariance @ x / k**2
    #
    # where k is the number of selected assets.

    k = number_of_assets_to_select
    quadratic_matrix = covariance / (k ** 2)

    # --------------------------------------------------
    # Construct the constrained binary problem
    # --------------------------------------------------

    problem = QuadraticProgram("QuantumRisk_Portfolio")

    for asset in asset_names:
        problem.binary_var(name=asset)

    problem.minimize(
        quadratic=quadratic_matrix
    )

    problem.linear_constraint(
        linear={asset: 1 for asset in asset_names},
        sense="==",
        rhs=k,
        name=f"select_exactly_{k}",
    )

    # --------------------------------------------------
    # Convert the constrained problem to QUBO
    # --------------------------------------------------

    converter = QuadraticProgramToQubo()
    qubo = converter.convert(problem)

    return problem, qubo, asset_names, covariance


# --------------------------------------------------
# Display the default model when run directly
# --------------------------------------------------

if __name__ == "__main__":
    problem, qubo, assets, covariance = build_qubo()

    print("=" * 60)
    print("QUANTUMRISK: QUBO MODEL")
    print("=" * 60)

    print(f"\nNumber of assets: {len(assets)}")
    print(f"Assets: {assets}")
    print(f"Assets to select: {NUMBER_OF_ASSETS_TO_SELECT}")

    print("\nOriginal constrained optimization problem:")
    print(problem.prettyprint())

    print("\nConverted QUBO problem:")
    print(qubo.prettyprint())

    print("\nCovariance matrix:")
    print(
        pd.DataFrame(
            covariance,
            index=assets,
            columns=assets,
        ).round(8)
    )

    print("\nQUBO model built successfully!")
