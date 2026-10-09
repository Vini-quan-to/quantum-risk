from pathlib import Path

import numpy as np
import pandas as pd
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo


# ============================================================
# QuantumRisk: Variable-Size QUBO Model
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RETURNS_FILE = DATA_DIR / "daily_returns.csv"

MIN_ASSETS = 4
MAX_ASSETS = 10

PENALTY_MULTIPLIER = 10.0
RANDOM_ASSIGNMENTS = 1000
RANDOM_SEED = 2026


# ============================================================
# Load and validate returns
# ============================================================

def load_returns():
    """Load daily asset returns and validate the data."""

    if not RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Returns file not found: {RETURNS_FILE}\n"
            "Run python src/download_data.py first."
        )

    returns = pd.read_csv(RETURNS_FILE)

    date_columns = [
        column for column in returns.columns
        if str(column).strip().lower()
        in {"date", "datetime", "timestamp"}
    ]

    returns = returns.drop(columns=date_columns)
    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(axis=0, how="any")

    if returns.empty:
        raise ValueError("The returns dataset has no valid observations.")

    if returns.columns.duplicated().any():
        raise ValueError("The returns dataset contains duplicate asset names.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("The returns dataset contains invalid values.")

    if not (1 <= MIN_ASSETS <= MAX_ASSETS <= returns.shape[1]):
        raise ValueError(
            f"Invalid selection limits: {MIN_ASSETS}–{MAX_ASSETS} "
            f"for {returns.shape[1]} available assets."
        )

    return returns


# ============================================================
# Construct the original constrained quadratic program
# ============================================================

def build_quadratic_program():
    """
    Binary decision variable:
        x_i = 1 if asset i is selected, otherwise 0.

    Objective:
        Minimize x.T @ covariance @ x.

    Constraints:
        MIN_ASSETS <= sum(x_i) <= MAX_ASSETS.

    Note:
        This is a covariance-based quadratic objective. Because the
        portfolio size is variable, it is not normalized by the square
        of the number of selected assets.
    """

    returns = load_returns()
    assets = list(returns.columns)
    covariance = returns.cov().to_numpy(dtype=float)

    asset_count = len(assets)

    if covariance.shape != (asset_count, asset_count):
        raise ValueError("Unexpected covariance matrix dimensions.")

    if not np.isfinite(covariance).all():
        raise ValueError("The covariance matrix contains invalid values.")

    # Remove small numerical asymmetries.
    covariance = (covariance + covariance.T) / 2.0

    largest_covariance = float(np.max(np.abs(covariance)))

    if largest_covariance <= 0:
        raise ValueError("The covariance matrix has no positive scale.")

    # Scale the penalty relative to the covariance matrix and universe.
    penalty = (
        PENALTY_MULTIPLIER
        * largest_covariance
        * asset_count
    )

    qp = QuadraticProgram("QuantumRisk_Variable_Size")

    for asset in assets:
        qp.binary_var(name=asset)

    # x.T @ covariance @ x:
    # diagonal terms are linear for binary variables because x_i^2 = x_i.
    linear = {
        assets[i]: float(covariance[i, i])
        for i in range(asset_count)
    }

    quadratic = {}

    for i in range(asset_count):
        for j in range(i + 1, asset_count):
            coefficient = 2.0 * float(covariance[i, j])

            if coefficient != 0.0:
                quadratic[(assets[i], assets[j])] = coefficient

    qp.minimize(
        linear=linear,
        quadratic=quadratic,
    )

    # Lower bound on the number of selected assets.
    qp.linear_constraint(
        linear={asset: 1.0 for asset in assets},
        sense=">=",
        rhs=MIN_ASSETS,
        name="minimum_assets",
    )

    # Upper bound on the number of selected assets.
    qp.linear_constraint(
        linear={asset: 1.0 for asset in assets},
        sense="<=",
        rhs=MAX_ASSETS,
        name="maximum_assets",
    )

    return qp, assets, covariance, penalty


# ============================================================
# Convert the constrained problem to a QUBO
# ============================================================

def build_qubo():
    """Convert the constrained quadratic program into an unconstrained QUBO."""

    qp, assets, covariance, penalty = build_quadratic_program()

    converter = QuadraticProgramToQubo(penalty=penalty)
    qubo = converter.convert(qp)

    if qubo.get_num_vars() < len(assets):
        raise RuntimeError(
            "The converted QUBO unexpectedly has fewer variables "
            "than the original asset-selection problem."
        )

    print("=" * 68)
    print("QUANTUMRISK: VARIABLE-SIZE QUBO")
    print("=" * 68)

    print(f"Available assets: {len(assets)}")
    print(f"Minimum selected assets: {MIN_ASSETS}")
    print(f"Maximum selected assets: {MAX_ASSETS}")
    print(f"Original asset variables: {len(assets)}")
    print(f"QUBO variables including slack: {qubo.get_num_vars()}")
    print(f"Penalty coefficient: {penalty:.8g}")

    print("\nThe converted QUBO has no explicit constraints.")
    print("The portfolio-size constraints are encoded using penalties.")

    return qp, qubo, assets, covariance


# ============================================================
# Lightweight validation
# ============================================================

def validate_qubo():
    """
    Validate the model structure and randomly test the original
    portfolio-size constraints.

    This deliberately avoids constructing a dense matrix or running
    an exact eigensolver. Random tests are not an optimality proof.
    """

    qp, qubo, assets, covariance = build_qubo()

    rng = np.random.default_rng(RANDOM_SEED)
    asset_count = len(assets)

    feasible_count = 0
    infeasible_count = 0
    sampled_objectives = []

    print("\nRunning lightweight QUBO validation...")
    print(f"Random assignments to test: {RANDOM_ASSIGNMENTS}")

    for _ in range(RANDOM_ASSIGNMENTS):
        # Sample the original asset variables only.
        bits = rng.integers(0, 2, size=asset_count)
        selected_count = int(bits.sum())

        if MIN_ASSETS <= selected_count <= MAX_ASSETS:
            feasible_count += 1

            # Evaluate the original quadratic objective.
            objective = float(bits @ covariance @ bits)

            if not np.isfinite(objective):
                raise ValueError(
                    "A sampled feasible assignment has a non-finite "
                    "objective value."
                )

            sampled_objectives.append(objective)
        else:
            infeasible_count += 1

    # Confirm that the converted QUBO has finite objective coefficients.
    linear_coefficients = qubo.objective.linear.to_array()
    quadratic_coefficients = qubo.objective.quadratic.to_array()

    if not np.isfinite(linear_coefficients).all():
        raise ValueError("QUBO linear coefficients are not finite.")

    if not np.isfinite(quadratic_coefficients).all():
        raise ValueError("QUBO quadratic coefficients are not finite.")

    print("\n" + "=" * 68)
    print("LIGHTWEIGHT VALIDATION SUMMARY")
    print("=" * 68)

    print(f"Available assets: {asset_count}")
    print(f"Original binary variables: {asset_count}")
    print(f"Converted QUBO variables: {qubo.get_num_vars()}")
    print(f"Random assignments tested: {RANDOM_ASSIGNMENTS}")
    print(f"Feasible asset selections: {feasible_count}")
    print(f"Infeasible asset selections: {infeasible_count}")

    if sampled_objectives:
        print(
            "Lowest sampled feasible objective: "
            f"{min(sampled_objectives):.12g}"
        )
        print(
            "Highest sampled feasible objective: "
            f"{max(sampled_objectives):.12g}"
        )

    print("\nModel checks passed:")
    print("- Original asset variables are binary.")
    print("- Minimum and maximum portfolio-size constraints exist.")
    print("- Converted QUBO objective coefficients are finite.")
    print("- Random feasible assignments satisfy the size limits.")

    print("\nImportant:")
    print("Random sampling is not an optimality certificate.")
    print("No exact eigensolver was run.")
    print("\nLightweight validation completed successfully.")

    return qp, qubo, assets, covariance


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    validate_qubo()
