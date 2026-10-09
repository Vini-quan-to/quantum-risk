
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
        column
        for column in returns.columns
        if str(column).strip().lower()
        in {"date", "datetime", "timestamp"}
    ]

    returns = returns.drop(columns=date_columns)
    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(axis=0, how="any")

    if returns.empty:
        raise ValueError(
            "The returns dataset has no valid observations."
        )

    if returns.columns.duplicated().any():
        raise ValueError(
            "The returns dataset contains duplicate asset names."
        )

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError(
            "The returns dataset contains invalid values."
        )

    asset_count = returns.shape[1]

    if not (1 <= MIN_ASSETS <= MAX_ASSETS <= asset_count):
        raise ValueError(
            f"Invalid selection limits: {MIN_ASSETS}–{MAX_ASSETS} "
            f"for {asset_count} available assets."
        )

    if len(returns) < 2:
        raise ValueError(
            "At least two valid return observations are required."
        )

    return returns


# ============================================================
# Covariance matrix
# ============================================================

def calculate_covariance(returns):
    """Calculate a finite, symmetric sample covariance matrix."""

    covariance = returns.cov().to_numpy(dtype=float)

    if covariance.shape != (returns.shape[1], returns.shape[1]):
        raise ValueError(
            "Unexpected covariance matrix dimensions."
        )

    if not np.isfinite(covariance).all():
        raise ValueError(
            "The covariance matrix contains invalid values."
        )

    # Remove tiny floating-point asymmetries.
    covariance = (covariance + covariance.T) / 2.0

    if np.max(np.abs(covariance)) <= 0:
        raise ValueError(
            "The covariance matrix has no positive scale."
        )

    return covariance


# ============================================================
# Objective evaluation
# ============================================================

def evaluate_covariance_objective(bits, covariance):
    """
    Evaluate the current QUBO objective: x.T @ covariance @ x.

    This is an unnormalized selected-asset covariance objective.
    It is NOT equal-weight portfolio variance when portfolio size
    varies.

    For a selected portfolio of size k, equal-weight variance is:

        (x.T @ covariance @ x) / k**2

    for k > 0.
    """

    bits = np.asarray(bits, dtype=float)

    if bits.ndim != 1:
        raise ValueError("Asset selection must be a one-dimensional array.")

    if bits.size != covariance.shape[0]:
        raise ValueError(
            "Asset selection length does not match covariance dimensions."
        )

    if not np.isin(bits, [0.0, 1.0]).all():
        raise ValueError("Asset selection must contain only binary values.")

    return float(bits @ covariance @ bits)


def evaluate_equal_weight_variance(bits, covariance):
    """Evaluate the actual variance of an equal-weight selected portfolio."""

    bits = np.asarray(bits, dtype=float)

    if bits.ndim != 1:
        raise ValueError("Asset selection must be a one-dimensional array.")

    if bits.size != covariance.shape[0]:
        raise ValueError(
            "Asset selection length does not match covariance dimensions."
        )

    if not np.isin(bits, [0.0, 1.0]).all():
        raise ValueError("Asset selection must contain only binary values.")

    selected_count = int(bits.sum())

    if selected_count == 0:
        raise ValueError(
            "Equal-weight variance is undefined for an empty portfolio."
        )

    return evaluate_covariance_objective(bits, covariance) / (
        selected_count ** 2
    )


# ============================================================
# Construct the original constrained quadratic program
# ============================================================

def build_quadratic_program():
    """
    Binary decision variable:
        x_i = 1 if asset i is selected, otherwise 0.

    Current objective:
        Minimize x.T @ covariance @ x.

    Constraints:
        MIN_ASSETS <= sum(x_i) <= MAX_ASSETS.

    Important:
        The current objective is not normalized by portfolio size.
        Equal-weight variance requires division by selected_count**2.
        That variable denominator requires a separate formulation.
    """

    returns = load_returns()
    assets = list(returns.columns)
    covariance = calculate_covariance(returns)

    asset_count = len(assets)

    # Scale the penalty relative to the covariance matrix and universe.
    largest_covariance = float(np.max(np.abs(covariance)))
    penalty = (
        PENALTY_MULTIPLIER
        * largest_covariance
        * asset_count
    )

    qp = QuadraticProgram("QuantumRisk_Variable_Size")

    for asset in assets:
        qp.binary_var(name=asset)

    # x.T @ covariance @ x.
    # For binary variables, x_i**2 = x_i, so diagonal terms
    # are represented as linear coefficients.
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

    qp.linear_constraint(
        linear={asset: 1.0 for asset in assets},
        sense=">=",
        rhs=MIN_ASSETS,
        name="minimum_assets",
    )

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
    """Convert the constrained quadratic program into a QUBO."""

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
    print(f"Auxiliary variables: {qubo.get_num_vars() - len(assets)}")
    print(f"Penalty coefficient: {penalty:.8g}")

    print("\nObjective: x.T @ covariance @ x")
    print("Portfolio-size constraints are encoded as QUBO penalties.")
    print(
        "WARNING: This is not normalized equal-weight variance "
        "for variable portfolio sizes."
    )

    return qp, qubo, assets, covariance


# ============================================================
# Lightweight validation
# ============================================================

def validate_qubo():
    """
    Validate the model structure, objective evaluation, and random
    portfolio-size feasibility. This is not a global-optimum proof.
    """

    qp, qubo, assets, covariance = build_qubo()

    rng = np.random.default_rng(RANDOM_SEED)
    asset_count = len(assets)

    feasible_count = 0
    infeasible_count = 0
    sampled_objectives = []
    sampled_equal_weight_variances = []

    print("\nRunning lightweight QUBO validation...")
    print(f"Random assignments to test: {RANDOM_ASSIGNMENTS}")

    for _ in range(RANDOM_ASSIGNMENTS):
        bits = rng.integers(0, 2, size=asset_count)
        selected_count = int(bits.sum())

        if MIN_ASSETS <= selected_count <= MAX_ASSETS:
            feasible_count += 1

            objective = evaluate_covariance_objective(
                bits, covariance
            )
            equal_weight_variance = evaluate_equal_weight_variance(
                bits, covariance
            )

            if not np.isfinite(objective):
                raise ValueError(
                    "A sampled feasible assignment has a non-finite "
                    "objective value."
                )

            if not np.isfinite(equal_weight_variance):
                raise ValueError(
                    "A sampled feasible portfolio has a non-finite "
                    "equal-weight variance."
                )

            # Verify the relationship between the two quantities.
            expected_variance = objective / (selected_count ** 2)

            if not np.isclose(
                equal_weight_variance,
                expected_variance,
                rtol=1e-10,
                atol=1e-14,
            ):
                raise AssertionError(
                    "Equal-weight variance calculation failed."
                )

            sampled_objectives.append(objective)
            sampled_equal_weight_variances.append(
                equal_weight_variance
            )
        else:
            infeasible_count += 1

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
            "Lowest sampled covariance objective: "
            f"{min(sampled_objectives):.12g}"
        )
        print(
            "Highest sampled covariance objective: "
            f"{max(sampled_objectives):.12g}"
        )
        print(
            "Lowest sampled equal-weight variance: "
            f"{min(sampled_equal_weight_variances):.12g}"
        )
        print(
            "Highest sampled equal-weight variance: "
            f"{max(sampled_equal_weight_variances):.12g}"
        )

    print("\nModel checks passed:")
    print("- Asset-selection variables are binary.")
    print("- Minimum and maximum portfolio-size constraints exist.")
    print("- Converted QUBO coefficients are finite.")
    print("- Random feasible portfolios satisfy the size limits.")
    print("- Equal-weight variance uses the selected portfolio size.")

    print("\nLimitations:")
    print("- Random sampling is not an optimality certificate.")
    print("- Penalty strength has not been proven sufficient.")
    print("- The QUBO objective is still unnormalized.")
    print(
        "- Equal-weight variance is evaluated for validation only; "
        "it is not yet the optimized QUBO objective."
    )

    print("\nLightweight validation completed successfully.")

    return qp, qubo, assets, covariance


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    validate_qubo()
