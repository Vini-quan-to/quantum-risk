from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from qubo_model import (
    MIN_ASSETS,
    MAX_ASSETS,
    build_qubo,
)


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "qubo_validation.csv"

RANDOM_ASSIGNMENTS = 1000
RANDOM_SEED = 2026
MAX_SLACK_BITS_FOR_EXHAUSTIVE_SEARCH = 12
TOLERANCE = 1e-9


# ============================================================
# Helpers
# ============================================================

def original_objective(bits, covariance):
    """Evaluate x.T @ covariance @ x for binary asset variables."""

    bits = np.asarray(bits, dtype=float)
    return float(bits @ covariance @ bits)


def get_variable_names(qubo):
    """Return converted QUBO variable names in their actual order."""

    return [variable.name for variable in qubo.variables]


def find_best_slack_assignment(qubo, asset_bits, asset_count):
    """
    Find the lowest QUBO energy over all auxiliary-bit assignments.

    This is practical only when the number of slack variables is small.
    The function does not allocate a dense matrix.
    """

    variable_names = get_variable_names(qubo)

    asset_positions = {
        name: index
        for index, name in enumerate(variable_names)
        if name in asset_bits
    }

    if len(asset_positions) != asset_count:
        missing = sorted(set(asset_bits) - set(asset_positions))
        raise ValueError(
            "Could not map every asset variable into the converted QUBO. "
            f"Missing: {missing}"
        )

    slack_positions = [
        index
        for index, name in enumerate(variable_names)
        if name not in asset_bits
    ]

    if len(slack_positions) > MAX_SLACK_BITS_FOR_EXHAUSTIVE_SEARCH:
        raise ValueError(
            f"Found {len(slack_positions)} slack variables. "
            "Exhaustive slack search is limited to "
            f"{MAX_SLACK_BITS_FOR_EXHAUSTIVE_SEARCH} bits."
        )

    full_bits = np.zeros(len(variable_names), dtype=float)

    for asset, bit in asset_bits.items():
        full_bits[variable_names.index(asset)] = bit

    best_energy = float("inf")
    best_full_bits = None

    # Search all auxiliary-bit combinations, not all asset portfolios.
    for slack_values in product(
        (0.0, 1.0),
        repeat=len(slack_positions),
    ):
        candidate = full_bits.copy()

        for position, bit in zip(slack_positions, slack_values):
            candidate[position] = bit

        energy = float(qubo.objective.evaluate(candidate))

        if energy < best_energy:
            best_energy = energy
            best_full_bits = candidate.copy()

    return best_energy, best_full_bits, len(slack_positions)


# ============================================================
# Main validation
# ============================================================

def main():
    print("=" * 68)
    print("QUANTUMRISK: VARIABLE-SIZE QUBO VALIDATION")
    print("=" * 68)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    original_problem, qubo, assets, covariance = build_qubo()

    asset_count = len(assets)
    variable_names = get_variable_names(qubo)

    if covariance.shape != (asset_count, asset_count):
        raise ValueError("Covariance matrix dimensions do not match assets.")

    if not np.isfinite(covariance).all():
        raise ValueError("Covariance matrix contains non-finite values.")

    if not np.allclose(covariance, covariance.T, atol=1e-10):
        raise ValueError("Covariance matrix is not symmetric.")

    asset_set = set(assets)

    if not asset_set.issubset(set(variable_names)):
        raise ValueError(
            "The converted QUBO does not contain every asset variable."
        )

    slack_count = len(variable_names) - asset_count

    print(f"\nAvailable assets: {asset_count}")
    print(f"Allowed portfolio sizes: {MIN_ASSETS}–{MAX_ASSETS}")
    print(f"Original asset variables: {asset_count}")
    print(f"Converted QUBO variables: {len(variable_names)}")
    print(f"Auxiliary slack variables: {slack_count}")
    print(f"Random asset selections: {RANDOM_ASSIGNMENTS}")

    if slack_count > MAX_SLACK_BITS_FOR_EXHAUSTIVE_SEARCH:
        raise ValueError(
            f"Cannot exhaustively test {slack_count} slack bits with "
            f"the configured limit of "
            f"{MAX_SLACK_BITS_FOR_EXHAUSTIVE_SEARCH}."
        )

    rng = np.random.default_rng(RANDOM_SEED)

    records = []
    seen = set()

    for trial in range(RANDOM_ASSIGNMENTS):
        # Draw a random portfolio size within the permitted range.
        size = int(rng.integers(MIN_ASSETS, MAX_ASSETS + 1))

        selected_indices = rng.choice(
            asset_count,
            size=size,
            replace=False,
        )

        selected_indices = sorted(selected_indices.tolist())
        selected = [assets[index] for index in selected_indices]

        # Avoid duplicate portfolios where practical.
        key = tuple(selected)

        if key in seen:
            continue

        seen.add(key)

        asset_bits = {
            asset: int(asset in selected)
            for asset in assets
        }

        bits = np.array(
            [asset_bits[asset] for asset in assets],
            dtype=float,
        )

        direct_objective = original_objective(bits, covariance)

        # The original QP objective is x.T Sigma x.
        # This differs from equal-weight variance when portfolio size
        # varies, because equal-weight variance divides by size**2.
        equal_weight_variance = (
            direct_objective / (size ** 2)
        )

        best_energy, best_full_bits, _ = find_best_slack_assignment(
            qubo=qubo,
            asset_bits=asset_bits,
            asset_count=asset_count,
        )

        record = {
            "trial": trial + 1,
            "assets": ", ".join(selected),
            "number_of_assets": size,
            "feasible_by_size": MIN_ASSETS <= size <= MAX_ASSETS,
            "direct_quadratic_objective": direct_objective,
            "equal_weight_daily_variance": equal_weight_variance,
            "minimum_energy_over_slack": best_energy,
            "energy_minus_direct_objective": (
                best_energy - direct_objective
            ),
            "slack_assignment": ", ".join(
                f"{variable_names[index]}={int(best_full_bits[index])}"
                for index in range(len(variable_names))
                if variable_names[index] not in asset_set
            ),
        }

        records.append(record)

        if (trial + 1) % 100 == 0:
            print(f"Processed {trial + 1}/{RANDOM_ASSIGNMENTS} trials...")

    results = pd.DataFrame(records)

    if results.empty:
        raise RuntimeError("No validation records were generated.")

    # For feasible assignments, the constraint penalties should be
    # satisfiable by the auxiliary bits. A constant energy offset is
    # acceptable because it does not change objective rankings.
    feasible_results = results[
        results["feasible_by_size"]
    ].copy()

    if feasible_results.empty:
        raise RuntimeError("No feasible selections were sampled.")

    offsets = feasible_results["energy_minus_direct_objective"]

    offset_spread = float(offsets.max() - offsets.min())
    maximum_absolute_offset = float(offsets.abs().max())

    results.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 68)
    print("VALIDATION SUMMARY")
    print("=" * 68)

    print(f"Unique portfolios tested: {len(results)}")
    print(f"Feasible portfolios tested: {len(feasible_results)}")
    print(f"Slack variables: {slack_count}")

    print(
        "Maximum absolute energy/objective difference: "
        f"{maximum_absolute_offset:.12g}"
    )
    print(
        "Spread of energy-minus-objective across feasible samples: "
        f"{offset_spread:.12g}"
    )

    if offset_spread <= TOLERANCE:
        print(
            "\nPASS: Feasible sampled portfolios have a constant "
            "QUBO energy offset relative to the original objective."
        )
    else:
        print(
            "\nWARNING: The sampled feasible portfolios do not show "
            "a constant energy offset. Inspect the penalty encoding."
        )

    print("\nInterpretation:")
    print("- This is a sampled validation, not a global-optimum proof.")
    print("- It does not prove that the penalty is sufficiently strong.")
    print("- It does not establish equal-weight variance optimality.")
    print(f"\nDetailed results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
