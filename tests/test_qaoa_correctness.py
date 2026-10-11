
from pathlib import Path
import importlib.util
import sys

import numpy as np


# ============================================================
# LOAD OPTIMIZER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_FILE = PROJECT_ROOT / "src" / "quantum_optimizer.py"

passed = 0
failed = 0


def load_optimizer_module():
    print(f"Project root:   {PROJECT_ROOT}")
    print(f"Optimizer file: {SOURCE_FILE}")

    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(f"Optimizer not found: {SOURCE_FILE}")

    spec = importlib.util.spec_from_file_location(
        "quantum_optimizer", SOURCE_FILE
    )

    if spec is None or spec.loader is None:
        raise ImportError("Could not load quantum_optimizer.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check(condition, description):
    global passed, failed

    if condition:
        passed += 1
        print(f"PASS: {description}")
    else:
        failed += 1
        print(f"FAIL: {description}")


# ============================================================
# TESTS
# ============================================================

def main():
    optimizer = load_optimizer_module()

    print("=" * 72)
    print("QUANTUMRISK: XY-MIXER CORRECTNESS TESTS")
    print("=" * 72)

    rng = np.random.default_rng(42)

    n = 6
    k = 2
    reps = 1

    # make_feasible_basis returns (indices, bitstrings).
    feasible_indices, bitstrings = optimizer.make_feasible_basis(n, k)

    feasible_indices = np.asarray(feasible_indices, dtype=int)
    bitstrings = np.asarray(bitstrings)

    expected_count = 15  # C(6, 2) = 15

    check(
        len(feasible_indices) == expected_count,
        "Correct number of feasible states",
    )

    check(
        bitstrings.shape == (expected_count, n),
        "Feasible bitstrings have the correct shape",
    )

    check(
        np.all(np.sum(bitstrings, axis=1) == k),
        "Every feasible bitstring selects exactly k assets",
    )

    check(
        len(np.unique(feasible_indices)) == len(feasible_indices),
        "Feasible state indices are unique",
    )

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    initial_state = np.zeros(2**n, dtype=np.complex128)
    initial_state[feasible_indices] = 1 / np.sqrt(len(feasible_indices))

    check(
        np.isclose(np.linalg.norm(initial_state), 1.0, atol=1e-10),
        "Initial state is normalized",
    )

    initial_probabilities = np.abs(initial_state) ** 2
    infeasible_indices = np.array(
        [
            index
            for index in range(2**n)
            if index.bit_count() != k
        ],
        dtype=int,
    )

    check(
        np.isclose(
            np.sum(initial_probabilities[infeasible_indices]),
            0.0,
            atol=1e-10,
        ),
        "Initial state contains only feasible portfolios",
    )

    # --------------------------------------------------------
    # XY mixer: use the actual function signature
    # apply_xy_gate(state, beta, q1, q2, asset_count)
    # --------------------------------------------------------

    random_state = (
        rng.normal(size=2**n)
        + 1j * rng.normal(size=2**n)
    ).astype(np.complex128)

    random_state /= np.linalg.norm(random_state)

    mixed_random = optimizer.apply_xy_gate(
        random_state.copy(), 0.37, 0, 1, n
    )

    check(
        np.isclose(
            np.linalg.norm(mixed_random),
            np.linalg.norm(random_state),
            atol=1e-10,
        ),
        "XY mixer preserves statevector norm",
    )

    mixed_feasible = optimizer.apply_xy_gate(
        initial_state.copy(), 0.37, 0, 1, n
    )

    mixed_probabilities = np.abs(mixed_feasible) ** 2

    check(
        np.isclose(
            np.sum(mixed_probabilities[infeasible_indices]),
            0.0,
            atol=1e-10,
        ),
        "XY mixer preserves fixed cardinality",
    )

    # --------------------------------------------------------
    # Full QAOA simulation
    # --------------------------------------------------------

    print("\nTesting full QAOA simulation...")

    costs = rng.random(len(feasible_indices))
    gammas = [0.23]
    betas = [0.41]
    parameters = np.array(gammas + betas, dtype=float)

    final_state = optimizer.simulate_feasible_qaoa(
        parameters,
        costs,
        feasible_indices,
        n,
        reps,
    )

    final_state = np.asarray(final_state)
    check(
        final_state.shape == (2**n,),
        "Full QAOA returns a full statevector",
    )

    check(
        np.isclose(np.linalg.norm(final_state), 1.0, atol=1e-9),
        "Full QAOA statevector is normalized",
    )

    final_probabilities = np.abs(final_state) ** 2

    check(
        np.isclose(
            np.sum(final_probabilities[infeasible_indices]),
            0.0,
            atol=1e-9,
        ),
        "Full QAOA preserves fixed portfolio size",
    )

    check(
        np.isclose(np.sum(final_probabilities), 1.0, atol=1e-9),
        "Full QAOA probabilities sum to one",
    )

    # --------------------------------------------------------
    # Portfolio variance formula
    # --------------------------------------------------------

    returns = rng.normal(size=(100, n)) * 0.01
    covariance = np.cov(returns, rowvar=False)

    selected = [0, 2]
    weights = np.zeros(n)
    weights[selected] = 1.0 / k

    variance_full = weights @ covariance @ weights

    selected_covariance = covariance[np.ix_(selected, selected)]
    variance_selected = np.sum(selected_covariance) / (k**2)

    check(
        np.isclose(variance_full, variance_selected, atol=1e-12),
        "Portfolio variance formulas agree",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("TEST SUMMARY")
    print("=" * 72)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 72)

    if failed:
        raise SystemExit(
            "Some checks failed. Investigate them before trusting results."
        )

    print("All correctness checks passed.")


if __name__ == "__main__":
    main()
