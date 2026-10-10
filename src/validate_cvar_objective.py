"""
Independent validation of QuantumRisk's CVaR objective and cost scaling.

Run from the repository root:
    python src/validate_cvar_objective.py

This script does not run COBYLA and does not modify the optimizer or results.
It independently recomputes portfolio CVaR, checks portfolio/mask/cost ordering,
checks cost scaling, and validates the expected-cost identities used by QAOA.
"""

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "daily_returns.csv"
PORTFOLIO_SIZE = 4
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 20261010
ATOL = 1e-11
RTOL = 1e-8


def independent_load_returns(path):
    """Load returns independently using the project's date/index-first format."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path, index_col=0)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="any")

    if frame.empty or frame.shape[1] < PORTFOLIO_SIZE:
        raise ValueError("Dataset does not contain enough complete asset returns.")

    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Dataset contains non-finite values.")

    return [str(c) for c in frame.columns], values


def independent_var_cvar(losses, confidence):
    """Independently apply the optimizer's documented empirical convention."""
    var_loss = float(np.quantile(losses, confidence, method="linear"))
    tail = losses[losses >= var_loss]
    if tail.size == 0:
        raise ValueError("Empty CVaR tail.")
    return var_loss, float(tail.mean())


def independent_scale(costs):
    """Replicate documented scaling independently for cross-checking."""
    scale = float(np.std(costs))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.ptp(costs))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (costs - float(np.min(costs))) / scale, scale


def assert_close(name, actual, expected, atol=ATOL, rtol=RTOL):
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        max_error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        raise AssertionError(f"{name} FAILED; maximum absolute error={max_error:.3e}")
    print(f"PASS: {name}")


def main():
    # Import the project module only to compare its implementation with this
    # independent calculation. Importing does not run its main() optimizer.
    sys.path.insert(0, str(ROOT / "src"))
    import quantum_cvar_optimizer as optimizer

    print("=== QuantumRisk CVaR Objective Validation ===")
    print(f"Dataset: {DATA_FILE}")
    print("This validation does not run QAOA parameter optimization.")
    print()

    names, returns = independent_load_returns(DATA_FILE)
    n_assets = len(names)
    portfolios = list(combinations(range(n_assets), PORTFOLIO_SIZE))

    independent_vars = []
    independent_cvars = []
    independent_variances = []
    independent_masks = []

    for portfolio in portfolios:
        portfolio_returns = returns[:, portfolio].mean(axis=1)
        losses = -portfolio_returns
        var_loss, cvar_loss = independent_var_cvar(losses, CONFIDENCE_LEVEL)

        independent_vars.append(var_loss)
        independent_cvars.append(cvar_loss)
        independent_variances.append(float(np.var(portfolio_returns, ddof=1)))

        mask = 0
        for i in portfolio:
            mask |= 1 << i
        independent_masks.append(mask)

    independent_vars = np.asarray(independent_vars, dtype=float)
    independent_cvars = np.asarray(independent_cvars, dtype=float)
    independent_variances = np.asarray(independent_variances, dtype=float)
    independent_masks = np.asarray(independent_masks, dtype=np.int64)

    # Compare dataset interpretation and every per-portfolio objective value.
    project_names, project_returns = optimizer.load_returns(DATA_FILE)
    if names != project_names:
        raise AssertionError("Asset order differs between independent and project loaders.")
    assert_close("Loaded return matrix", returns, project_returns)

    project_portfolio_rows = [
        optimizer.portfolio_cvar(p, project_returns) for p in portfolios
    ]
    project_vars = np.asarray([r[0] for r in project_portfolio_rows])
    project_cvars = np.asarray([r[1] for r in project_portfolio_rows])
    project_variances = np.asarray([r[2] for r in project_portfolio_rows])

    assert_close("VaR for every portfolio", project_vars, independent_vars)
    assert_close("CVaR for every portfolio", project_cvars, independent_cvars)
    assert_close("Sample variance for every portfolio",
                 project_variances, independent_variances)

    project_masks = np.asarray(
        [optimizer.portfolio_mask(p) for p in portfolios], dtype=np.int64
    )
    assert_close("Portfolio-to-bitmask ordering", project_masks, independent_masks)

    # Compare the project's scaled objective costs against an independent scale.
    project_scaled, project_scale = optimizer.scale_costs(project_cvars)
    independent_scaled, independent_scale_value = independent_scale(independent_cvars)
    assert_close("Scaled CVaR cost vector", project_scaled, independent_scaled)
    assert_close("Cost scaling factor", project_scale, independent_scale_value)

    if not np.isclose(float(np.min(project_scaled)), 0.0, atol=ATOL):
        raise AssertionError("Scaled objective minimum is not zero.")
    if not np.isfinite(project_scaled).all():
        raise AssertionError("Scaled objective contains non-finite values.")
    print("PASS: Scaled costs are finite and minimum cost is zero")

    # Confirm the exact optimum and its corresponding feasible-state index.
    exact_idx = int(np.argmin(independent_cvars))
    exact_portfolio = tuple(names[i] for i in portfolios[exact_idx])
    print()
    print(f"Assets: {n_assets}; observations: {len(returns)}")
    print(f"Feasible portfolios: {len(portfolios):,}")
    print(f"Minimum-CVaR portfolio: {', '.join(exact_portfolio)}")
    print(f"Minimum CVaR loss: {independent_cvars[exact_idx]:.12f}")
    print(f"Minimum VaR loss: {independent_vars[exact_idx]:.12f}")
    print(f"Scaled cost at optimum: {project_scaled[exact_idx]:.12g}")

    if not np.isclose(project_scaled[exact_idx], 0.0, atol=ATOL):
        raise AssertionError("Minimum-CVaR portfolio does not map to zero scaled cost.")
    print("PASS: Minimum-CVaR portfolio maps to minimum scaled cost")

    # Test objective identities using several probability distributions.
    # This validates the mathematics independently of the circuit sampler.
    rng = np.random.default_rng(RANDOM_SEED)
    distributions = {
        "uniform": np.full(len(portfolios), 1.0 / len(portfolios)),
        "one-hot optimum": np.eye(1, len(portfolios), exact_idx, dtype=float).ravel(),
        "random normalized": rng.random(len(portfolios)),
    }
    distributions["random normalized"] /= distributions["random normalized"].sum()

    min_cvar = float(np.min(independent_cvars))
    for label, probabilities in distributions.items():
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
            raise AssertionError(f"{label}: invalid probability distribution.")

        # Independently compute expectation in raw and scaled units.
        expected_raw_cvar = float(np.sum(probabilities * independent_cvars))
        expected_scaled_cost = float(np.sum(probabilities * independent_scaled))

        # Since scaled = (CVaR - min_CVaR) / scale, this identity must hold.
        reconstructed_scaled = (
            expected_raw_cvar - min_cvar
        ) / independent_scale_value

        assert_close(
            f"{label}: expected scaled cost identity",
            expected_scaled_cost,
            reconstructed_scaled,
        )

        reconstructed_raw = (
            expected_scaled_cost * independent_scale_value + min_cvar
        )
        assert_close(
            f"{label}: reconstruct expected raw CVaR",
            reconstructed_raw,
            expected_raw_cvar,
        )

        print(
            f"  {label}: E[CVaR]={expected_raw_cvar:.12f}; "
            f"E[scaled cost]={expected_scaled_cost:.12f}"
        )

    print()
    print("PASS: CVaR objective, cost scaling, feasible-state ordering,")
    print("      and expected-cost identities all agree.")
    print("NOTE: This validates the objective math, not optimizer convergence")
    print("      or quantum advantage. The circuit tests address circuit behavior.")


if __name__ == "__main__":
    main()
