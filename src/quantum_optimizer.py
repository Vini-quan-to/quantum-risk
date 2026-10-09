
from pathlib import Path
import os

import numpy as np
import pandas as pd

from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo


# ============================================================
# QuantumRisk: Fixed-Cardinality QAOA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RETURNS_FILE = DATA_DIR / "daily_returns.csv"

RESULTS_FILE = DATA_DIR / "quantum_portfolio_results.csv"
COMPARISON_FILE = DATA_DIR / "portfolio_size_comparison.csv"

MIN_ASSETS = 4
MAX_ASSETS = 10

# Smoke-test defaults: start with one portfolio size.
# Set RUN_ALL_SIZES=1 in PowerShell only after the smoke test passes.
RUN_ALL_SIZES = os.getenv("RUN_ALL_SIZES", "0") == "1"
TEST_PORTFOLIO_SIZE = 4

SHOTS = 128
QAOA_REPS = 1
COBYLA_MAXITER = 5
PENALTY_MULTIPLIER = 10.0
RANDOM_SEED = 2026


# ============================================================
# Load returns
# ============================================================

def load_returns():
    """Load and validate daily asset returns."""

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
        raise ValueError("No valid observations in the returns file.")

    if returns.columns.duplicated().any():
        raise ValueError("Duplicate asset names found.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns contain non-finite values.")

    if not (
        1 <= MIN_ASSETS <= MAX_ASSETS <= returns.shape[1]
    ):
        raise ValueError(
            f"Invalid portfolio size range {MIN_ASSETS}–{MAX_ASSETS} "
            f"for {returns.shape[1]} assets."
        )

    if len(returns) < 2:
        raise ValueError("At least two observations are required.")

    return returns


# ============================================================
# Build a fixed-cardinality QUBO
# ============================================================

def build_fixed_cardinality_qubo(returns, portfolio_size):
    """Build a quadratic objective with an exact-cardinality constraint."""

    assets = list(returns.columns)
    asset_count = len(assets)

    if not MIN_ASSETS <= portfolio_size <= MAX_ASSETS:
        raise ValueError("Requested portfolio size is out of range.")

    covariance = returns.cov().to_numpy(dtype=float)
    covariance = (covariance + covariance.T) / 2.0

    if not np.isfinite(covariance).all():
        raise ValueError("Covariance matrix contains non-finite values.")

    scale = float(np.max(np.abs(covariance)))

    if scale <= 0:
        raise ValueError("Covariance matrix has no positive scale.")

    penalty = PENALTY_MULTIPLIER * scale * asset_count

    qp = QuadraticProgram(f"QuantumRisk_k{portfolio_size}")

    for asset in assets:
        qp.binary_var(name=asset)

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

    qp.minimize(linear=linear, quadratic=quadratic)

    qp.linear_constraint(
        linear={asset: 1.0 for asset in assets},
        sense="==",
        rhs=portfolio_size,
        name="exact_portfolio_size",
    )

    qubo = QuadraticProgramToQubo(penalty=penalty).convert(qp)

    return qp, qubo, assets, covariance


# ============================================================
# Evaluate an equal-weight portfolio
# ============================================================

def evaluate_portfolio(returns, selected_assets):
    """Calculate equal-weight portfolio risk and return metrics."""

    if not selected_assets:
        raise ValueError("Cannot evaluate an empty portfolio.")

    portfolio_returns = returns[selected_assets].mean(axis=1)
    losses = -portfolio_returns.to_numpy()

    var_95 = float(np.quantile(losses, 0.95))
    tail_losses = losses[losses >= var_95]
    cvar_95 = float(tail_losses.mean())

    daily_mean = float(portfolio_returns.mean())
    daily_volatility = float(portfolio_returns.std(ddof=1))

    wealth = (1.0 + portfolio_returns).cumprod()
    drawdowns = 1.0 - wealth / wealth.cummax()

    k = len(selected_assets)
    covariance = returns[selected_assets].cov().to_numpy(dtype=float)
    equal_weight_variance = float(
        np.ones(k) @ covariance @ np.ones(k) / (k * k)
    )

    return {
        "portfolio_size": k,
        "equal_weight_variance": equal_weight_variance,
        "daily_mean_return": daily_mean,
        "annualized_mean_return": daily_mean * 252,
        "annualized_volatility": daily_volatility * np.sqrt(252),
        "historical_var_95": var_95,
        "historical_cvar_95": cvar_95,
        "maximum_drawdown": float(drawdowns.max()),
    }


# ============================================================
# Run QAOA for one size
# ============================================================

def solve_one_size(returns, portfolio_size):
    """Run a small QAOA smoke test for a fixed portfolio size."""

    qp, qubo, assets, covariance = build_fixed_cardinality_qubo(
        returns, portfolio_size
    )

    print(f"\nBuilding QAOA for exactly {portfolio_size} assets...", flush=True)
    print(f"Asset qubits: {len(assets)}", flush=True)
    print(f"QUBO variables: {qubo.get_num_vars()}", flush=True)
    print(f"Shots: {SHOTS}", flush=True)
    print(f"COBYLA max iterations: {COBYLA_MAXITER}", flush=True)

    sampler = StatevectorSampler(
        default_shots=SHOTS,
        seed=RANDOM_SEED,
    )

    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=COBYLA_MAXITER),
        reps=QAOA_REPS,
        initial_point=np.full(2 * QAOA_REPS, 0.5),
    )

    solver = MinimumEigenOptimizer(qaoa)

    print("Starting QAOA solve...", flush=True)
    result = solver.solve(qubo)

    # The first variables correspond to the original asset variables.
    asset_bits = np.rint(result.x[:len(assets)]).astype(int)

    selected_assets = [
        assets[i]
        for i, bit in enumerate(asset_bits)
        if bit == 1
    ]

    print("QAOA solve returned.", flush=True)
    print("Selected assets:", selected_assets, flush=True)
    print(f"Selected count: {len(selected_assets)}", flush=True)

    if len(selected_assets) != portfolio_size:
        raise RuntimeError(
            f"Expected {portfolio_size} selected assets, "
            f"but decoded {len(selected_assets)}. "
            "Do not treat this solution as feasible."
        )

    metrics = evaluate_portfolio(returns, selected_assets)

    return {
        "portfolio_size": portfolio_size,
        "selected_assets": selected_assets,
        "qubo_objective": float(result.fval),
        **metrics,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 68)
    print("QUANTUMRISK: QAOA SMOKE TEST", flush=True)
    print("=" * 68)

    returns = load_returns()

    print(f"Available assets: {returns.shape[1]}", flush=True)
    print(f"Historical observations: {len(returns)}", flush=True)

    if RUN_ALL_SIZES:
        portfolio_sizes = range(MIN_ASSETS, MAX_ASSETS + 1)
    else:
        portfolio_sizes = [TEST_PORTFOLIO_SIZE]

    results = []

    for k in portfolio_sizes:
        print(f"\n{'-' * 60}", flush=True)
        print(f"Portfolio size: {k}", flush=True)

        try:
            result = solve_one_size(returns, k)
            results.append(result)
        except Exception as exc:
            print(
                f"FAILED for k={k}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

    comparison = pd.DataFrame(results)
    comparison["selected_assets"] = comparison["selected_assets"].apply(
        lambda names: ", ".join(names)
    )

    comparison = comparison.sort_values(
        "equal_weight_variance"
    ).reset_index(drop=True)

    comparison.to_csv(COMPARISON_FILE, index=False)

    best = comparison.iloc[0]
    best_assets = [
        name.strip()
        for name in best["selected_assets"].split(",")
        if name.strip()
    ]

    pd.DataFrame({
        "asset": best_assets,
        "weight": [1.0 / len(best_assets)] * len(best_assets),
    }).to_csv(RESULTS_FILE, index=False)

    print("\n" + "=" * 68, flush=True)
    print("SMOKE TEST COMPLETED", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f"\nResults: {RESULTS_FILE}", flush=True)
    print(f"Comparison: {COMPARISON_FILE}", flush=True)
    print(
        "\nThis is a heuristic candidate, not a global-optimum proof.",
        flush=True,
    )


if __name__ == "__main__":
    main()
