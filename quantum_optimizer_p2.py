"""
QuantumRisk optimizer v2: dimensionless cost scaling.

Main model improvement:
- Portfolio variances are around 1e-5. Using them directly in exp(-i * gamma * cost)
  makes the cost-layer phases tiny for gamma values near [-pi, pi].
- This version centers and scales costs before the QAOA phase and optimizer objective.
- Reported portfolio variances remain in their original units.
- The exact classical benchmark remains exhaustive enumeration.

Run from the repository root:
    python src/quantum_optimizer.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ----------------------------- Configuration -------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RETURNS_FILE = PROJECT_ROOT / "data" / "daily_returns.csv"
RESULTS_FILE = PROJECT_ROOT / "data" / "quantum_portfolio_results.csv"

PORTFOLIO_SIZE = 4
DEPTH_SCHEDULE = (2,)  # Experiment: deeper circuit than the p=1 baseline.
OPTIMIZER_RESTARTS = 5
MAX_ITERATIONS_PER_RESTART = 60
RANDOM_SEED = 2026
OPTIMIZER_METHOD = "COBYLA"

_XY_GATE_INDEX_CACHE: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}


# ------------------------------- Data ---------------------------------------

def load_returns(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Returns file not found: {path}\n"
            "Expected data/daily_returns.csv in the project."
        )
    frame = pd.read_csv(path, index_col=0)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if frame.empty or frame.shape[1] < PORTFOLIO_SIZE:
        raise ValueError("Insufficient valid data to construct the requested portfolio.")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Returns data contains non-finite values.")
    return [str(c) for c in frame.columns], values


def portfolio_variance(indices: tuple[int, ...], covariance: np.ndarray) -> float:
    k = len(indices)
    sub_cov = covariance[np.ix_(indices, indices)]
    return float(sub_cov.sum() / (k * k))


def portfolio_mask(indices: tuple[int, ...]) -> int:
    mask = 0
    for i in indices:
        mask |= 1 << i
    return mask


def scale_costs(costs: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Center and scale costs so QAOA phases are not negligibly small.
    Subtracting the minimum adds only a global phase; division by standard
    deviation gives the phase Hamiltonian a useful dimensionless scale.
    """
    scale = float(np.std(costs))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.ptp(costs))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (costs - float(np.min(costs))) / scale, scale


# --------------------------- XY mixer simulation ----------------------------

def precompute_xy_gate_indices(asset_count: int):
    state_size = 1 << asset_count
    dtype = np.uint32 if state_size <= np.iinfo(np.uint32).max else np.uint64
    indices = np.arange(state_size, dtype=dtype)
    gate_pairs = []
    for qa in range(asset_count):
        qb = (qa + 1) % asset_count
        ma, mb = 1 << qa, 1 << qb
        i01 = indices[((indices & ma) == 0) & ((indices & mb) != 0)]
        i10 = i01 ^ ma ^ mb
        gate_pairs.append((i01, i10))
    return gate_pairs


def get_xy_gate_indices(asset_count: int):
    if asset_count not in _XY_GATE_INDEX_CACHE:
        _XY_GATE_INDEX_CACHE[asset_count] = precompute_xy_gate_indices(asset_count)
    return _XY_GATE_INDEX_CACHE[asset_count]


def apply_xy_gate(state, beta: float, i01: np.ndarray, i10: np.ndarray) -> None:
    a01 = state[i01].copy()
    a10 = state[i10].copy()
    c, s = np.cos(beta), -1j * np.sin(beta)
    state[i01] = c * a01 + s * a10
    state[i10] = c * a10 + s * a01


def simulate_qaoa(
    parameters: np.ndarray,
    asset_count: int,
    feasible_indices: np.ndarray,
    scaled_costs: np.ndarray,
    depth: int,
) -> np.ndarray:
    state = np.zeros(1 << asset_count, dtype=np.complex128)
    state[feasible_indices] = 1 / np.sqrt(len(feasible_indices))
    gammas, betas = parameters[:depth], parameters[depth:]
    gate_pairs = get_xy_gate_indices(asset_count)

    for layer in range(depth):
        state[feasible_indices] *= np.exp(-1j * gammas[layer] * scaled_costs)
        for i01, i10 in gate_pairs:
            apply_xy_gate(state, betas[layer], i01, i10)

    probabilities = np.abs(state[feasible_indices]) ** 2
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Invalid probability distribution from simulation.")
    return probabilities / total


# ------------------------------- Optimizer ----------------------------------

def optimize_depth(
    depth: int,
    feasible_indices: np.ndarray,
    scaled_costs: np.ndarray,
    asset_count: int,
    rng: np.random.Generator,
):
    best_x = None
    best_fun = float("inf")
    nfev_total = 0
    records = []

    for restart in range(OPTIMIZER_RESTARTS):
        x0 = rng.uniform(-np.pi, np.pi, size=2 * depth)

        def objective(x):
            probabilities = simulate_qaoa(
                x, asset_count, feasible_indices, scaled_costs, depth
            )
            return float(np.dot(probabilities, scaled_costs))

        result = minimize(
            objective,
            x0,
            method=OPTIMIZER_METHOD,
            options={"maxiter": MAX_ITERATIONS_PER_RESTART, "rhobeg": 0.5, "tol": 1e-3},
        )
        nfev_total += int(getattr(result, "nfev", 0))
        records.append(
            {
                "restart": restart + 1,
                "objective_scaled": float(result.fun),
                "nfev": int(getattr(result, "nfev", 0)),
                "success": bool(result.success),
                "message": str(result.message),
            }
        )
        print(
            f"  start {restart + 1}/{OPTIMIZER_RESTARTS}: "
            f"scaled_objective={result.fun:.7f}, "
            f"nfev={result.nfev}, success={result.success}"
        )
        if np.isfinite(result.fun) and result.fun < best_fun:
            best_fun = float(result.fun)
            best_x = np.asarray(result.x, dtype=float).copy()

    if best_x is None:
        raise RuntimeError("All optimizer restarts failed to produce a valid result.")
    return best_x, best_fun, nfev_total, records


# --------------------------------- Main -------------------------------------

def main() -> None:
    started = time.perf_counter()
    print("QuantumRisk optimizer — normalized cost model")
    print("-" * 55)

    names, returns = load_returns(RETURNS_FILE)
    covariance = np.cov(returns, rowvar=False, ddof=1)
    covariance = 0.5 * (covariance + covariance.T)
    n, k = len(names), PORTFOLIO_SIZE

    portfolios = list(combinations(range(n), k))
    costs = np.asarray([portfolio_variance(p, covariance) for p in portfolios])
    scaled_costs, cost_scale = scale_costs(costs)
    feasible_indices = np.asarray([portfolio_mask(p) for p in portfolios], dtype=np.int64)

    exact_pos = int(np.argmin(costs))
    exact_portfolio = portfolios[exact_pos]
    exact_variance = float(costs[exact_pos])

    print(f"Assets: {n} | observations: {len(returns)} | k: {k}")
    print(f"Depth schedule: {DEPTH_SCHEDULE} | objective: scaled expectation")
    print(f"Restarts per depth: {OPTIMIZER_RESTARTS} | max evaluations/restart: {MAX_ITERATIONS_PER_RESTART}")
    print(f"Feasible portfolios: {len(portfolios):,}")
    print(f"Cost scaling: (variance - minimum) / std; std={cost_scale:.6e}")
    print(
        f"Exact classical optimum: {[names[i] for i in exact_portfolio]}; "
        f"variance={exact_variance:.11g}"
    )

    rng = np.random.default_rng(RANDOM_SEED)
    all_rows = []
    nfev_total = 0
    opt_started = time.perf_counter()

    for depth in DEPTH_SCHEDULE:
        print(f"\nOptimizing depth p={depth}")
        parameters, scaled_objective, nfev, records = optimize_depth(
            depth, feasible_indices, scaled_costs, n, rng
        )
        nfev_total += nfev
        probabilities = simulate_qaoa(
            parameters, n, feasible_indices, scaled_costs, depth
        )
        order = np.argsort(probabilities)[::-1]
        modal_pos = int(order[0])
        modal_portfolio = portfolios[modal_pos]
        modal_variance = float(costs[modal_pos])
        expected_variance = float(np.dot(probabilities, costs))
        gap_pct = (modal_variance / exact_variance - 1) * 100 if exact_variance else np.nan
        top20 = order[:min(20, len(order))]
        best_top20_pos = int(top20[np.argmin(costs[top20])])

        print("\nFINAL RESULTS")
        print(f"Modal portfolio: {[names[i] for i in modal_portfolio]}")
        print(f"Modal probability: {probabilities[modal_pos]:.8f}")
        print(f"Modal daily variance: {modal_variance:.11g}")
        print(f"Modal probability rank: 1/{len(portfolios):,}; variance gap: {gap_pct:.2f}%")
        print(f"Probability of exact optimum: {probabilities[exact_pos]:.8f}")
        print(f"Expected variance under distribution: {expected_variance:.11g}")
        print(
            "Best variance among top-20 probable states: "
            f"{[names[i] for i in portfolios[best_top20_pos]]} "
            f"(rank {int(np.where(order == best_top20_pos)[0][0]) + 1})"
        )
        print(f"Best scaled objective: {scaled_objective:.7f}")
        for rec in records:
            print(f"  restart detail {rec['restart']}: {rec['message']}")

        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        for pos, portfolio in enumerate(portfolios):
            all_rows.append(
                {
                    "depth": depth,
                    "portfolio": "|".join(names[i] for i in portfolio),
                    "probability": float(probabilities[pos]),
                    "daily_variance": float(costs[pos]),
                    "rank_by_probability": int(ranks[pos]),
                    "is_exact_optimum": bool(pos == exact_pos),
                    "scaled_cost": float(scaled_costs[pos]),
                }
            )

    opt_seconds = time.perf_counter() - opt_started
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(RESULTS_FILE, index=False)
    print(f"\nEvaluations: {nfev_total}")
    print(f"Optimization time: {opt_seconds:.2f}s")
    print(f"Total runtime: {time.perf_counter() - started:.2f}s")
    print(f"Saved results to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
