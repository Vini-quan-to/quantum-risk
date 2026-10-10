"""
QuantumRisk — exhaustive vs simulated annealing vs greedy scaling benchmark.

Run from the repository root:
    python src/scaling_experiment.py

The current project has 20 real assets. For requested sizes greater than the
available real assets, this script creates clearly labelled synthetic assets
from resampled/correlated transformations of the observed returns. Therefore,
sizes > the real asset count are algorithmic stress tests, NOT real-market
evidence. Do not describe them otherwise in a report.

Methods:
- Exhaustive equal-weight k-asset search, only when combination count is under
  EXHAUSTIVE_MAX_COMBINATIONS.
- Simulated annealing over fixed-cardinality portfolios.
- Greedy forward selection.
All methods minimize sample daily variance. Results are not QAOA results.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RETURNS_FILE = DATA / "daily_returns.csv"
RESULTS_FILE = DATA / "scaling_classical_results.csv"

PORTFOLIO_SIZE = 4
SIZES_TO_TEST = (10, 15, 20, 30, 40, 50, 60)
EXHAUSTIVE_MAX_COMBINATIONS = 250_000
ANNEALING_RESTARTS = 8
ANNEALING_STEPS = 5_000
RANDOM_SEED = 2061


def load_returns() -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(RETURNS_FILE, index_col=0)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if frame.empty or frame.shape[1] < PORTFOLIO_SIZE:
        raise ValueError("daily_returns.csv needs at least four valid assets.")
    return [str(c) for c in frame.columns], frame.to_numpy(dtype=float)


def expand_synthetic_assets(names: list[str], returns: np.ndarray, target_n: int, seed: int):
    """Add synthetic correlated assets; retain real columns unchanged."""
    if target_n <= len(names):
        return names[:target_n], returns[:, :target_n], False
    rng = np.random.default_rng(seed)
    cols = [returns[:, i].copy() for i in range(returns.shape[1])]
    out_names = names.copy()
    base_std = np.std(returns, axis=0, ddof=1)
    base_std[base_std == 0] = 1e-8
    while len(cols) < target_n:
        parent = int(rng.integers(0, len(names)))
        # Small perturbation of an observed series preserves rough market-like
        # correlation while avoiding exact duplicate synthetic columns.
        noise = rng.normal(0.0, np.std(returns[:, parent], ddof=1) * 0.20, size=len(returns))
        scale = float(rng.uniform(0.85, 1.15))
        synthetic = returns[:, parent] * scale + noise
        cols.append(synthetic)
        out_names.append(f"SYN_{len(out_names) + 1:02d}_from_{names[parent]}")
    return out_names, np.column_stack(cols), True


def variance_for_indices(returns: np.ndarray, indices: tuple[int, ...] | list[int]) -> float:
    portfolio_returns = returns[:, list(indices)].mean(axis=1)
    return float(np.var(portfolio_returns, ddof=1))


def exhaustive(returns: np.ndarray, k: int):
    n = returns.shape[1]
    total = comb(n, k)
    if total > EXHAUSTIVE_MAX_COMBINATIONS:
        return None, None, total
    best_indices = None
    best_value = float("inf")
    start = time.perf_counter()
    # Evaluate in chunks of combinations to avoid retaining all portfolios.
    for combo in combinations(range(n), k):
        value = variance_for_indices(returns, combo)
        if value < best_value:
            best_value, best_indices = value, combo
    return best_indices, best_value, time.perf_counter() - start


def greedy_forward(returns: np.ndarray, k: int):
    start = time.perf_counter()
    n = returns.shape[1]
    selected: list[int] = []
    remaining = set(range(n))
    while len(selected) < k:
        best_idx = None
        best_value = float("inf")
        for idx in remaining:
            candidate = selected + [idx]
            value = variance_for_indices(returns, candidate)
            if value < best_value:
                best_idx, best_value = idx, value
        selected.append(best_idx)
        remaining.remove(best_idx)
    return tuple(sorted(selected)), variance_for_indices(returns, selected), time.perf_counter() - start


def simulated_annealing(returns: np.ndarray, k: int, seed: int):
    rng = np.random.default_rng(seed)
    n = returns.shape[1]
    best_global = None
    best_global_value = float("inf")
    start = time.perf_counter()

    # Cache objective values for visited portfolios.
    cache: dict[tuple[int, ...], float] = {}

    def objective(state):
        key = tuple(sorted(state))
        if key not in cache:
            cache[key] = variance_for_indices(returns, key)
        return cache[key]

    for restart in range(ANNEALING_RESTARTS):
        state = sorted(rng.choice(n, size=k, replace=False).tolist())
        current = objective(state)
        best_local, best_local_value = state.copy(), current
        t0 = max(current * 0.10, 1e-12)
        tf = max(current * 1e-5, 1e-14)

        for step in range(ANNEALING_STEPS):
            fraction = step / max(ANNEALING_STEPS - 1, 1)
            temperature = t0 * (tf / t0) ** fraction
            outgoing_pos = int(rng.integers(0, k))
            outgoing = state[outgoing_pos]
            incoming_candidates = list(set(range(n)) - set(state))
            incoming = int(rng.choice(incoming_candidates))
            candidate = state.copy()
            candidate[outgoing_pos] = incoming
            candidate.sort()
            candidate_value = objective(candidate)
            delta = candidate_value - current
            if delta <= 0 or rng.random() < np.exp(-delta / max(temperature, 1e-16)):
                state, current = candidate, candidate_value
            if current < best_local_value:
                best_local, best_local_value = state.copy(), current
            if current < best_global_value:
                best_global, best_global_value = state.copy(), current

    return tuple(best_global), best_global_value, time.perf_counter() - start, len(cache)


def main():
    names, real_returns = load_returns()
    rows = []
    print("QuantumRisk — classical scaling experiment")
    print("=" * 48)
    print(f"Real assets available: {len(names)}")
    print(f"Portfolio size k: {PORTFOLIO_SIZE}")
    print(f"Exhaustive search limit: {EXHAUSTIVE_MAX_COMBINATIONS:,} combinations")
    print("Synthetic expansions are stress tests, not real-market results.\n")

    for n in SIZES_TO_TEST:
        expanded_names, returns, synthetic = expand_synthetic_assets(
            names, real_returns, n, RANDOM_SEED + n
        )
        actual_n = len(expanded_names)
        total_combinations = comb(actual_n, PORTFOLIO_SIZE)
        print(f"--- N={actual_n}, combinations={total_combinations:,} ---")

        exact_indices, exact_value, exact_elapsed = exhaustive(returns, PORTFOLIO_SIZE)
        if exact_indices is not None:
            exact_assets = [expanded_names[i] for i in exact_indices]
            exact_status = "completed"
            print(f"Exhaustive: {exact_elapsed:.4f}s, variance={exact_value:.10g}")
        else:
            exact_assets = []
            exact_status = "skipped_combination_limit"
            exact_elapsed = np.nan
            exact_value = np.nan
            print("Exhaustive: skipped (over configured combination limit)")

        greedy_idx, greedy_value, greedy_time = greedy_forward(returns, PORTFOLIO_SIZE)
        sa_idx, sa_value, sa_time, sa_evals = simulated_annealing(
            returns, PORTFOLIO_SIZE, RANDOM_SEED + n
        )

        def gap(value):
            if exact_indices is None or not np.isfinite(exact_value) or exact_value == 0:
                return np.nan
            return 100.0 * (value / exact_value - 1.0)

        greedy_assets = [expanded_names[i] for i in greedy_idx]
        sa_assets = [expanded_names[i] for i in sa_idx]
        print(f"Greedy: {greedy_time:.4f}s, variance={greedy_value:.10g}, gap={gap(greedy_value):.2f}%")
        print(f"Simulated annealing: {sa_time:.4f}s, variance={sa_value:.10g}, gap={gap(sa_value):.2f}%, unique evaluations={sa_evals:,}\n")

        for method, indices, value, elapsed, evals in [
            ("exhaustive", exact_indices, exact_value, exact_elapsed, total_combinations if exact_indices is not None else np.nan),
            ("greedy_forward", greedy_idx, greedy_value, greedy_time, np.nan),
            ("simulated_annealing", sa_idx, sa_value, sa_time, sa_evals),
        ]:
            rows.append({
                "asset_count": actual_n,
                "portfolio_size": PORTFOLIO_SIZE,
                "combination_count": total_combinations,
                "synthetic_expansion_used": synthetic,
                "data_note": "synthetic stress test" if synthetic else "real dataset subset",
                "method": method,
                "status": exact_status if method == "exhaustive" else "completed",
                "portfolio_assets": ", ".join(expanded_names[i] for i in indices) if indices is not None else "",
                "daily_variance": value,
                "runtime_seconds": elapsed,
                "objective_evaluations": evals,
                "gap_vs_exact_percent": gap(value),
                "random_seed": RANDOM_SEED + n if method == "simulated_annealing" else np.nan,
            })

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_FILE, index=False)
    print(f"Saved results: {RESULTS_FILE}")
    print("Interpretation: runtime depends on implementation and hardware; this is not a quantum speedup test.")


if __name__ == "__main__":
    main()
