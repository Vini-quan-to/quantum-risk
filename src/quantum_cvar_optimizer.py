"""
QuantumRisk CVaR optimizer — QAOA-style classical statevector simulation.

Run from the repository root:
    python src/quantum_cvar_optimizer.py

Input:
    data/daily_returns.csv

Outputs:
    data/quantum_cvar_portfolio_results.csv
    data/quantum_cvar_optimizer_log.csv
    data/quantum_cvar_optimizer_parameters.csv

This is a classical simulation, not a quantum-hardware run.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ----------------------------- Configuration -------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETURNS_FILE = PROJECT_ROOT / "data" / "daily_returns.csv"
RESULTS_FILE = PROJECT_ROOT / "data" / "quantum_cvar_portfolio_results.csv"
RESTART_LOG_FILE = PROJECT_ROOT / "data" / "quantum_cvar_optimizer_log.csv"
PARAMETERS_FILE = PROJECT_ROOT / "data" / "quantum_cvar_optimizer_parameters.csv"

PORTFOLIO_SIZE = 4
CONFIDENCE_LEVEL = 0.95
DEPTH_SCHEDULE = (2,)
OPTIMIZER_RESTARTS = 5
MAX_ITERATIONS_PER_RESTART = 80
RANDOM_SEED = 2030
OPTIMIZER_METHOD = "COBYLA"
INITIAL_TRUST_REGION = 0.25
OPTIMIZER_TOLERANCE = 1e-2

_XY_GATE_INDEX_CACHE: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}


# ------------------------------- Data ---------------------------------------

def load_returns(path: Path) -> tuple[list[str], np.ndarray]:
    """Load daily asset returns from a CSV whose first column is the date/index."""
    if not path.exists():
        raise FileNotFoundError(
            f"Returns file not found: {path}\n"
            f"Expected file: {PROJECT_ROOT / 'data' / 'daily_returns.csv'}"
        )

    frame = pd.read_csv(path, index_col=0)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="any")

    if frame.empty or frame.shape[1] < PORTFOLIO_SIZE:
        raise ValueError(
            f"Need at least {PORTFOLIO_SIZE} assets and valid complete return rows."
        )

    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Returns data contains non-finite values.")

    return [str(c) for c in frame.columns], values


# ------------------------------- Risk metrics -------------------------------

def historical_var_cvar(
    losses: np.ndarray,
    confidence: float = CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """
    Return empirical VaR and CVaR in loss units.

    VaR is the linear empirical quantile. CVaR is the mean of all observations
    with losses >= VaR. With ties at the threshold, the tail can contain more
    than exactly (1-confidence) of the observations. This matches the
    convention used by src/cvar_baseline.py.
    """
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a non-empty one-dimensional array.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("CONFIDENCE_LEVEL must be between 0 and 1.")

    var_loss = float(np.quantile(losses, confidence, method="linear"))
    tail_losses = losses[losses >= var_loss]
    if tail_losses.size == 0:
        raise ValueError("Could not determine observations in the VaR tail.")
    return var_loss, float(tail_losses.mean())


def portfolio_cvar(
    indices: tuple[int, ...],
    returns: np.ndarray,
) -> tuple[float, float, float]:
    """Return (VaR loss, CVaR loss, sample variance) for an equal-weight portfolio."""
    k = len(indices)
    if k == 0:
        raise ValueError("Portfolio must contain at least one asset.")

    portfolio_returns = returns[:, indices].mean(axis=1)
    losses = -portfolio_returns
    var_loss, cvar_loss = historical_var_cvar(losses)
    variance = float(np.var(portfolio_returns, ddof=1))
    return var_loss, cvar_loss, variance


# ------------------------------- Cost scaling -------------------------------

def scale_costs(costs: np.ndarray) -> tuple[np.ndarray, float]:
    """Center and scale objective costs to a useful dimensionless range."""
    scale = float(np.std(costs))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.ptp(costs))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (costs - float(np.min(costs))) / scale, scale


def portfolio_mask(indices: tuple[int, ...]) -> int:
    mask = 0
    for i in indices:
        mask |= 1 << i
    return mask


# --------------------------- XY mixer simulation ----------------------------

def precompute_xy_gate_indices(asset_count: int):
    """Precompute basis-state pairs swapped by adjacent XY mixer gates."""
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


def apply_xy_gate(
    state: np.ndarray,
    beta: float,
    i01: np.ndarray,
    i10: np.ndarray,
) -> None:
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
    """Return probabilities over feasible fixed-cardinality basis states."""
    state = np.zeros(1 << asset_count, dtype=np.complex128)
    state[feasible_indices] = 1.0 / np.sqrt(len(feasible_indices))
    gammas = parameters[:depth]
    betas = parameters[depth:]
    gate_pairs = get_xy_gate_indices(asset_count)

    for layer in range(depth):
        state[feasible_indices] *= np.exp(-1j * gammas[layer] * scaled_costs)
        for i01, i10 in gate_pairs:
            apply_xy_gate(state, betas[layer], i01, i10)

    probabilities = np.abs(state[feasible_indices]) ** 2
    total = float(probabilities.sum())
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
        objective_history = []
        parameter_history = []

        def objective(x):
            probabilities = simulate_qaoa(
                x, asset_count, feasible_indices, scaled_costs, depth
            )
            value = float(np.dot(probabilities, scaled_costs))
            objective_history.append(value)
            parameter_history.append(np.asarray(x, dtype=float).copy())
            return value

        result = minimize(
            objective,
            x0,
            method=OPTIMIZER_METHOD,
            options={
                "maxiter": MAX_ITERATIONS_PER_RESTART,
                "rhobeg": INITIAL_TRUST_REGION,
                "tol": OPTIMIZER_TOLERANCE,
                "catol": 1e-6,
            },
        )

        initial_objective = (
            float(objective_history[0]) if objective_history else float("nan")
        )
        best_seen_objective = (
            float(min(objective_history)) if objective_history else float(result.fun)
        )
        final_window = objective_history[-min(10, len(objective_history)):]
        final_window_range = (
            float(max(final_window) - min(final_window))
            if final_window else float("nan")
        )
        hit_eval_cap = int(getattr(result, "nfev", 0)) >= MAX_ITERATIONS_PER_RESTART
        nfev = int(getattr(result, "nfev", 0))
        nfev_total += nfev

        records.append({
            "stage": "restart",
            "restart": restart + 1,
            "depth": depth,
            "initial_objective_scaled": initial_objective,
            "final_objective_scaled": float(result.fun),
            "best_seen_objective_scaled": best_seen_objective,
            "improvement": float(initial_objective - best_seen_objective),
            "last_10_eval_range": final_window_range,
            "nfev": nfev,
            "hit_eval_cap": bool(hit_eval_cap),
            "success": bool(result.success),
            "message": str(result.message),
        })

        print(
            f"  start {restart + 1}/{OPTIMIZER_RESTARTS}: "
            f"scaled_objective={float(result.fun):.7f}, nfev={nfev}, "
            f"success={bool(result.success)}, "
            f"improvement={initial_objective - best_seen_objective:.5g}, "
            f"last10_range={final_window_range:.3g}, hit_cap={hit_eval_cap}"
        )

        # Preserve the parameter vector corresponding to the best objective
        # actually evaluated during this restart.
        if objective_history:
            candidate_idx = int(np.argmin(objective_history))
            candidate_x = parameter_history[candidate_idx]
            candidate_fun = float(objective_history[candidate_idx])
        else:
            candidate_x = np.asarray(result.x, dtype=float).copy()
            candidate_fun = float(result.fun)

        if np.isfinite(candidate_fun) and candidate_fun < best_fun:
            best_fun = candidate_fun
            best_x = np.asarray(candidate_x, dtype=float).copy()

    if best_x is None:
        raise RuntimeError("All optimizer restarts failed to produce a valid result.")

    # Evaluate the returned parameters to ensure reported objective matches them.
    best_probabilities = simulate_qaoa(
        best_x, asset_count, feasible_indices, scaled_costs, depth
    )
    best_fun = float(np.dot(best_probabilities, scaled_costs))
    return best_x, best_fun, nfev_total, records


def refine_parameters(
    initial_parameters: np.ndarray,
    depth: int,
    feasible_indices: np.ndarray,
    scaled_costs: np.ndarray,
    asset_count: int,
):
    """Refine the best multistart parameters and keep the best point evaluated."""
    history: list[float] = []
    history_parameters: list[np.ndarray] = []

    def objective(x):
        probabilities = simulate_qaoa(
            x, asset_count, feasible_indices, scaled_costs, depth
        )
        value = float(np.dot(probabilities, scaled_costs))
        history.append(value)
        history_parameters.append(np.asarray(x, dtype=float).copy())
        return value

    initial_value = objective(initial_parameters)
    result = minimize(
        objective,
        np.asarray(initial_parameters, dtype=float),
        method="COBYLA",
        options={"maxiter": 250, "rhobeg": 0.05, "tol": 1e-3, "catol": 1e-7},
    )

    best_idx = int(np.argmin(history))
    refined_parameters = history_parameters[best_idx]
    refined_value = float(history[best_idx])
    final_window = history[-min(10, len(history)):]
    final_range = (
        float(max(final_window) - min(final_window))
        if final_window else float("nan")
    )

    print("\nLocal refinement")
    print(f"  initial objective: {initial_value:.7f}")
    print(f"  optimizer final:   {float(result.fun):.7f}")
    print(f"  best seen:         {refined_value:.7f}")
    print(f"  evaluations:       {int(getattr(result, 'nfev', 0))}")
    print(f"  last-10 range:     {final_range:.6g}")
    print(f"  success:           {bool(result.success)}")
    print(f"  message:           {result.message}")

    record = {
        "stage": "refinement",
        "restart": 0,
        "depth": depth,
        "initial_objective_scaled": initial_value,
        "final_objective_scaled": float(result.fun),
        "best_seen_objective_scaled": refined_value,
        "improvement": float(initial_value - refined_value),
        "last_10_eval_range": final_range,
        "nfev": int(getattr(result, "nfev", 0)),
        "hit_eval_cap": int(getattr(result, "nfev", 0)) >= 250,
        "success": bool(result.success),
        "message": str(result.message),
        "kept_refinement": refined_value < initial_value,
    }

    if refined_value >= initial_value:
        return np.asarray(initial_parameters, dtype=float), initial_value, record
    return refined_parameters, refined_value, record


# --------------------------------- Main -------------------------------------

def main() -> None:
    started = time.perf_counter()
    print("QuantumRisk optimizer — historical CVaR objective")
    print("-" * 55)

    names, returns = load_returns(RETURNS_FILE)
    n = len(names)

    portfolios = list(combinations(range(n), PORTFOLIO_SIZE))
    risk_rows = [portfolio_cvar(p, returns) for p in portfolios]
    var_losses = np.asarray([row[0] for row in risk_rows], dtype=float)
    cvar_losses = np.asarray([row[1] for row in risk_rows], dtype=float)
    variances = np.asarray([row[2] for row in risk_rows], dtype=float)

    scaled_costs, cost_scale = scale_costs(cvar_losses)
    feasible_indices = np.asarray(
        [portfolio_mask(p) for p in portfolios], dtype=np.int64
    )

    exact_pos = int(np.argmin(cvar_losses))
    exact_portfolio = portfolios[exact_pos]
    exact_cvar = float(cvar_losses[exact_pos])

    print(f"Assets: {n} | observations: {len(returns)} | k: {PORTFOLIO_SIZE}")
    print(f"Objective: historical CVaR loss at {CONFIDENCE_LEVEL:.0%} confidence")
    print(f"Depth schedule: {DEPTH_SCHEDULE} | objective: scaled expectation")
    print(
        f"Restarts per depth: {OPTIMIZER_RESTARTS} | "
        f"max evaluations/restart: {MAX_ITERATIONS_PER_RESTART}"
    )
    print(f"Feasible portfolios: {len(portfolios):,}")
    print(f"Cost scaling: (CVaR - minimum) / std; std={cost_scale:.6e}")
    print(
        f"Exact classical CVaR optimum: {[names[i] for i in exact_portfolio]}; "
        f"CVaR loss={exact_cvar:.11g}; VaR loss={var_losses[exact_pos]:.11g}"
    )

    rng = np.random.default_rng(RANDOM_SEED)
    all_rows = []
    restart_log_rows = []
    nfev_total = 0
    opt_started = time.perf_counter()

    for depth in DEPTH_SCHEDULE:
        print(f"\nOptimizing depth p={depth}")
        parameters, _, nfev, records = optimize_depth(
            depth, feasible_indices, scaled_costs, n, rng
        )
        nfev_total += nfev
        restart_log_rows.extend(records)

        parameters, scaled_objective, refinement_record = refine_parameters(
            parameters, depth, feasible_indices, scaled_costs, n
        )
        nfev_total += int(refinement_record["nfev"])
        restart_log_rows.append(refinement_record)

        PARAMETERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        parameter_names = (
            [f"gamma_{i + 1}" for i in range(depth)]
            + [f"beta_{i + 1}" for i in range(depth)]
        )
        pd.DataFrame({
            "parameter": parameter_names,
            "value": parameters.tolist(),
        }).to_csv(PARAMETERS_FILE, index=False)

        probabilities = simulate_qaoa(
            parameters, n, feasible_indices, scaled_costs, depth
        )
        order = np.argsort(probabilities)[::-1]
        modal_pos = int(order[0])
        modal_portfolio = portfolios[modal_pos]
        modal_cvar = float(cvar_losses[modal_pos])
        expected_cvar = float(np.dot(probabilities, cvar_losses))
        gap_pct = (
            (modal_cvar / exact_cvar - 1.0) * 100.0
            if abs(exact_cvar) > 1e-15 else float("nan")
        )
        top20 = order[:min(20, len(order))]
        best_top20_pos = int(top20[np.argmin(cvar_losses[top20])])

        print("\nFINAL RESULTS")
        print(f"Modal portfolio: {[names[i] for i in modal_portfolio]}")
        print(f"Modal probability: {probabilities[modal_pos]:.8f}")
        print(f"Modal CVaR loss: {modal_cvar:.11g}")
        print(f"Modal VaR loss: {var_losses[modal_pos]:.11g}")
        print(
            f"Modal probability rank: 1/{len(portfolios):,}; "
            f"CVaR gap: {gap_pct:.2f}%"
        )
        print(f"Probability of exact CVaR optimum: {probabilities[exact_pos]:.8f}")
        print(f"Expected CVaR under distribution: {expected_cvar:.11g}")
        print(
            "Best CVaR among top-20 probable states: "
            f"{[names[i] for i in portfolios[best_top20_pos]]} "
            f"(probability rank {int(np.where(order == best_top20_pos)[0][0]) + 1})"
        )
        print(f"Best scaled objective: {scaled_objective:.7f}")

        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)

        for pos, portfolio in enumerate(portfolios):
            all_rows.append({
                "depth": depth,
                "portfolio": "|".join(names[i] for i in portfolio),
                "assets": ", ".join(names[i] for i in portfolio),
                "probability": float(probabilities[pos]),
                "historical_var_loss": float(var_losses[pos]),
                "historical_cvar_loss": float(cvar_losses[pos]),
                "daily_variance": float(variances[pos]),
                "rank_by_probability": int(ranks[pos]),
                "is_exact_cvar_optimum": bool(pos == exact_pos),
                "scaled_cost": float(scaled_costs[pos]),
            })

    opt_seconds = time.perf_counter() - opt_started
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(RESULTS_FILE, index=False)
    pd.DataFrame(restart_log_rows).to_csv(RESTART_LOG_FILE, index=False)

    print(f"\nEvaluations: {nfev_total}")
    print(f"Optimization time: {opt_seconds:.2f}s")
    print(f"Total runtime: {time.perf_counter() - started:.2f}s")
    print(f"Saved portfolio distribution to: {RESULTS_FILE}")
    print(f"Saved restart diagnostics to: {RESTART_LOG_FILE}")
    print(f"Saved best parameters to: {PARAMETERS_FILE}")


if __name__ == "__main__":
    main()
