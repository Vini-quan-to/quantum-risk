"""QuantumRisk: fixed-cardinality XY-mixer QAOA-style statevector experiment.

This is a classical NumPy simulation, not execution on quantum hardware.
The portfolio problem is fixed-size and equal-weight; exact enumeration is
used as a benchmark when feasible.
"""
from itertools import combinations
from math import comb
from pathlib import Path
import time

import numpy as np
import pandas as pd
from types import SimpleNamespace

# ---------------------------- Configuration ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RETURNS_FILE = DATA_DIR / "daily_returns.csv"
RESULTS_FILE = DATA_DIR / "quantum_portfolio_results.csv"
COMPARISON_FILE = DATA_DIR / "portfolio_size_comparison.csv"
TOP_PORTFOLIOS_FILE = DATA_DIR / "qaoa_top_portfolios.csv"

PORTFOLIO_SIZE = 4
# Keep the first comparison controlled. Set (1, 2, 3) to test depth growth.
DEPTH_SCHEDULE = (1,)
OBJECTIVE = "expectation"  # "expectation" or "lower_tail_cvar"
CVAR_ALPHA = 0.20            # fraction of probability mass in the lowest-cost tail
OPTIMIZER_RESTARTS = 5
MAX_ITERATIONS_PER_RESTART = 40
TOP_READOUT_COUNT = 20
RANDOM_SEED = 2026
MAX_QUBITS = 20
TRADING_DAYS = 252
TOP_RISK_FRACTION = 0.01


def minimize_without_scipy(fun, x0, args=(), options=None):
    """Small derivative-free coordinate/pattern search; avoids SciPy DLLs.

    This is a fallback optimizer, not a drop-in implementation of COBYLA.
    It evaluates +/- coordinate moves and shrinks the step when no move helps.
    """
    options = options or {}
    max_evaluations = int(options.get("maxiter", 40))
    step = float(options.get("rhobeg", 0.2))
    tolerance = float(options.get("tol", 1e-5))
    x = np.asarray(x0, dtype=float).copy()
    best_value = float(fun(x, *args))
    evaluations = 1
    while evaluations < max_evaluations and step > tolerance:
        improved = False
        for j in range(len(x)):
            for direction in (-1.0, 1.0):
                if evaluations >= max_evaluations:
                    break
                candidate = x.copy()
                candidate[j] += direction * step
                value = float(fun(candidate, *args))
                evaluations += 1
                if np.isfinite(value) and value < best_value:
                    x, best_value = candidate, value
                    improved = True
        if not improved:
            step *= 0.5
    return SimpleNamespace(
        x=x, fun=best_value, nfev=evaluations, success=np.isfinite(best_value),
        message="Fallback derivative-free coordinate search completed"
    )


def load_returns():
    if not RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {RETURNS_FILE}. Place daily_returns.csv in the data folder."
        )
    returns = pd.read_csv(RETURNS_FILE)
    date_columns = [c for c in returns.columns if str(c).strip().lower() in {"date", "datetime", "timestamp"}]
    returns = returns.drop(columns=date_columns)
    returns = returns.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if returns.empty or len(returns) < 2:
        raise ValueError("At least two valid return observations are required.")
    if returns.columns.duplicated().any():
        raise ValueError("Duplicate asset names detected in the CSV.")
    if returns.shape[1] > MAX_QUBITS:
        raise ValueError(f"Supports at most {MAX_QUBITS} assets; found {returns.shape[1]}.")
    if not 1 <= PORTFOLIO_SIZE <= returns.shape[1]:
        raise ValueError("PORTFOLIO_SIZE must be between 1 and the number of assets.")
    if not np.isfinite(returns.to_numpy(dtype=np.float64)).all():
        raise ValueError("Returns contain invalid numerical values.")
    if (returns <= -1.0).any().any():
        raise ValueError("A return <= -100% is invalid for wealth/drawdown calculations.")
    return returns


def make_feasible_basis(asset_count, portfolio_size):
    """Return statevector indices and bit rows with exactly k selected bits."""
    indices = np.arange(1 << asset_count, dtype=np.uint32)
    positions = np.arange(asset_count, dtype=np.uint32)
    bits = ((indices[:, None] >> positions[None, :]) & 1).astype(np.uint8)
    mask = bits.sum(axis=1) == portfolio_size
    feasible_indices, feasible_bits = indices[mask], bits[mask].astype(bool)
    expected = comb(asset_count, portfolio_size)
    if len(feasible_indices) != expected:
        raise RuntimeError(f"Expected {expected} feasible states, got {len(feasible_indices)}.")
    return feasible_indices, feasible_bits


def build_feasible_costs(covariance, feasible_bits, portfolio_size):
    """Return normalized x'Cov x costs and scale; constant equal weights preserve ranking."""
    scale = float(np.max(np.abs(covariance)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Covariance scale is zero or invalid; check returns data.")
    x = feasible_bits.astype(np.float64)
    costs = np.einsum("bi,ij,bj->b", x, covariance / scale, x, optimize=True)
    costs /= portfolio_size ** 2
    return costs, scale


def apply_xy_gate(state, beta, qubit_a, qubit_b, qubit_count):
    """Apply exp[-i beta (XX+YY)/2], preserving Hamming weight."""
    if not (0 <= qubit_a < qubit_count and 0 <= qubit_b < qubit_count):
        raise ValueError("Qubit index is outside the statevector.")
    if qubit_a == qubit_b:
        return state
    mask_a, mask_b = 1 << qubit_a, 1 << qubit_b
    indices = np.arange(len(state), dtype=np.uint32)
    index_01 = indices[((indices & mask_a) == 0) & ((indices & mask_b) != 0)]
    index_10 = index_01 ^ mask_a ^ mask_b
    amp_01, amp_10 = state[index_01].copy(), state[index_10].copy()
    c, s = np.cos(beta), -1j * np.sin(beta)
    state[index_01] = c * amp_01 + s * amp_10
    state[index_10] = c * amp_10 + s * amp_01
    return state


def simulate_feasible_qaoa(parameters, costs, feasible_indices, asset_count, reps):
    if reps < 1 or len(parameters) != 2 * reps:
        raise ValueError(f"Expected {2 * reps} parameters for reps={reps}.")
    if len(costs) != len(feasible_indices):
        raise ValueError("Cost count does not match feasible basis size.")
    gammas, betas = parameters[:reps], parameters[reps:]
    state = np.zeros(1 << asset_count, dtype=np.complex128)
    state[feasible_indices] = 1 / np.sqrt(len(feasible_indices))
    for layer in range(reps):
        state[feasible_indices] *= np.exp(-1j * gammas[layer] * costs)
        for qubit in range(asset_count):
            apply_xy_gate(state, betas[layer], qubit, (qubit + 1) % asset_count, asset_count)
    return state


def lower_tail_cvar(probabilities, costs, alpha=CVAR_ALPHA):
    """Probability-weighted mean cost in the best alpha probability mass.

    The fractional boundary state is included, so total mass is exactly alpha.
    This is a lower-tail objective (not conventional upper-tail loss CVaR).
    """
    if not 0 < alpha <= 1:
        raise ValueError("CVAR_ALPHA must be in (0, 1].")
    p = np.asarray(probabilities, dtype=float)
    c = np.asarray(costs, dtype=float)
    order = np.argsort(c, kind="stable")
    remaining, weighted_cost = float(alpha), 0.0
    for idx in order:
        take = min(float(p[idx]), remaining)
        weighted_cost += take * float(c[idx])
        remaining -= take
        if remaining <= 1e-14:
            break
    if remaining > 1e-8:
        raise ValueError("Probability mass is less than alpha.")
    return weighted_cost / alpha


def qaoa_objective(parameters, costs, feasible_indices, asset_count, reps, objective=OBJECTIVE, alpha=CVAR_ALPHA):
    state = simulate_feasible_qaoa(parameters, costs, feasible_indices, asset_count, reps)
    probabilities = np.abs(state[feasible_indices]) ** 2
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0 or not np.isfinite(probabilities).all():
        return float("inf")
    probabilities /= total
    if objective == "expectation":
        value = float(np.dot(probabilities, costs))
    elif objective == "lower_tail_cvar":
        value = lower_tail_cvar(probabilities, costs, alpha)
    else:
        raise ValueError("OBJECTIVE must be 'expectation' or 'lower_tail_cvar'.")
    return value if np.isfinite(value) else float("inf")


def exact_classical_optimum(covariance, feasible_bits, assets, portfolio_size):
    x = feasible_bits.astype(np.float64)
    variances = np.einsum("bi,ij,bj->b", x, covariance, x, optimize=True) / portfolio_size ** 2
    best_idx = int(np.argmin(variances))
    selected = [assets[i] for i in np.flatnonzero(feasible_bits[best_idx])]
    return selected, float(variances[best_idx]), variances


def evaluate_portfolio(returns, selected_assets):
    if len(selected_assets) != PORTFOLIO_SIZE:
        raise ValueError(f"Expected exactly {PORTFOLIO_SIZE} selected assets; got {len(selected_assets)}.")
    r = returns[selected_assets].mean(axis=1)
    losses = -r.to_numpy()
    var95 = float(np.quantile(losses, 0.95))
    tail = losses[losses >= var95]
    wealth = (1 + r).cumprod()
    drawdowns = 1 - wealth / wealth.cummax()
    return {
        "portfolio_size": len(selected_assets),
        "equal_weight_variance": float(r.var(ddof=1)),
        "daily_mean_return": float(r.mean()),
        "annualized_mean_return": float(r.mean() * TRADING_DAYS),
        "annualized_volatility": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "historical_var_95": var95,
        "historical_cvar_95": float(tail.mean()),
        "maximum_drawdown": float(drawdowns.max()),
    }


def summarize_probability_distribution(probabilities, variances, top_risk_fraction=TOP_RISK_FRACTION):
    p = np.asarray(probabilities, dtype=float)
    v = np.asarray(variances, dtype=float)
    if p.shape != v.shape or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError("Invalid or misaligned probability distribution and variances.")
    p = p / p.sum()
    count = len(v)
    top_count = max(1, int(np.ceil(count * top_risk_fraction)))
    low_idx = np.argsort(v, kind="stable")[:top_count]
    order = np.argsort(v, kind="stable")
    ranks = np.empty(count, dtype=float)
    ranks[order] = np.arange(1, count + 1, dtype=float)
    return {
        "qaoa_expected_daily_variance": float(np.dot(p, v)),
        "qaoa_probability_mass_lowest_risk_1pct": float(p[low_idx].sum()),
        "qaoa_expected_risk_percentile_rank": float(np.dot(p, ranks / count * 100)),
        "lowest_risk_portfolio_count_for_mass_metric": top_count,
    }


def interpolate_parameters(previous, new_reps, rng):
    """Warm-start a deeper circuit by retaining old angles and initializing new layers."""
    old_reps = len(previous) // 2
    if new_reps <= old_reps:
        return previous[:new_reps].copy().tolist() + previous[old_reps:old_reps + new_reps].copy().tolist()
    gammas = list(previous[:old_reps]) + list(rng.uniform(0, np.pi, new_reps - old_reps))
    betas = list(previous[old_reps:]) + list(rng.uniform(0, np.pi / 2, new_reps - old_reps))
    return np.asarray(gammas + betas, dtype=float)


def main():
    start_time = time.perf_counter()
    rng = np.random.default_rng(RANDOM_SEED)
    print("=" * 76)
    print("QUANTUMRISK: FIXED-CARDINALITY XY-MIXER STATEVECTOR EXPERIMENT")
    print("=" * 76)
    returns = load_returns()
    assets, asset_count = list(returns.columns), len(returns.columns)
    print(f"Assets: {asset_count} | observations: {len(returns)} | k: {PORTFOLIO_SIZE}")
    print(f"Depth schedule: {DEPTH_SCHEDULE} | objective: {OBJECTIVE}")
    print(f"Restarts per depth: {OPTIMIZER_RESTARTS} | max evaluations/restart: {MAX_ITERATIONS_PER_RESTART}")

    covariance = returns.cov().to_numpy(dtype=np.float64)
    covariance = (covariance + covariance.T) / 2
    feasible_indices, feasible_bits = make_feasible_basis(asset_count, PORTFOLIO_SIZE)
    print(f"Feasible portfolios: {len(feasible_indices):,}")
    costs, covariance_scale = build_feasible_costs(covariance, feasible_bits, PORTFOLIO_SIZE)
    _, classical_variance, all_variances = exact_classical_optimum(covariance, feasible_bits, assets, PORTFOLIO_SIZE)
    # The exact portfolio is recovered explicitly for output clarity.
    exact_idx = int(np.argmin(all_variances))
    exact_assets = [assets[i] for i in np.flatnonzero(feasible_bits[exact_idx])]
    print(f"Exact classical optimum: {exact_assets}; variance={classical_variance:.12g}")

    best_result, best_reps = None, None
    previous_best_parameters = None
    total_evaluations = 0
    optimization_start = time.perf_counter()
    depth_rows = []
    for reps in DEPTH_SCHEDULE:
        if not 1 <= reps <= 4:
            raise ValueError("For this initial experiment, DEPTH_SCHEDULE values must be 1..4.")
        if previous_best_parameters is not None:
            warm = interpolate_parameters(previous_best_parameters, reps, rng)
            starts = [warm] + [np.concatenate([rng.uniform(0, np.pi, reps), rng.uniform(0, np.pi / 2, reps)]) for _ in range(max(0, OPTIMIZER_RESTARTS - 1))]
        else:
            starts = [np.concatenate([rng.uniform(0, np.pi, reps), rng.uniform(0, np.pi / 2, reps)]) for _ in range(OPTIMIZER_RESTARTS)]
        depth_best = None
        print(f"\nOptimizing depth p={reps}")
        for restart, initial in enumerate(starts, start=1):
            result = minimize_without_scipy(
                qaoa_objective, initial,
                args=(costs, feasible_indices, asset_count, reps, OBJECTIVE, CVAR_ALPHA),
                options={"maxiter": MAX_ITERATIONS_PER_RESTART, "rhobeg": 0.2, "tol": 1e-5},
            )
            total_evaluations += int(getattr(result, "nfev", 0))
            print(f"  start {restart}/{len(starts)}: objective={result.fun:.10g}, nfev={getattr(result, 'nfev', '?')}")
            if np.isfinite(result.fun) and (depth_best is None or result.fun < depth_best.fun):
                depth_best = result
        if depth_best is None:
            raise RuntimeError(f"All optimization starts failed at p={reps}.")
        previous_best_parameters = np.asarray(depth_best.x, dtype=float)
        best_result, best_reps = depth_best, reps
        depth_rows.append({"depth": reps, "objective": OBJECTIVE, "objective_value": float(depth_best.fun), "nfev_total_cumulative": total_evaluations, "optimizer_success": bool(depth_best.success), "optimizer_message": str(depth_best.message)})

    optimization_time = time.perf_counter() - optimization_start
    state = simulate_feasible_qaoa(best_result.x, costs, feasible_indices, asset_count, best_reps)
    probabilities = np.abs(state[feasible_indices]) ** 2
    p_sum = float(probabilities.sum())
    if not np.isfinite(p_sum) or p_sum <= 0:
        raise RuntimeError("Final QAOA probability distribution is invalid.")
    probabilities /= p_sum
    modal_idx = int(np.argmax(probabilities))
    modal_assets = [assets[i] for i in np.flatnonzero(feasible_bits[modal_idx])]
    modal_variance = float(all_variances[modal_idx])
    modal_probability = float(probabilities[modal_idx])
    modal_rank = int(np.sum(all_variances < modal_variance) + 1)
    gap_percent = (modal_variance - classical_variance) / classical_variance * 100 if classical_variance > 0 else float("nan")
    modal_metrics = evaluate_portfolio(returns, modal_assets)
    distribution_metrics = summarize_probability_distribution(probabilities, all_variances)

    # Hybrid readout: first select the most probable states, then rank only those by actual historical risk.
    top_n = min(TOP_READOUT_COUNT, len(probabilities))
    top_idx = np.argsort(probabilities)[::-1][:top_n]
    top_rows = []
    for probability_rank, idx in enumerate(top_idx, start=1):
        selected = [assets[i] for i in np.flatnonzero(feasible_bits[idx])]
        risk_rank = int(np.sum(all_variances < all_variances[idx]) + 1)
        metrics = evaluate_portfolio(returns, selected)
        top_rows.append({
            "probability_rank": probability_rank,
            "risk_rank_among_all_feasible": risk_rank,
            "selected_assets": ", ".join(selected),
            "state_probability": float(probabilities[idx]),
            "daily_variance": float(all_variances[idx]),
            "variance_gap_vs_exact_percent": float((all_variances[idx] - classical_variance) / classical_variance * 100) if classical_variance > 0 else float("nan"),
            **metrics,
        })
    top_df = pd.DataFrame(top_rows)
    hybrid_best = top_df.sort_values("daily_variance", ascending=True).iloc[0]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"asset": modal_assets, "weight": np.full(len(modal_assets), 1 / len(modal_assets))}).to_csv(RESULTS_FILE, index=False)
    pd.DataFrame(top_rows).to_csv(TOP_PORTFOLIOS_FILE, index=False)
    pd.DataFrame(depth_rows).to_csv(DATA_DIR / "qaoa_depth_results.csv", index=False)

    total_time = time.perf_counter() - start_time
    row = {
        **modal_metrics, **distribution_metrics,
        "selected_assets": ", ".join(modal_assets),
        "selected_state_probability": modal_probability,
        "qaoa_objective": OBJECTIVE,
        "qaoa_cvar_alpha": CVAR_ALPHA if OBJECTIVE == "lower_tail_cvar" else np.nan,
        "qaoa_best_objective_value": float(best_result.fun),
        "optimizer_function_evaluations": total_evaluations,
        "optimizer_restarts_per_depth": OPTIMIZER_RESTARTS,
        "max_iterations_per_restart": MAX_ITERATIONS_PER_RESTART,
        "qaoa_repetitions": best_reps,
        "random_seed": RANDOM_SEED,
        "optimization_runtime_seconds": optimization_time,
        "total_runtime_seconds": total_time,
        "portfolio_rank": modal_rank,
        "exact_classical_assets": ", ".join(exact_assets),
        "exact_classical_daily_variance": classical_variance,
        "variance_gap_percent": gap_percent,
        "probability_of_exact_optimum": float(probabilities[exact_idx]),
        "hybrid_top_readout_count": top_n,
        "best_variance_in_top_probability_readout_assets": str(hybrid_best["selected_assets"]),
        "best_variance_in_top_probability_readout": float(hybrid_best["daily_variance"]),
        "best_variance_in_top_readout_risk_rank": int(hybrid_best["risk_rank_among_all_feasible"]),
        "covariance_normalization_scale": covariance_scale,
        "optimizer_success": bool(best_result.success),
        "optimizer_message": str(best_result.message),
        "method_note": "Classical NumPy statevector simulation; fixed-cardinality XY mixer; classical derivative-free coordinate search (SciPy-free fallback); no quantum hardware or quantum-advantage claim.",
    }
    pd.DataFrame([row]).to_csv(COMPARISON_FILE, index=False)

    print("\n" + "=" * 76 + "\nFINAL RESULTS\n" + "=" * 76)
    print(f"Modal portfolio: {modal_assets}")
    print(f"Modal probability: {modal_probability:.8f}")
    print(f"Modal daily variance: {modal_variance:.12g}")
    print(f"Modal rank: {modal_rank:,}/{len(feasible_indices):,}; variance gap: {gap_percent:.2f}%")
    print(f"Probability of exact optimum: {probabilities[exact_idx]:.8f}")
    print(f"Expected variance under distribution: {distribution_metrics['qaoa_expected_daily_variance']:.12g}")
    print(f"Best variance among top-{top_n} probable states: {hybrid_best['selected_assets']} (rank {int(hybrid_best['risk_rank_among_all_feasible'])})")
    print(f"Evaluations: {total_evaluations}; optimization: {optimization_time:.2f}s; total: {total_time:.2f}s")
    print(f"Saved modal portfolio: {RESULTS_FILE}")
    print(f"Saved metrics: {COMPARISON_FILE}")
    print(f"Saved hybrid readout: {TOP_PORTFOLIOS_FILE}")
    print(f"Saved depth history: {DATA_DIR / 'qaoa_depth_results.csv'}")
    print("Note: results are historical-data estimates for equal-weight portfolios, not future-performance guarantees.")
    print("Note: this is classical simulation, not quantum hardware or evidence of quantum advantage.")


if __name__ == "__main__":
    main()
