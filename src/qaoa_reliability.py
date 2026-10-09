
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qubo_model import build_qubo


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_FILE = DATA_DIR / "qaoa_reliability.csv"
SUMMARY_FILE = DATA_DIR / "qaoa_reliability_summary.csv"
BENCHMARK_FILE = DATA_DIR / "benchmark_results.csv"

NUMBER_OF_RUNS = 10
NUMBER_TO_SELECT = 4

SHOTS = 1024
REPS = 1
MAX_ITERATIONS = 50
BASE_SEED = 2026


# --------------------------------------------------
# Benchmark
# --------------------------------------------------

def load_exact_benchmark():
    """Load the exact minimum-variance portfolio from the benchmark CSV."""

    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {BENCHMARK_FILE}\n"
            "Run python src/classical_benchmark.py first."
        )

    benchmark = pd.read_csv(BENCHMARK_FILE)

    required = {"assets", "daily_variance"}
    missing = required - set(benchmark.columns)

    if missing:
        raise ValueError(
            f"Benchmark CSV is missing columns: {sorted(missing)}"
        )

    benchmark = benchmark.dropna(subset=["daily_variance"])

    if benchmark.empty:
        raise ValueError("Benchmark has no valid daily variance values.")

    row = benchmark.loc[benchmark["daily_variance"].idxmin()]

    exact_assets = set(str(row["assets"]).split(", "))
    exact_variance = float(row["daily_variance"])

    return exact_assets, exact_variance


# --------------------------------------------------
# Portfolio metrics
# --------------------------------------------------

def calculate_variance(assets, selected_assets, covariance):
    """Calculate equal-weight daily variance for a selected portfolio."""

    indices = [assets.index(asset) for asset in selected_assets]
    selected_covariance = covariance[np.ix_(indices, indices)]

    weights = np.full(len(indices), 1.0 / len(indices))

    return float(weights @ selected_covariance @ weights)


# --------------------------------------------------
# Run one trial
# --------------------------------------------------

def run_trial(
    qubo,
    assets,
    covariance,
    exact_assets,
    exact_variance,
    seed,
    pass_manager,
):
    """Run one independent QAOA trial using Qiskit Aer."""

    rng = np.random.default_rng(seed)

    initial_point = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=2 * REPS,
    )

    sampler = SamplerV2(
        default_shots=SHOTS,
        seed=seed,
    )

    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=MAX_ITERATIONS),
        reps=REPS,
        initial_point=initial_point,
        transpiler=pass_manager,
    )

    solver = MinimumEigenOptimizer(qaoa)

    start = perf_counter()
    result = solver.solve(qubo)
    runtime = perf_counter() - start

    bits = np.rint(result.x).astype(int)

    selected_assets = {
        assets[index]
        for index, bit in enumerate(bits)
        if bit == 1
    }

    selected_count = len(selected_assets)
    feasible = selected_count == NUMBER_TO_SELECT

    if feasible:
        variance = calculate_variance(
            assets,
            selected_assets,
            covariance,
        )

        absolute_gap = variance - exact_variance

        relative_gap = (
            absolute_gap / abs(exact_variance)
            if not np.isclose(exact_variance, 0.0)
            else np.nan
        )

        matched = selected_assets == exact_assets
    else:
        variance = np.nan
        absolute_gap = np.nan
        relative_gap = np.nan
        matched = False

    return {
        "seed": seed,
        "assets": ", ".join(sorted(selected_assets)),
        "number_of_assets": selected_count,
        "feasible": feasible,
        "qubo_energy": float(result.fval),
        "daily_variance": variance,
        "exact_daily_variance": exact_variance,
        "absolute_variance_gap": absolute_gap,
        "relative_variance_gap": relative_gap,
        "matched_exact_optimum": matched,
        "runtime_seconds": runtime,
        "shots": SHOTS,
        "reps": REPS,
        "max_iterations": MAX_ITERATIONS,
        "error": "",
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("=" * 68)
    print("QUANTUMRISK: QISKIT AER RELIABILITY EXPERIMENT")
    print("=" * 68)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    _, qubo, assets, covariance = build_qubo()

    if len(assets) < NUMBER_TO_SELECT:
        raise ValueError(
            f"Cannot select {NUMBER_TO_SELECT} assets from {len(assets)}."
        )

    exact_assets, exact_variance = load_exact_benchmark()

    if not exact_assets.issubset(set(assets)):
        raise ValueError(
            "The benchmark assets do not match the QUBO asset universe."
        )

    print(f"\nAvailable assets: {len(assets)}")
    print(f"Assets to select: {NUMBER_TO_SELECT}")
    print(f"Independent trials: {NUMBER_OF_RUNS}")
    print(f"Shots per evaluation: {SHOTS}")
    print(f"QAOA repetitions (p): {REPS}")
    print(f"Maximum optimizer iterations: {MAX_ITERATIONS}")

    print("\nExact classical optimum:")
    print(", ".join(sorted(exact_assets)))
    print(f"Exact daily variance: {exact_variance:.12g}")

    # Aer backend and transpilation pass manager.
    backend = AerSimulator()

    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend,
        seed_transpiler=BASE_SEED,
    )

    records = []

    for run_index in range(NUMBER_OF_RUNS):
        seed = BASE_SEED + run_index

        print(f"\n--- Trial {run_index + 1}/{NUMBER_OF_RUNS} ---")
        print(f"Random seed: {seed}")

        try:
            record = run_trial(
                qubo=qubo,
                assets=assets,
                covariance=covariance,
                exact_assets=exact_assets,
                exact_variance=exact_variance,
                seed=seed,
                pass_manager=pass_manager,
            )

            print(f"Selected assets: {record['assets']}")
            print(f"Feasible: {record['feasible']}")
            print(
                "Exact optimum matched: "
                f"{record['matched_exact_optimum']}"
            )

            if record["feasible"]:
                print(
                    "Daily variance: "
                    f"{record['daily_variance']:.12g}"
                )
                print(
                    "Relative variance gap: "
                    f"{record['relative_variance_gap']:.4%}"
                )

            print(f"Runtime: {record['runtime_seconds']:.3f} seconds")

        except Exception as exc:
            print(f"Trial failed: {type(exc).__name__}: {exc}")

            record = {
                "seed": seed,
                "assets": "",
                "number_of_assets": np.nan,
                "feasible": False,
                "qubo_energy": np.nan,
                "daily_variance": np.nan,
                "exact_daily_variance": exact_variance,
                "absolute_variance_gap": np.nan,
                "relative_variance_gap": np.nan,
                "matched_exact_optimum": False,
                "runtime_seconds": np.nan,
                "shots": SHOTS,
                "reps": REPS,
                "max_iterations": MAX_ITERATIONS,
                "error": f"{type(exc).__name__}: {exc}",
            }

        records.append(record)

    # Save detailed trial-level results.
    results = pd.DataFrame(records)
    results.to_csv(OUTPUT_FILE, index=False)

    completed_mask = results["error"].eq("")
    completed_count = int(completed_mask.sum())
    feasible_results = results[results["feasible"]]
    feasible_count = len(feasible_results)

    exact_count = int(results["matched_exact_optimum"].sum())

    valid_gaps = feasible_results["relative_variance_gap"].dropna()
    valid_runtimes = results.loc[
        completed_mask, "runtime_seconds"
    ].dropna()

    summary = {
        "asset_count": len(assets),
        "selection_count": NUMBER_TO_SELECT,
        "requested_runs": NUMBER_OF_RUNS,
        "completed_runs": completed_count,
        "failed_runs": NUMBER_OF_RUNS - completed_count,
        "feasible_runs": feasible_count,
        "feasibility_rate": feasible_count / NUMBER_OF_RUNS,
        "exact_optimum_runs": exact_count,
        "exact_optimum_rate": exact_count / NUMBER_OF_RUNS,
        "exact_optimum_rate_among_feasible": (
            exact_count / feasible_count if feasible_count else np.nan
        ),
        "mean_relative_variance_gap": (
            float(valid_gaps.mean()) if not valid_gaps.empty else np.nan
        ),
        "best_daily_variance": (
            float(feasible_results["daily_variance"].min())
            if feasible_count else np.nan
        ),
        "mean_daily_variance": (
            float(feasible_results["daily_variance"].mean())
            if feasible_count else np.nan
        ),
        "mean_runtime_seconds": (
            float(valid_runtimes.mean())
            if not valid_runtimes.empty else np.nan
        ),
        "runtime_std_seconds": (
            float(valid_runtimes.std(ddof=1))
            if len(valid_runtimes) > 1 else 0.0
        ),
        "shots": SHOTS,
        "reps": REPS,
        "max_iterations": MAX_ITERATIONS,
        "base_seed": BASE_SEED,
    }

    pd.DataFrame([summary]).to_csv(SUMMARY_FILE, index=False)

    # Print summary.
    print("\n" + "=" * 68)
    print("RELIABILITY SUMMARY")
    print("=" * 68)

    print(f"Requested trials: {NUMBER_OF_RUNS}")
    print(f"Completed trials: {completed_count}/{NUMBER_OF_RUNS}")
    print(f"Failed trials: {NUMBER_OF_RUNS - completed_count}")
    print(f"Feasible portfolios: {feasible_count}/{NUMBER_OF_RUNS}")
    print(f"Exact optimum found: {exact_count}/{NUMBER_OF_RUNS}")
    print(f"Observed exact-optimum rate: {exact_count / NUMBER_OF_RUNS:.1%}")

    if feasible_count:
        print(
            "Exact-optimum rate among feasible results: "
            f"{exact_count / feasible_count:.1%}"
        )

    if not valid_gaps.empty:
        print(f"Mean relative variance gap: {valid_gaps.mean():.4%}")

    if not valid_runtimes.empty:
        print(f"Mean QAOA runtime: {valid_runtimes.mean():.3f} seconds")
        print(
            "Runtime standard deviation: "
            f"{valid_runtimes.std(ddof=1):.3f} seconds"
        )

    print(f"\nDetailed results saved to: {OUTPUT_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print("\nAer reliability experiment completed.")


if __name__ == "__main__":
    main()
