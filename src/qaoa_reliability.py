
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qubo_model import build_qubo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "qaoa_reliability.csv"

NUMBER_OF_RUNS = 5
SHOTS = 1024
REPS = 1
MAX_ITERATIONS = 50
RANDOM_SEED = 2026


def main():
    print("=" * 65)
    print("QUANTUMRISK: QAOA RELIABILITY EXPERIMENT")
    print("=" * 65)

    original_problem, qubo, assets, covariance = build_qubo()

    # Exact classical optimum among feasible portfolios.
    benchmark_file = PROJECT_ROOT / "data" / "benchmark_results.csv"
    benchmark = pd.read_csv(benchmark_file)

    exact_row = benchmark.loc[benchmark["daily_variance"].idxmin()]
    exact_assets = set(exact_row["assets"].split(", "))
    exact_variance = float(exact_row["daily_variance"])

    print("\nExact classical optimum:")
    print(", ".join(sorted(exact_assets)))
    print(f"Daily variance: {exact_variance:.12g}")

    results = []

    for run_number in range(NUMBER_OF_RUNS):
        seed = RANDOM_SEED + run_number
        rng = np.random.default_rng(seed)

        # QAOA has two parameters per repetition.
        initial_point = rng.uniform(
            0,
            2 * np.pi,
            size=2 * REPS,
        )

        sampler = StatevectorSampler(
            default_shots=SHOTS,
            seed=seed,
        )

        qaoa = QAOA(
            sampler=sampler,
            optimizer=COBYLA(maxiter=MAX_ITERATIONS),
            reps=REPS,
            initial_point=initial_point,
        )

        solver = MinimumEigenOptimizer(qaoa)

        print(f"\nRun {run_number + 1}/{NUMBER_OF_RUNS}")
        result = solver.solve(qubo)

        selected = {
            assets[i]
            for i, bit in enumerate(np.rint(result.x).astype(int))
            if bit == 1
        }

        feasible = len(selected) == 4

        if feasible:
            indices = [assets.index(asset) for asset in selected]
            weights = np.full(4, 0.25)
            variance = float(
                weights @ covariance[np.ix_(indices, indices)] @ weights
            )
        else:
            variance = np.nan

        matched = feasible and selected == exact_assets

        print("Selected:", ", ".join(sorted(selected)))
        print("Feasible:", feasible)
        print(f"QUBO energy: {result.fval:.12g}")
        print("Matched exact optimum:", matched)

        results.append(
            {
                "run": run_number + 1,
                "seed": seed,
                "assets": ", ".join(sorted(selected)),
                "number_of_assets": len(selected),
                "feasible": feasible,
                "qubo_energy": float(result.fval),
                "daily_variance": variance,
                "matched_exact_optimum": matched,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    feasible_results = results_df[results_df["feasible"]]
    success_count = int(
        results_df["matched_exact_optimum"].sum()
    )

    print("\n" + "=" * 65)
    print("EXPERIMENT SUMMARY")
    print("=" * 65)
    print(f"Runs completed: {len(results_df)}")
    print(f"Feasible results: {len(feasible_results)}")
    print(f"Exact optimum found: {success_count}/{NUMBER_OF_RUNS}")
    print(
        "Observed exact-optimum rate: "
        f"{success_count / NUMBER_OF_RUNS:.1%}"
    )

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("\nReliability experiment completed.")


if __name__ == "__main__":
    main()
