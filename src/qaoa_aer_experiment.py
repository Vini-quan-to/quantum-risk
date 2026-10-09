
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "qaoa_aer_results.csv"
BENCHMARK_FILE = PROJECT_ROOT / "data" / "benchmark_results.csv"

NUMBER_OF_RUNS = 5
SHOTS = 1024
REPS = 1
MAX_ITERATIONS = 50
BASE_SEED = 4100


def main():
    print("=" * 65)
    print("QUANTUMRISK: QAOA WITH QISKIT AER")
    print("=" * 65)

    _, qubo, assets, covariance = build_qubo()

    benchmark = pd.read_csv(BENCHMARK_FILE)
    exact_row = benchmark.loc[benchmark["daily_variance"].idxmin()]
    exact_assets = set(exact_row["assets"].split(", "))
    exact_variance = float(exact_row["daily_variance"])

    # Prepare a transpiler that decomposes QAOA into gates
    # supported by the Aer simulator.
    backend = AerSimulator()

    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend,
        seed_transpiler=123,
    )

    print("\nExact classical optimum:", ", ".join(sorted(exact_assets)))
    print(f"Exact daily variance: {exact_variance:.12g}")
    print(f"Shots per circuit evaluation: {SHOTS}")
    print(f"QAOA repetitions (p): {REPS}")
    print(f"Optimizer iterations per run: up to {MAX_ITERATIONS}")
    print("Aer circuit transpilation: enabled")

    results = []

    for run_number in range(NUMBER_OF_RUNS):
        seed = BASE_SEED + run_number
        rng = np.random.default_rng(seed)

        initial_point = rng.uniform(
            0,
            2 * np.pi,
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

        print(f"\n--- Run {run_number + 1}/{NUMBER_OF_RUNS} ---")

        start = perf_counter()
        result = solver.solve(qubo)
        elapsed = perf_counter() - start

        solution = np.rint(result.x).astype(int)

        selected_assets = {
            assets[i]
            for i, bit in enumerate(solution)
            if bit == 1
        }

        feasible = len(selected_assets) == 4

        if feasible:
            indices = [
                assets.index(asset)
                for asset in selected_assets
            ]

            sub_covariance = covariance[np.ix_(indices, indices)]
            weights = np.full(4, 0.25)

            variance = float(
                weights @ sub_covariance @ weights
            )
        else:
            variance = np.nan

        matched = feasible and selected_assets == exact_assets

        print("Selected assets:", ", ".join(sorted(selected_assets)))
        print("Feasible:", feasible)
        print(f"QUBO energy: {result.fval:.12g}")
        print(f"Daily variance: {variance:.12g}")
        print("Matched exact optimum:", matched)
        print(f"Runtime: {elapsed:.2f} seconds")

        results.append({
            "run": run_number + 1,
            "seed": seed,
            "shots": SHOTS,
            "reps": REPS,
            "max_iterations": MAX_ITERATIONS,
            "assets": ", ".join(sorted(selected_assets)),
            "number_of_assets": len(selected_assets),
            "feasible": feasible,
            "qubo_energy": float(result.fval),
            "daily_variance": variance,
            "matched_exact_optimum": matched,
            "runtime_seconds": elapsed,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    successful = int(results_df["matched_exact_optimum"].sum())
    feasible_count = int(results_df["feasible"].sum())

    print("\n" + "=" * 65)
    print("AER EXPERIMENT SUMMARY")
    print("=" * 65)
    print(f"Runs completed: {len(results_df)}")
    print(f"Feasible solutions: {feasible_count}/{NUMBER_OF_RUNS}")
    print(f"Exact optimum found: {successful}/{NUMBER_OF_RUNS}")
    print(
        "Observed exact-optimum rate: "
        f"{successful / NUMBER_OF_RUNS:.1%}"
    )
    print(
        "Mean runtime: "
        f"{results_df['runtime_seconds'].mean():.2f} seconds"
    )
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("\nAer experiment completed.")


if __name__ == "__main__":
    main()
