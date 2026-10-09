
from itertools import combinations
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
OUTPUT_FILE = PROJECT_ROOT / "data" / "scalability_results.csv"

ASSET_COUNTS = [5, 6, 7, 8]
NUMBER_TO_SELECT = 4

SHOTS = 1024
REPS = 1
MAX_ITERATIONS = 50
BASE_SEED = 6200


# --------------------------------------------------
# Exact classical reference
# --------------------------------------------------

def find_exact_optimum(assets, covariance):
    """Enumerate every feasible equal-weight portfolio."""

    best_assets = None
    best_variance = float("inf")

    for selection in combinations(range(len(assets)), NUMBER_TO_SELECT):
        sub_covariance = covariance[np.ix_(selection, selection)]
        weights = np.full(NUMBER_TO_SELECT, 1.0 / NUMBER_TO_SELECT)

        variance = float(weights @ sub_covariance @ weights)

        if variance < best_variance:
            best_variance = variance
            best_assets = [assets[i] for i in selection]

    return best_assets, best_variance


# --------------------------------------------------
# QAOA experiment
# --------------------------------------------------

def run_qaoa(qubo, assets, covariance, seed, pass_manager):
    """Solve one instance using QAOA and return its metrics."""

    sampler = SamplerV2(
        default_shots=SHOTS,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(0, 2 * np.pi, size=2 * REPS)

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
    selected = [assets[i] for i, bit in enumerate(bits) if bit == 1]

    feasible = len(selected) == NUMBER_TO_SELECT

    if feasible:
        indices = [assets.index(asset) for asset in selected]
        sub_covariance = covariance[np.ix_(indices, indices)]
        weights = np.full(NUMBER_TO_SELECT, 1.0 / NUMBER_TO_SELECT)
        variance = float(weights @ sub_covariance @ weights)
    else:
        variance = np.nan

    return {
        "selected_assets": ", ".join(sorted(selected)),
        "selected_count": len(selected),
        "feasible": feasible,
        "qubo_energy": float(result.fval),
        "daily_variance": variance,
        "runtime_seconds": runtime,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("=" * 68)
    print("QUANTUMRISK: QAOA SCALABILITY EXPERIMENT")
    print("=" * 68)

    backend = AerSimulator()
    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend,
        seed_transpiler=123,
    )

    all_results = []

    for asset_count in ASSET_COUNTS:
        print(f"\n{'-' * 60}")
        print(f"Testing {asset_count} assets; selecting {NUMBER_TO_SELECT}")
        print(f"{'-' * 60}")

        _, qubo, assets, covariance = build_qubo(
            asset_limit=asset_count,
            number_of_assets_to_select=NUMBER_TO_SELECT,
        )

        exact_start = perf_counter()
        exact_assets, exact_variance = find_exact_optimum(
            assets,
            covariance,
        )
        exact_runtime = perf_counter() - exact_start

        print("Exact classical optimum:", ", ".join(exact_assets))
        print(f"Exact daily variance: {exact_variance:.12g}")
        print(f"Feasible combinations: {len(list(combinations(assets, NUMBER_TO_SELECT)))}")

        seed = BASE_SEED + asset_count

        try:
            quantum = run_qaoa(
                qubo,
                assets,
                covariance,
                seed,
                pass_manager,
            )

            matched = (
                quantum["feasible"]
                and set(quantum["selected_assets"].split(", "))
                == set(exact_assets)
            )

            relative_gap = (
                (quantum["daily_variance"] - exact_variance)
                / abs(exact_variance)
                if quantum["feasible"] and exact_variance != 0
                else np.nan
            )

            print("QAOA assets:", quantum["selected_assets"])
            print("QAOA feasible:", quantum["feasible"])
            print("Matched exact optimum:", matched)
            print(f"QAOA runtime: {quantum['runtime_seconds']:.3f} seconds")
            print(f"Classical runtime: {exact_runtime:.6f} seconds")

            all_results.append({
                "asset_count": asset_count,
                "selection_count": NUMBER_TO_SELECT,
                "feasible_combinations": len(
                    list(combinations(assets, NUMBER_TO_SELECT))
                ),
                "exact_assets": ", ".join(sorted(exact_assets)),
                "exact_daily_variance": exact_variance,
                "qaoa_assets": quantum["selected_assets"],
                "qaoa_feasible": quantum["feasible"],
                "qaoa_daily_variance": quantum["daily_variance"],
                "relative_variance_gap": relative_gap,
                "matched_exact_optimum": matched,
                "qaoa_runtime_seconds": quantum["runtime_seconds"],
                "classical_runtime_seconds": exact_runtime,
                "shots": SHOTS,
                "reps": REPS,
                "max_iterations": MAX_ITERATIONS,
                "seed": seed,
                "error": "",
            })

        except Exception as exc:
            print(f"QAOA failed for {asset_count} assets: {exc}")

            all_results.append({
                "asset_count": asset_count,
                "selection_count": NUMBER_TO_SELECT,
                "feasible_combinations": len(
                    list(combinations(assets, NUMBER_TO_SELECT))
                ),
                "exact_assets": ", ".join(sorted(exact_assets)),
                "exact_daily_variance": exact_variance,
                "qaoa_assets": "",
                "qaoa_feasible": False,
                "qaoa_daily_variance": np.nan,
                "relative_variance_gap": np.nan,
                "matched_exact_optimum": False,
                "qaoa_runtime_seconds": np.nan,
                "classical_runtime_seconds": exact_runtime,
                "shots": SHOTS,
                "reps": REPS,
                "max_iterations": MAX_ITERATIONS,
                "seed": seed,
                "error": str(exc),
            })

    results = pd.DataFrame(all_results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 68}")
    print("SCALABILITY SUMMARY")
    print(f"{'=' * 68}")

    print(
        results[
            [
                "asset_count",
                "feasible_combinations",
                "matched_exact_optimum",
                "relative_variance_gap",
                "qaoa_runtime_seconds",
                "classical_runtime_seconds",
            ]
        ].to_string(index=False)
    )

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("\nScalability experiment completed.")


if __name__ == "__main__":
    main()
