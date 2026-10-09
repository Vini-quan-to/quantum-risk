
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "final_report.md"


def read_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        print(f"Warning: {filename} not found; skipping that section.")
        return None
    return pd.read_csv(path)


def format_percent(value):
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.4f}%"


def main():
    benchmark = read_csv("benchmark_results.csv")
    comparison = read_csv("risk_comparison.csv")
    validation = read_csv("qubo_validation.csv")
    scalability = read_csv("scalability_results.csv")
    aer_results = read_csv("qaoa_aer_results.csv")

    lines = [
        "# QuantumRisk",
        "",
        "## 1. Project overview",
        "",
        "QuantumRisk investigates a portfolio-selection problem using "
        "a Quadratic Unconstrained Binary Optimization (QUBO) model "
        "and the Quantum Approximate Optimization Algorithm (QAOA).",
        "",
        "The portfolio contains exactly four equally weighted ETFs. "
        "The objective is to minimize historical portfolio variance. "
        "A classical exhaustive search provides a reference solution.",
        "",
        "## 2. Data and methodology",
        "",
        "- Assets: SPY, QQQ, IWM, EFA, EEM, TLT, GLD, and USO.",
        "- Data: historical daily asset returns stored in `daily_returns.csv`.",
        "- Portfolio constraint: select exactly four assets.",
        "- Portfolio weights: 25% per selected asset.",
        "- Risk objective: minimize estimated daily portfolio variance.",
        "- Quantum method: QAOA using Qiskit and a local Aer simulator.",
        "- Classical baseline: exhaustive enumeration of feasible portfolios.",
        "",
        "For a selected-asset vector x and covariance matrix Sigma, "
        "the objective is x^T Sigma x / 16, subject to sum(x_i) = 4 "
        "and x_i in {0, 1}.",
        "",
        "## 3. QUBO validation",
        "",
    ]

    if validation is not None:
        lines.extend([
            f"- Feasible portfolios checked: {len(validation)}.",
            "- The detailed validation is available in "
            "`qubo_validation.csv`.",
            "- The QUBO was validated against direct portfolio-variance "
            "calculations in the eight-asset model.",
        ])
    else:
        lines.append("Validation CSV was not found.")

    lines.extend(["", "## 4. Quantum versus classical risk", ""])

    if comparison is not None:
        lines.append(comparison.to_markdown(index=False))
    else:
        lines.append("Comparison CSV was not found.")

    lines.extend(["", "## 5. QAOA Aer experiment", ""])

    if aer_results is not None:
        if "matched_exact_optimum" in aer_results.columns:
            matches = int(
                aer_results["matched_exact_optimum"]
                .fillna(False)
                .astype(bool)
                .sum()
            )
            total = len(aer_results)
            lines.append(
                f"- Exact optimum found in {matches} of {total} recorded runs."
            )

        if "runtime_seconds" in aer_results.columns:
            mean_runtime = aer_results["runtime_seconds"].mean()
            lines.append(
                f"- Mean recorded QAOA runtime: {mean_runtime:.4f} seconds."
            )

        lines.extend([
            "",
            "Full results: `qaoa_aer_results.csv`.",
        ])
    else:
        lines.append("Aer experiment CSV was not found.")

    lines.extend(["", "## 6. Scalability experiment", ""])

    if scalability is not None:
        columns = [
            column for column in [
                "asset_count",
                "feasible_combinations",
                "matched_exact_optimum",
                "relative_variance_gap",
                "qaoa_runtime_seconds",
                "classical_runtime_seconds",
            ]
            if column in scalability.columns
        ]

        lines.append(scalability[columns].to_markdown(index=False))
        lines.extend([
            "",
            "These are small pilot instances. The results do not establish "
            "quantum advantage; the classical search is expected to be "
            "very effective at these problem sizes.",
        ])
    else:
        lines.append("Scalability CSV was not found.")

    lines.extend([
        "",
        "## 7. Visualizations",
        "",
        "- `figures/risk_comparison.png`",
        "- `figures/return_volatility.png`",
        "- `figures/qaoa_reliability.png`",
        "",
        "## 8. Limitations",
        "",
        "- Results depend on the historical sample and covariance estimate.",
        "- Historical risk metrics do not guarantee future performance.",
        "- The portfolio uses equal weights rather than optimized continuous weights.",
        "- QAOA results depend on circuit depth, optimizer settings, sampling, "
        "and penalty strength.",
        "- Small simulated instances do not demonstrate quantum speedup.",
        "",
        "## 9. Conclusion",
        "",
        "QuantumRisk demonstrates an end-to-end workflow for encoding a "
        "constrained portfolio-selection problem as QUBO, solving it with "
        "QAOA, validating the objective against classical enumeration, "
        "and comparing historical portfolio risk. Further experiments "
        "with larger instances, repeated seeds, and stronger classical "
        "baselines are required before drawing conclusions about scalability "
        "or quantum advantage.",
        "",
        "## Reproducibility",
        "",
        "Run the project scripts from the repository root using the project's "
        "Python virtual environment. The `data/` directory contains the "
        "experiment outputs and `figures/` contains the generated plots.",
        "",
    ])

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 60)
    print("QUANTUMRISK FINAL REPORT")
    print("=" * 60)
    print(f"Report saved to: {OUTPUT_FILE}")
    print(f"Report size: {OUTPUT_FILE.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

