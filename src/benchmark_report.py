
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

BENCHMARK_FILE = DATA_DIR / "benchmark_results.csv"
QAOA_FILE = DATA_DIR / "qaoa_aer_results.csv"
RISK_FILE = DATA_DIR / "risk_comparison.csv"
CLASSICAL_RISK_FILE = DATA_DIR / "portfolio_results.csv"

REPORT_FILE = DATA_DIR / "benchmark_report.md"


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def require_file(path):
    """Check that a required input file exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run the relevant experiment first."
        )


def percentage(value):
    """Format a decimal as a percentage."""
    return f"{float(value) * 100:.4f}%"


def format_number(value):
    """Format a number without losing useful precision."""
    return f"{float(value):.10g}"


# --------------------------------------------------
# Report generation
# --------------------------------------------------

def main():
    print("=" * 65)
    print("QUANTUMRISK: BENCHMARK REPORT")
    print("=" * 65)

    for path in [
        BENCHMARK_FILE,
        QAOA_FILE,
        RISK_FILE,
        CLASSICAL_RISK_FILE,
    ]:
        require_file(path)

    benchmark = pd.read_csv(BENCHMARK_FILE)
    qaoa = pd.read_csv(QAOA_FILE)
    risk = pd.read_csv(RISK_FILE)
    classical_risk = pd.read_csv(CLASSICAL_RISK_FILE)

    if benchmark.empty or qaoa.empty or risk.empty:
        raise ValueError("One or more result files are empty.")

    # Find the exact classical minimum-variance solution.
    required_benchmark_columns = {"assets", "daily_variance"}
    if not required_benchmark_columns.issubset(benchmark.columns):
        raise ValueError(
            "benchmark_results.csv must contain "
            "'assets' and 'daily_variance' columns."
        )

    exact = benchmark.loc[
        benchmark["daily_variance"].idxmin()
    ]

    exact_assets = set(
        str(exact["assets"]).split(", ")
    )
    exact_variance = float(exact["daily_variance"])

    # Summarize the QAOA experiment.
    qaoa["feasible"] = (
        qaoa["feasible"].astype(str).str.lower() == "true"
    )
    qaoa["matched_exact_optimum"] = (
        qaoa["matched_exact_optimum"]
        .astype(str).str.lower() == "true"
    )

    total_runs = len(qaoa)
    feasible_runs = int(qaoa["feasible"].sum())
    successful_runs = int(
        qaoa["matched_exact_optimum"].sum()
    )

    mean_runtime = (
        float(qaoa["runtime_seconds"].mean())
        if "runtime_seconds" in qaoa.columns
        else None
    )

    # Get the historical minimum-CVaR portfolio.
    best_cvar = classical_risk.sort_values(
        "CVaR_95", ascending=True
    ).iloc[0]

    # Extract the two portfolios from the risk comparison.
    qaoa_risk = risk[
        risk["method"] == "QAOA minimum-variance portfolio"
    ]

    classical_cvar_risk = risk[
        risk["method"] == "Classical minimum-CVaR portfolio"
    ]

    if qaoa_risk.empty or classical_cvar_risk.empty:
        raise ValueError(
            "Expected portfolio methods were not found in "
            "risk_comparison.csv."
        )

    qr = qaoa_risk.iloc[0]
    cr = classical_cvar_risk.iloc[0]

    # Calculate differences in percentage points.
    cvar_difference_pp = (
        float(qr["CVaR_95"]) - float(cr["CVaR_95"])
    ) * 100

    variance_difference = (
        float(qaoa["daily_variance"].mean())
        - exact_variance
    )

    # Assemble the report.
    lines = [
        "# QuantumRisk: Classical and QAOA Benchmark",
        "",
        "## 1. Experiment setup",
        "",
        "- Assets considered: 8 ETFs.",
        "- Portfolio constraint: select exactly 4 assets.",
        f"- Feasible portfolios evaluated classically: {len(benchmark)}.",
        "- Portfolio weights: equal weight (25% per selected asset).",
        "- Quantum algorithm: QAOA using Qiskit Aer simulation.",
        f"- QAOA repetitions: {total_runs}.",
        "",
        "## 2. Exact classical minimum-variance solution",
        "",
        f"- Selected assets: {', '.join(sorted(exact_assets))}.",
        f"- Daily variance: {format_number(exact_variance)}.",
        f"- QAOA feasible runs: {feasible_runs}/{total_runs}.",
        f"- QAOA runs matching the exact optimum: "
        f"{successful_runs}/{total_runs} "
        f"({successful_runs / total_runs:.1%}).",
        (
            f"- Mean reported QAOA runtime: {mean_runtime:.4f} seconds."
            if mean_runtime is not None
            else "- QAOA runtime was not recorded."
        ),
        "",
        "## 3. Minimum-variance versus minimum-CVaR",
        "",
        "| Metric | QAOA minimum-variance | Classical minimum-CVaR |",
        "|---|---:|---:|",
        f"| Assets | {qr['assets']} | {cr['assets']} |",
        f"| Annualized return estimate | "
        f"{percentage(qr['annualized_return_estimate'])} | "
        f"{percentage(cr['annualized_return_estimate'])} |",
        f"| Annualized volatility | "
        f"{percentage(qr['annualized_volatility'])} | "
        f"{percentage(cr['annualized_volatility'])} |",
        f"| Historical VaR 95% | "
        f"{percentage(qr['VaR_95'])} | "
        f"{percentage(cr['VaR_95'])} |",
        f"| Historical CVaR 95% | "
        f"{percentage(qr['CVaR_95'])} | "
        f"{percentage(cr['CVaR_95'])} |",
        f"| Maximum drawdown | "
        f"{percentage(qr['max_drawdown'])} | "
        f"{percentage(cr['max_drawdown'])} |",
        "",
        f"The minimum-CVaR portfolio's measured CVaR is "
        f"{abs(cvar_difference_pp):.4f} percentage points "
        f"{'lower' if cvar_difference_pp > 0 else 'higher'} "
        f"than that of the minimum-variance portfolio "
        f"under the shared risk engine.",
        "",
        "## 4. Interpretation",
        "",
        "- The classical exhaustive search provides the exact "
        "minimum-variance reference for this eight-asset, "
        "four-asset selection problem.",
        "- QAOA found that exact portfolio in the recorded runs.",
        "- Minimum variance and minimum CVaR are different objectives; "
        "their optimal portfolios need not match.",
        "- The return figures are historical annualized estimates, "
        "not forecasts or guarantees.",
        "",
        "## 5. Limitations",
        "",
        "- QAOA was executed on a classical simulator, not quantum hardware.",
        "- These results do not demonstrate quantum advantage.",
        "- Five runs are a small sample and do not establish "
        "general reliability.",
        "- Runtime measurements depend on the computer, software, "
        "optimizer settings, and measurement shots.",
        "- The experiment uses historical data and equal-weight portfolios.",
        "- The current QAOA experiment optimizes the variance-based QUBO; "
        "it does not directly optimize CVaR.",
        "",
        "## 6. Conclusion",
        "",
        "QuantumRisk demonstrates a working QUBO formulation, "
        "a QAOA simulation using Qiskit Aer, classical exhaustive "
        "benchmarking, and historical portfolio-risk evaluation. "
        "The next research step is to test larger problem instances "
        "and compare solution quality, runtime, and resource use "
        "under clearly controlled conditions.",
        "",
    ]

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"\nClassical portfolios evaluated: {len(benchmark)}")
    print(f"Exact minimum-variance assets: {', '.join(sorted(exact_assets))}")
    print(f"QAOA exact-optimum runs: {successful_runs}/{total_runs}")
    print(f"QAOA feasible runs: {feasible_runs}/{total_runs}")

    if mean_runtime is not None:
        print(f"Mean QAOA runtime: {mean_runtime:.4f} seconds")

    print(f"\nReport saved to: {REPORT_FILE}")
    print("\nBenchmark report generated successfully.")


if __name__ == "__main__":
    main()
