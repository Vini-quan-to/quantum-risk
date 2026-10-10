from itertools import combinations
from pathlib import Path
from time import perf_counter
import math

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# QUANTUMRISK DASHBOARD
# Classical exact benchmark + previous QAOA simulation reference
# ------------------------------------------------------------

ROOT = Path(".")
DATA_FILE = ROOT / "daily_returns.csv"
if not DATA_FILE.exists():
    DATA_FILE = ROOT / "data" / "daily_returns.csv"

OUTPUT_FILE = ROOT / "classical_k4_results.csv"
SUMMARY_FILE = ROOT / "classical_k4_summary.csv"
FIGURES_DIR = ROOT / "figures"

K = 4
CONFIDENCE_LEVEL = 0.95
TRADING_DAYS = 252
BATCH_SIZE = 2000

# Previous local QAOA-style simulation results.
# These are reference values, not a new QAOA run by this script.
QAOA_REFERENCE = {
    "assets": "EFA, TLT, USO, XLP",
    "selected_bitstring_probability": 0.00273088,
    "daily_variance": 4.44507648303e-05,
    "annualized_volatility": 0.10583758,
    "rank": 1039,
    "total_portfolios": 4845,
    "variance_gap_percent": 229.93,
    "source_note": "Previous local QAOA-style statevector simulation; not quantum hardware.",
}


def load_returns():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "daily_returns.csv was not found in the project root or data/."
        )

    data = pd.read_csv(DATA_FILE)
    date_columns = [
        c for c in data.columns
        if str(c).strip().lower() in ("date", "datetime", "timestamp")
    ]
    data = data.drop(columns=date_columns)
    data = data.select_dtypes(include=[np.number])
    data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    if data.empty or data.shape[1] < K:
        raise ValueError("Dataset needs valid returns for at least 4 assets.")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Dataset contains non-finite returns.")
    return data


def evaluate_batch(returns, index_batch, assets):
    selected = returns[:, index_batch]
    portfolio_returns = selected.mean(axis=2).T

    variances = portfolio_returns.var(axis=1, ddof=1)
    means = portfolio_returns.mean(axis=1)
    volatilities = portfolio_returns.std(axis=1, ddof=1)
    losses = -portfolio_returns
    var95 = np.quantile(losses, CONFIDENCE_LEVEL, axis=1)
    cvar95 = np.array([
        row[row >= threshold].mean()
        for row, threshold in zip(losses, var95)
    ])

    wealth = np.cumprod(1.0 + portfolio_returns, axis=1)
    peaks = np.maximum.accumulate(wealth, axis=1)
    drawdowns = 1.0 - wealth / peaks
    max_drawdowns = drawdowns.max(axis=1)

    records = []
    for i, indices in enumerate(index_batch):
        records.append({
            "assets": ", ".join(assets[j] for j in indices),
            "number_of_assets": K,
            "mean_daily_return": float(means[i]),
            "annualized_return_estimate": float(means[i] * TRADING_DAYS),
            "daily_variance": float(variances[i]),
            "annualized_volatility": float(
                volatilities[i] * math.sqrt(TRADING_DAYS)
            ),
            "VaR_95": float(var95[i]),
            "CVaR_95": float(cvar95[i]),
            "max_drawdown": float(max_drawdowns[i]),
        })
    return records


def run_classical_benchmark():
    start = perf_counter()
    print("=" * 72)
    print("QUANTUMRISK | CLASSICAL EXACT SEARCH + QAOA REFERENCE DASHBOARD")
    print("=" * 72)

    df = load_returns()
    assets = list(df.columns)
    returns = df.to_numpy(dtype=float)
    total = math.comb(len(assets), K)

    print(f"Dataset: {DATA_FILE}")
    print(f"Assets: {len(assets)}")
    print(f"Observations: {len(df)}")
    print(f"Candidate portfolios: {total:,}")
    print("\nEvaluating every four-asset portfolio...")

    best_variance = None
    best_cvar = None
    processed = 0
    iterator = combinations(range(len(assets)), K)
    all_records = []

    while True:
        batch = []
        for _ in range(BATCH_SIZE):
            try:
                batch.append(next(iterator))
            except StopIteration:
                break
        if not batch:
            break

        records = evaluate_batch(returns, np.asarray(batch, dtype=int), assets)
        all_records.extend(records)

        for record in records:
            if (
                best_variance is None
                or record["daily_variance"] < best_variance["daily_variance"]
            ):
                best_variance = record
            if best_cvar is None or record["CVaR_95"] < best_cvar["CVaR_95"]:
                best_cvar = record

        processed += len(batch)
        print(f"Progress: {processed:,} / {total:,}")

    elapsed = perf_counter() - start
    results_df = pd.DataFrame(all_records).sort_values("daily_variance")
    results_df.to_csv(OUTPUT_FILE, index=False)

    summary = pd.DataFrame([
        {
            "objective": "minimum_variance",
            **best_variance,
            "portfolios_evaluated": processed,
            "runtime_seconds": elapsed,
        },
        {
            "objective": "minimum_CVaR_95",
            **best_cvar,
            "portfolios_evaluated": processed,
            "runtime_seconds": elapsed,
        },
    ])
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n" + "=" * 72)
    print("CLASSICAL EXACT RESULT")
    print("=" * 72)
    print(f"Runtime: {elapsed:.3f} seconds")
    for key, value in best_variance.items():
        print(f"{key}: {value}")

    return best_variance, best_cvar, elapsed, processed


def print_comparison(classical_best, runtime, processed):
    classical_variance = classical_best["daily_variance"]
    classical_volatility = classical_best["annualized_volatility"]
    qaoa_variance = QAOA_REFERENCE["daily_variance"]
    qaoa_volatility = QAOA_REFERENCE["annualized_volatility"]
    variance_gap = (qaoa_variance / classical_variance - 1.0) * 100.0

    print("\n" + "=" * 72)
    print("CLASSICAL vs PREVIOUS QAOA-STYLE SIMULATION")
    print("=" * 72)
    print(f"{'Metric':<34}{'Classical exact':>18}{'QAOA reference':>20}")
    print("-" * 72)
    print(f"{'Selected assets':<34}{classical_best['assets']:>18}{QAOA_REFERENCE['assets']:>20}")
    print(f"{'Daily variance':<34}{classical_variance:>18.8e}{qaoa_variance:>20.8e}")
    print(f"{'Annualized volatility':<34}{classical_volatility:>17.2%}{qaoa_volatility:>19.2%}")
    print(f"{'Portfolios searched':<34}{processed:>18,}{'4,845 possible':>20}")
    print(f"{'Classical runtime':<34}{runtime:>17.3f} s{'—':>20}")
    print(f"\nQAOA selected bitstring probability: "
          f"{QAOA_REFERENCE['selected_bitstring_probability']:.8f}")
    print(f"QAOA reference rank: {QAOA_REFERENCE['rank']:,} / "
          f"{QAOA_REFERENCE['total_portfolios']:,}")
    print(f"QAOA variance gap vs exact optimum: {variance_gap:.2f}%")
    print("\nInterpretation:")
    print("- The classical method exhaustively evaluates all feasible 4-asset portfolios.")
    print("- The QAOA values are from a previous local statevector simulation.")
    print("- This script does not run QAOA or use quantum hardware.")
    print("- These results do not demonstrate quantum advantage.")


def create_charts(classical_best):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib is not installed, so charts were skipped.")
        print("Install it with: python -m pip install matplotlib")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    classical_var = classical_best["daily_variance"]
    qaoa_var = QAOA_REFERENCE["daily_variance"]
    classical_vol = classical_best["annualized_volatility"]
    qaoa_vol = QAOA_REFERENCE["annualized_volatility"]

    # Chart 1: daily variance
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        ["Classical exact\n(optimum)", "Previous QAOA-style\nsimulation"],
        [classical_var, qaoa_var],
    )
    ax.set_title("QuantumRisk: Daily Portfolio Variance")
    ax.set_ylabel("Daily variance (lower is better)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, [classical_var, qaoa_var]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3e}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    variance_path = FIGURES_DIR / "quantumrisk_variance_comparison.png"
    fig.savefig(variance_path, dpi=180)
    plt.close(fig)

    # Chart 2: annualized volatility
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        ["Classical exact\n(optimum)", "Previous QAOA-style\nsimulation"],
        [classical_vol * 100, qaoa_vol * 100],
    )
    ax.set_title("QuantumRisk: Annualized Volatility")
    ax.set_ylabel("Annualized volatility (%)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, [classical_vol * 100, qaoa_vol * 100]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    volatility_path = FIGURES_DIR / "quantumrisk_volatility_comparison.png"
    fig.savefig(volatility_path, dpi=180)
    plt.close(fig)

    print("\nCharts saved:")
    print(f"- {variance_path}")
    print(f"- {volatility_path}")


def main():
    classical_best, best_cvar, runtime, processed = run_classical_benchmark()
    print_comparison(classical_best, runtime, processed)
    create_charts(classical_best)

    print("\nSaved files:")
    print(f"- {OUTPUT_FILE}")
    print(f"- {SUMMARY_FILE}")
    print("\nQuantumRisk dashboard report completed.")


if __name__ == "__main__":
    main()
