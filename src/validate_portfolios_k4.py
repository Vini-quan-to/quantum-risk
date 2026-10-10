from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


# ============================================================
# QuantumRisk: Fixed-size classical benchmark (k = 4)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "daily_returns.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "classical_k4_results.csv"
SUMMARY_FILE = PROJECT_ROOT / "data" / "classical_k4_summary.csv"

PORTFOLIO_SIZE = 4
CONFIDENCE_LEVEL = 0.95
TRADING_DAYS = 252
BATCH_SIZE = 2000


# ============================================================
# Load data
# ============================================================

def load_returns():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {DATA_FILE}\n"
            "Run python src/download_data.py first."
        )

    data = pd.read_csv(DATA_FILE)

    date_columns = [
        column
        for column in data.columns
        if str(column).strip().lower()
        in ("date", "datetime", "timestamp")
    ]
    data = data.drop(columns=date_columns)
    data = data.select_dtypes(include=[np.number])
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=0, how="any")

    if data.empty:
        raise ValueError("The returns file contains no valid observations.")

    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Returns contain non-finite values.")

    if data.shape[1] < PORTFOLIO_SIZE:
        raise ValueError(
            f"Need at least {PORTFOLIO_SIZE} assets, "
            f"but found {data.shape[1]}."
        )

    return data


# ============================================================
# Evaluate a batch of equal-weighted portfolios
# ============================================================

def evaluate_batch(returns_array, index_batch, assets):
    """
    Evaluate equal-weighted portfolios, each containing exactly k assets.

    index_batch shape:
        (number of portfolios in batch, PORTFOLIO_SIZE)
    """
    batch_count, portfolio_size = index_batch.shape

    # Shape: (observations, batch_count, portfolio_size)
    selected_returns = returns_array[:, index_batch]

    # Equal-weighted portfolio returns, shape: (batch_count, observations)
    portfolio_returns = selected_returns.mean(axis=2).T

    daily_variance = portfolio_returns.var(axis=1, ddof=1)
    daily_mean = portfolio_returns.mean(axis=1)
    daily_volatility = portfolio_returns.std(axis=1, ddof=1)

    losses = -portfolio_returns
    var_95 = np.quantile(losses, CONFIDENCE_LEVEL, axis=1)

    # Empirical CVaR: mean of losses at or beyond the VaR threshold.
    cvar_95 = np.array([
        row[row >= threshold].mean()
        for row, threshold in zip(losses, var_95)
    ])

    wealth = np.cumprod(1.0 + portfolio_returns, axis=1)
    running_peak = np.maximum.accumulate(wealth, axis=1)
    drawdowns = 1.0 - wealth / running_peak
    max_drawdown = drawdowns.max(axis=1)

    records = []
    for row_index in range(batch_count):
        selected_assets = [
            assets[index] for index in index_batch[row_index]
        ]

        records.append({
            "assets": ", ".join(selected_assets),
            "number_of_assets": portfolio_size,
            "mean_daily_return": float(daily_mean[row_index]),
            "annualized_return_estimate": float(
                daily_mean[row_index] * TRADING_DAYS
            ),
            "daily_variance": float(daily_variance[row_index]),
            "annualized_volatility": float(
                daily_volatility[row_index] * np.sqrt(TRADING_DAYS)
            ),
            "VaR_95": float(var_95[row_index]),
            "CVaR_95": float(cvar_95[row_index]),
            "max_drawdown": float(max_drawdown[row_index]),
        })

    return records


# ============================================================
# Exact classical search for portfolios of exactly k = 4
# ============================================================

def main():
    print("=" * 72)
    print("QUANTUMRISK: CLASSICAL BENCHMARK (EXACTLY 4 ASSETS)")
    print("=" * 72)

    start_time = perf_counter()

    returns_df = load_returns()
    assets = list(returns_df.columns)
    returns_array = returns_df.to_numpy(dtype=float)
    asset_count = len(assets)

    total_portfolios = sum(
        1 for _ in combinations(range(asset_count), PORTFOLIO_SIZE)
    )

    print(f"\nAssets loaded: {asset_count}")
    print(f"Return observations: {len(returns_df)}")
    print(f"Portfolio size: exactly {PORTFOLIO_SIZE}")
    print(f"Candidate portfolios: {total_portfolios:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print("\nStarting exhaustive evaluation...")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    for output_path in (OUTPUT_FILE, SUMMARY_FILE):
        if output_path.exists():
            output_path.unlink()

    portfolios_processed = 0
    best_variance_portfolio = None
    best_cvar_portfolio = None
    best_variance = float("inf")
    best_cvar = float("inf")

    iterator = combinations(range(asset_count), PORTFOLIO_SIZE)

    while True:
        batch = []
        for _ in range(BATCH_SIZE):
            try:
                batch.append(next(iterator))
            except StopIteration:
                break

        if not batch:
            break

        index_batch = np.asarray(batch, dtype=int)
        batch_df = pd.DataFrame(
            evaluate_batch(returns_array, index_batch, assets)
        )

        batch_df.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=(portfolios_processed == 0),
            index=False,
        )

        variance_row = batch_df.loc[batch_df["daily_variance"].idxmin()]
        if float(variance_row["daily_variance"]) < best_variance:
            best_variance = float(variance_row["daily_variance"])
            best_variance_portfolio = variance_row.to_dict()

        cvar_row = batch_df.loc[batch_df["CVaR_95"].idxmin()]
        if float(cvar_row["CVaR_95"]) < best_cvar:
            best_cvar = float(cvar_row["CVaR_95"])
            best_cvar_portfolio = cvar_row.to_dict()

        portfolios_processed += len(batch)
        if (
            portfolios_processed % 20000 < len(batch)
            or portfolios_processed == total_portfolios
        ):
            print(
                f"Progress: {portfolios_processed:,} / "
                f"{total_portfolios:,}"
            )

    elapsed = perf_counter() - start_time

    print("\n" + "=" * 72)
    print("EXACT CLASSICAL RESULTS — FIXED k = 4")
    print("=" * 72)
    print(f"Assets available: {asset_count}")
    print(f"Portfolio size: exactly {PORTFOLIO_SIZE}")
    print(f"Portfolios evaluated: {portfolios_processed:,}")
    print(f"Runtime: {elapsed:.4f} seconds")

    print("\n--- MINIMUM-VARIANCE PORTFOLIO ---")
    for key, value in best_variance_portfolio.items():
        print(f"{key}: {value}")

    print("\n--- MINIMUM-HISTORICAL-CVaR PORTFOLIO ---")
    for key, value in best_cvar_portfolio.items():
        print(f"{key}: {value}")

    summary_rows = []
    for objective, portfolio in (
        ("minimum_variance", best_variance_portfolio),
        ("minimum_CVaR_95", best_cvar_portfolio),
    ):
        summary_rows.append({
            "objective": objective,
            **portfolio,
            "portfolios_evaluated": portfolios_processed,
            "runtime_seconds": elapsed,
        })

    pd.DataFrame(summary_rows).to_csv(SUMMARY_FILE, index=False)

    print(f"\nAll portfolios saved to: {OUTPUT_FILE}")
    print(f"Best-portfolio summary saved to: {SUMMARY_FILE}")
    print("\nClassical k=4 benchmark completed successfully.")
    print(
        "\nFair-comparison note: compare the minimum-variance result here "
        "with the quantum optimizer using the same four assets per "
        "portfolio, equal weights, return data, and variance definition."
    )


if __name__ == "__main__":
    main()
