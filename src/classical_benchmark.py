
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


# ============================================================
# QuantumRisk: Variable-size classical benchmark
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "daily_returns.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "benchmark_results.csv"

MIN_ASSETS = 4
MAX_ASSETS = 10

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

    if data.shape[1] < MAX_ASSETS:
        raise ValueError(
            f"Need at least {MAX_ASSETS} assets, "
            f"but found {data.shape[1]}."
        )

    if not 1 <= MIN_ASSETS <= MAX_ASSETS <= data.shape[1]:
        raise ValueError("Invalid minimum/maximum portfolio sizes.")

    return data


# ============================================================
# Count candidate portfolios
# ============================================================

def count_portfolios(asset_count):
    return sum(
        sum(1 for _ in combinations(range(asset_count), size))
        for size in range(MIN_ASSETS, MAX_ASSETS + 1)
    )


# ============================================================
# Evaluate a batch of portfolios
# ============================================================

def evaluate_batch(returns_array, index_batch, assets):
    """
    Evaluate equal-weighted portfolios of one fixed size.

    index_batch shape:
        (number of portfolios in batch, number of assets selected)

    portfolio_returns shape:
        (number of portfolios in batch, number of observations)
    """

    batch_size, portfolio_size = index_batch.shape

    # Shape before averaging:
    # (observations, batch_size, portfolio_size)
    selected_returns = returns_array[:, index_batch]

    # Average over the selected-assets axis, not the batch axis.
    # Result: (observations, batch_size)
    portfolio_returns = selected_returns.mean(axis=2).T

    # Daily sample variance and return statistics.
    daily_variance = portfolio_returns.var(axis=1, ddof=1)
    daily_mean = portfolio_returns.mean(axis=1)
    daily_volatility = portfolio_returns.std(axis=1, ddof=1)

    # Historical losses and 95% VaR.
    losses = -portfolio_returns

    var_95 = np.quantile(
        losses,
        CONFIDENCE_LEVEL,
        axis=1,
    )

    # Empirical tail-loss definition, consistently applied.
    cvar_95 = np.array([
        row[row >= threshold].mean()
        for row, threshold in zip(losses, var_95)
    ])

    # Maximum drawdown.
    wealth = np.cumprod(1.0 + portfolio_returns, axis=1)
    running_peak = np.maximum.accumulate(wealth, axis=1)
    drawdowns = 1.0 - wealth / running_peak
    max_drawdown = drawdowns.max(axis=1)

    records = []

    for row_index in range(batch_size):
        selected_assets = [
            assets[index]
            for index in index_batch[row_index]
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
# Classical exhaustive search
# ============================================================

def main():
    print("=" * 72)
    print("QUANTUMRISK: VARIABLE-SIZE CLASSICAL BENCHMARK")
    print("=" * 72)

    start_time = perf_counter()

    returns_df = load_returns()
    assets = list(returns_df.columns)
    returns_array = returns_df.to_numpy(dtype=float)

    asset_count = len(assets)
    total_portfolios = count_portfolios(asset_count)

    print(f"\nAssets loaded: {asset_count}")
    print(f"Return observations: {len(returns_df)}")
    print(f"Portfolio size range: {MIN_ASSETS}–{MAX_ASSETS}")
    print(f"Total candidate portfolios: {total_portfolios:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print("\nStarting exhaustive evaluation...")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    portfolios_processed = 0
    best_variance = None
    best_cvar = None
    best_variance_portfolio = None
    best_cvar_portfolio = None

    for portfolio_size in range(MIN_ASSETS, MAX_ASSETS + 1):
        print(
            f"\nEvaluating portfolios with {portfolio_size} assets..."
        )

        iterator = combinations(range(asset_count), portfolio_size)

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

            records = evaluate_batch(
                returns_array=returns_array,
                index_batch=index_batch,
                assets=assets,
            )

            batch_df = pd.DataFrame(records)

            # Write incrementally to avoid keeping every portfolio
            # in memory.
            batch_df.to_csv(
                OUTPUT_FILE,
                mode="a",
                header=(portfolios_processed == 0),
                index=False,
            )

            # Update best minimum-variance portfolio.
            variance_row = batch_df.loc[
                batch_df["daily_variance"].idxmin()
            ]

            if (
                best_variance is None
                or variance_row["daily_variance"] < best_variance
            ):
                best_variance = float(variance_row["daily_variance"])
                best_variance_portfolio = variance_row.to_dict()

            # Update best historical-CVaR portfolio.
            cvar_row = batch_df.loc[
                batch_df["CVaR_95"].idxmin()
            ]

            if best_cvar is None or cvar_row["CVaR_95"] < best_cvar:
                best_cvar = float(cvar_row["CVaR_95"])
                best_cvar_portfolio = cvar_row.to_dict()

            portfolios_processed += len(batch)

            if (
                portfolios_processed % 20000 < len(batch)
                or portfolios_processed == total_portfolios
            ):
                print(
                    f"Progress: {portfolios_processed:,}"
                    f" / {total_portfolios:,}"
                )

    elapsed = perf_counter() - start_time

    print("\n" + "=" * 72)
    print("EXACT CLASSICAL BENCHMARK RESULTS")
    print("=" * 72)

    print(f"Assets available: {asset_count}")
    print(f"Portfolio sizes evaluated: {MIN_ASSETS}–{MAX_ASSETS}")
    print(f"Portfolios evaluated: {portfolios_processed:,}")
    print(f"Runtime: {elapsed:.2f} seconds")

    print("\n--- MINIMUM-VARIANCE PORTFOLIO ---")
    for key, value in best_variance_portfolio.items():
        print(f"{key}: {value}")

    print("\n--- MINIMUM-HISTORICAL-CVaR PORTFOLIO ---")
    for key, value in best_cvar_portfolio.items():
        print(f"{key}: {value}")

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("\nClassical benchmark completed successfully.")


if __name__ == "__main__":
    main()
