
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# QuantumRisk: Exact classical benchmark
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "daily_returns.csv"
QAOA_FILE = PROJECT_ROOT / "data" / "quantum_portfolio_results.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "benchmark_results.csv"

ASSETS_TO_SELECT = 4
CONFIDENCE_LEVEL = 0.95
TRADING_DAYS = 252


def load_returns():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing file: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)

    date_columns = [
        col for col in data.columns
        if str(col).strip().lower()
        in ("date", "datetime", "timestamp")
    ]

    data = data.drop(columns=date_columns)
    data = data.select_dtypes(include=[np.number])
    data = data.dropna(axis=0, how="any")

    if data.shape[1] < ASSETS_TO_SELECT:
        raise ValueError("Not enough assets in the returns file.")

    return data


def evaluate_portfolio(returns, selected_assets):
    # Equal-weight portfolio.
    portfolio_returns = returns[selected_assets].mean(axis=1)

    losses = -portfolio_returns.to_numpy()
    var_95 = np.quantile(losses, CONFIDENCE_LEVEL)

    # Use the same empirical tail definition as our QAOA evaluation.
    tail_losses = losses[losses >= var_95]
    cvar_95 = tail_losses.mean()

    daily_mean = portfolio_returns.mean()
    daily_volatility = portfolio_returns.std(ddof=1)

    wealth = (1 + portfolio_returns).cumprod()
    running_peak = wealth.cummax()
    drawdowns = 1 - wealth / running_peak

    return {
        "assets": ", ".join(selected_assets),
        "mean_daily_return": daily_mean,
        "annualized_return_estimate": daily_mean * TRADING_DAYS,
        "annualized_volatility": daily_volatility * np.sqrt(TRADING_DAYS),
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "max_drawdown": drawdowns.max(),
    }


def main():
    print("=" * 68)
    print("QUANTUMRISK: EXACT CLASSICAL BENCHMARK")
    print("=" * 68)

    returns = load_returns()
    assets = list(returns.columns)

    covariance = returns.cov().to_numpy()
    asset_indices = {asset: i for i, asset in enumerate(assets)}

    results = []

    # Enumerate all possible portfolios with exactly four assets.
    for selected in combinations(assets, ASSETS_TO_SELECT):
        indices = [asset_indices[asset] for asset in selected]

        # Equal weights: each selected asset has weight 1/4.
        weights = np.full(ASSETS_TO_SELECT, 1 / ASSETS_TO_SELECT)

        sub_covariance = covariance[np.ix_(indices, indices)]

        # This is the same variance objective used by our QUBO:
        # x.T @ covariance @ x / 4**2.
        variance = float(weights @ sub_covariance @ weights)

        metrics = evaluate_portfolio(returns, list(selected))
        metrics["daily_variance"] = variance
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Exact minimum variance among all 70 feasible portfolios.
    min_variance = results_df.loc[
        results_df["daily_variance"].idxmin()
    ]

    # Best portfolio according to empirical historical CVaR.
    min_cvar = results_df.loc[
        results_df["CVaR_95"].idxmin()
    ]

    # Load the portfolio selected by the QAOA run.
    qaoa_metrics = None

    if QAOA_FILE.exists():
        qaoa_data = pd.read_csv(QAOA_FILE)

        if "asset" in qaoa_data.columns:
            qaoa_assets = qaoa_data["asset"].astype(str).tolist()

            if (
                len(qaoa_assets) == ASSETS_TO_SELECT
                and all(asset in assets for asset in qaoa_assets)
            ):
                qaoa_metrics = evaluate_portfolio(
                    returns, qaoa_assets
                )
                qaoa_metrics["daily_variance"] = float(
                    np.mean(
                        returns[qaoa_assets].mean(axis=1)
                        .to_numpy() ** 2
                    )
                )
                # Replace the previous line with the true centered
                # variance for consistency with the covariance objective.
                qaoa_metrics["daily_variance"] = float(
                    returns[qaoa_assets].mean(axis=1).var(ddof=1)
                )

    # Save every enumerated portfolio, ordered by variance.
    results_df = results_df.sort_values(
        "daily_variance"
    ).reset_index(drop=True)

    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nNumber of assets: {len(assets)}")
    print(f"Portfolios examined: {len(results_df)}")

    print("\n--- EXACT MINIMUM-VARIANCE PORTFOLIO ---")
    for key, value in min_variance.items():
        print(f"{key}: {value}")

    print("\n--- BEST HISTORICAL CVaR PORTFOLIO ---")
    for key, value in min_cvar.items():
        print(f"{key}: {value}")

    if qaoa_metrics is not None:
        print("\n--- QAOA PORTFOLIO ---")
        for key, value in qaoa_metrics.items():
            print(f"{key}: {value}")

        best_variance = float(min_variance["daily_variance"])
        qaoa_variance = float(qaoa_metrics["daily_variance"])

        if best_variance > 0:
            gap = 100 * (qaoa_variance - best_variance) / best_variance
            print(f"\nQAOA variance gap vs exact optimum: {gap:.4f}%")

    print(f"\nAll portfolio results saved to: {OUTPUT_FILE}")
    print("\nClassical benchmark completed successfully.")


if __name__ == "__main__":
    main()
