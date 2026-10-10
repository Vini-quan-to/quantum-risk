
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RETURNS_FILE = DATA_DIR / "daily_returns.csv"
OUTPUT_FILE = DATA_DIR / "classical_baseline_k4.csv"

PORTFOLIO_SIZE = 4


def load_returns():
    if not RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {RETURNS_FILE}. "
            "Run src/download_data.py first."
        )

    returns = pd.read_csv(RETURNS_FILE)

    date_columns = [
        col for col in returns.columns
        if str(col).strip().lower()
        in {"date", "datetime", "timestamp"}
    ]
    returns = returns.drop(columns=date_columns)
    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if returns.shape[1] < PORTFOLIO_SIZE:
        raise ValueError(
            f"Need at least {PORTFOLIO_SIZE} assets; "
            f"found {returns.shape[1]}."
        )

    if len(returns) < 2:
        raise ValueError("Need at least two valid observations.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns contain non-finite values.")

    return returns


def evaluate_portfolio(covariance, indices):
    # For an equal-weight portfolio, variance is
    # sum of the selected covariance submatrix / k^2.
    submatrix = covariance[np.ix_(indices, indices)]
    k = len(indices)

    return float(submatrix.sum() / (k * k))


def main():
    returns = load_returns()
    assets = list(returns.columns)
    covariance = returns.cov().to_numpy(dtype=float)
    covariance = (covariance + covariance.T) / 2.0

    if not np.isfinite(covariance).all():
        raise ValueError("Covariance matrix contains non-finite values.")

    print("=" * 60)
    print("QUANTUMRISK: CLASSICAL BASELINE")
    print("=" * 60)
    print(f"Assets: {len(assets)}")
    print(f"Historical observations: {len(returns)}")
    print(f"Portfolio size: {PORTFOLIO_SIZE}")

    total = 0
    best_variance = float("inf")
    best_indices = None

    for indices in combinations(range(len(assets)), PORTFOLIO_SIZE):
        variance = evaluate_portfolio(covariance, indices)
        total += 1

        if variance < best_variance:
            best_variance = variance
            best_indices = indices

    selected_assets = [assets[i] for i in best_indices]

    selected_returns = returns[selected_assets].mean(axis=1)
    daily_volatility = float(selected_returns.std(ddof=1))
    annualized_volatility = daily_volatility * np.sqrt(252)

    result = {
        "portfolio_size": PORTFOLIO_SIZE,
        "selected_assets": ", ".join(selected_assets),
        "equal_weight_variance": best_variance,
        "daily_volatility": daily_volatility,
        "annualized_volatility": annualized_volatility,
        "portfolios_evaluated": total,
        "method": "Exhaustive classical search",
    }

    pd.DataFrame([result]).to_csv(OUTPUT_FILE, index=False)

    print(f"\nPortfolios evaluated: {total}")
    print(f"Best assets: {', '.join(selected_assets)}")
    print(f"Equal-weight daily variance: {best_variance:.12g}")
    print(f"Daily volatility: {daily_volatility:.8%}")
    print(f"Annualized volatility: {annualized_volatility:.8%}")
    print(f"\nSaved baseline to: {OUTPUT_FILE}")
    print("\nThis is an exact optimum for k=4 under the sample covariance objective.")


if __name__ == "__main__":
    main()
