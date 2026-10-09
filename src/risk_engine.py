
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "daily_returns.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "portfolio_results.csv"

PORTFOLIO_SIZE = 4
CONFIDENCE_LEVEL = 0.95
TRADING_DAYS = 252


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_returns():
    """Load and validate daily asset returns."""

    returns = pd.read_csv(
        DATA_PATH,
        index_col="Date",
        parse_dates=True,
    )

    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns = returns.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(how="any")

    if returns.shape[1] < PORTFOLIO_SIZE:
        raise ValueError("Not enough assets for portfolio.")

    if len(returns) < 30:
        raise ValueError("Too few return observations.")

    return returns


# --------------------------------------------------
# Risk metrics
# --------------------------------------------------

def calculate_var_cvar(
    portfolio_returns,
    confidence=CONFIDENCE_LEVEL,
):
    """
    Calculate historical VaR and CVaR as positive
    loss measures, expressed in decimal returns.
    """

    losses = -np.asarray(portfolio_returns, dtype=float)

    var = np.quantile(
        losses,
        confidence,
        method="linear",
    )

    # Empirical tail mean, including all observations
    # whose losses are at least as large as the VaR.
    tail_losses = losses[losses >= var]

    cvar = (
        tail_losses.mean()
        if len(tail_losses) > 0
        else var
    )

    return float(var), float(cvar)


def calculate_metrics(portfolio_returns):
    """Calculate the main performance and risk metrics."""

    returns = np.asarray(portfolio_returns, dtype=float)

    var, cvar = calculate_var_cvar(returns)

    equity = np.concatenate([
        [1.0],
        np.cumprod(1.0 + returns),
    ])

    running_peak = np.maximum.accumulate(equity)

    drawdowns = equity / running_peak - 1.0
    max_drawdown = abs(float(drawdowns.min()))

    return {
        "mean_daily_return": float(returns.mean()),
        "annualized_return_estimate": float(
            returns.mean() * TRADING_DAYS
        ),
        "annualized_volatility": float(
            returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
        ),
        "VaR_95": var,
        "CVaR_95": cvar,
        "max_drawdown": max_drawdown,
    }


# --------------------------------------------------
# Portfolio evaluation
# --------------------------------------------------

def evaluate_portfolio(returns, selected_assets):
    """Evaluate an equal-weight portfolio."""

    portfolio_returns = returns[
        list(selected_assets)
    ].mean(axis=1)

    metrics = calculate_metrics(portfolio_returns)
    metrics["assets"] = ", ".join(selected_assets)

    return metrics


def exhaustive_search(returns):
    """
    Evaluate every feasible equal-weight portfolio
    containing exactly PORTFOLIO_SIZE assets.
    """

    assets = list(returns.columns)
    results = []

    for selected_assets in combinations(
        assets, PORTFOLIO_SIZE
    ):
        result = evaluate_portfolio(
            returns,
            selected_assets,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "CVaR_95",
        ascending=True,
    ).reset_index(drop=True)

    return results_df


# --------------------------------------------------
# Main program
# --------------------------------------------------

def main():
    print("=" * 65)
    print("QUANTUMRISK - CLASSICAL RISK ENGINE")
    print("=" * 65)

    returns = load_returns()

    print(f"\nObservations: {len(returns)}")
    print(f"Assets: {len(returns.columns)}")
    print(f"Portfolio size: {PORTFOLIO_SIZE}")
    print(f"Confidence level: {CONFIDENCE_LEVEL:.0%}")

    print("\nAnnualized asset volatility:")
    volatility = (
        returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    ).sort_values(ascending=False)

    print((volatility * 100).round(2).astype(str).add("%"))

    print("\nCalculating correlation matrix...")
    print(returns.corr().round(2))

    print("\nSearching all feasible portfolios...")
    results = exhaustive_search(returns)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(OUTPUT_PATH, index=False)

    best = results.iloc[0]

    print(f"\nPortfolios evaluated: {len(results)}")
    print("\nLOWEST HISTORICAL CVaR PORTFOLIO")
    print("-" * 45)
    print(f"Assets: {best['assets']}")
    print(
        f"Daily mean return: "
        f"{best['mean_daily_return'] * 100:.4f}%"
    )
    print(
        f"Annualized volatility: "
        f"{best['annualized_volatility'] * 100:.2f}%"
    )
    print(f"Historical VaR 95%: {best['VaR_95'] * 100:.4f}%")
    print(f"Historical CVaR 95%: {best['CVaR_95'] * 100:.4f}%")
    print(
        f"Maximum drawdown: "
        f"{best['max_drawdown'] * 100:.2f}%"
    )

    print("\nTOP 5 PORTFOLIOS BY HISTORICAL CVaR")
    print(
        results[
            [
                "assets",
                "VaR_95",
                "CVaR_95",
                "annualized_volatility",
                "max_drawdown",
            ]
        ].head().to_string(index=False)
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("\nRisk engine completed successfully.")


if __name__ == "__main__":
    main()
