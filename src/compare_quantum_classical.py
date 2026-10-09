from pathlib import Path

import pandas as pd

from risk_engine import load_returns, evaluate_portfolio


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QAOA_PATH = PROJECT_ROOT / "data" / "qaoa_aer_results.csv"
CLASSICAL_PATH = PROJECT_ROOT / "data" / "portfolio_results.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "risk_comparison.csv"


# --------------------------------------------------
# Portfolio evaluation
# --------------------------------------------------

def evaluate_named_portfolio(returns, assets, method):
    """Evaluate a portfolio using the shared risk engine."""

    assets = sorted(assets)

    if len(assets) != 4:
        raise ValueError(
            f"{method} portfolio must contain exactly 4 assets."
        )

    missing = set(assets) - set(returns.columns)

    if missing:
        raise ValueError(
            f"Unknown assets in {method} portfolio: {missing}"
        )

    metrics = evaluate_portfolio(returns, assets)
    metrics["method"] = method

    return metrics


# --------------------------------------------------
# Main comparison
# --------------------------------------------------

def main():
    print("=" * 65)
    print("QUANTUMRISK: QUANTUM VS CLASSICAL RISK COMPARISON")
    print("=" * 65)

    # Load the same historical returns for both portfolios.
    returns = load_returns()

    # Load QAOA experiment results and the classical CVaR ranking.
    qaoa_results = pd.read_csv(QAOA_PATH)
    classical_results = pd.read_csv(CLASSICAL_PATH)

    if qaoa_results.empty or classical_results.empty:
        raise ValueError("An input results file is empty.")

    # The classical risk engine sorts portfolios by CVaR ascending.
    best_classical_assets = (
        classical_results.iloc[0]["assets"].split(", ")
    )

    comparison = []

    # Evaluate each distinct portfolio produced by QAOA.
    qaoa_portfolios = qaoa_results[
        qaoa_results["feasible"].astype(str).str.lower().eq("true")
    ]

    unique_qaoa_assets = sorted(
        set(qaoa_portfolios["assets"].dropna())
    )

    for assets_text in unique_qaoa_assets:
        assets = assets_text.split(", ")

        metrics = evaluate_named_portfolio(
            returns,
            assets,
            "QAOA minimum-variance portfolio",
        )

        metrics["qaoa_runs"] = int(
            (qaoa_portfolios["assets"] == assets_text).sum()
        )

        comparison.append(metrics)

    # Evaluate the best portfolio selected by classical CVaR.
    classical_metrics = evaluate_named_portfolio(
        returns,
        best_classical_assets,
        "Classical minimum-CVaR portfolio",
    )

    classical_metrics["qaoa_runs"] = 0
    comparison.append(classical_metrics)

    results = pd.DataFrame(comparison)

    # Put the most useful comparison columns first.
    columns = [
        "method",
        "assets",
        "qaoa_runs",
        "mean_daily_return",
        "annualized_return_estimate",
        "annualized_volatility",
        "VaR_95",
        "CVaR_95",
        "max_drawdown",
    ]

    results = results[columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    # Display risk metrics as percentages for easier interpretation.
    display = results.copy()

    percentage_columns = [
        "mean_daily_return",
        "annualized_return_estimate",
        "annualized_volatility",
        "VaR_95",
        "CVaR_95",
        "max_drawdown",
    ]

    for column in percentage_columns:
        display[column] = (
            display[column] * 100
        ).round(4).astype(str) + "%"

    print(f"\nHistorical observations: {len(returns)}")
    print(f"Distinct feasible QAOA portfolios: {len(unique_qaoa_assets)}")

    print("\nRISK COMPARISON")
    print("-" * 65)
    print(display.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("\nComparison completed successfully.")


if __name__ == "__main__":
    main()