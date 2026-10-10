"""
QuantumRisk — classical CVaR baseline.

Run from the project root:
    python src/cvar_baseline.py

Input:
    data/daily_returns.csv

Outputs:
    data/classical_cvar_results.csv
    data/classical_cvar_summary.txt
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------- Configuration -------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETURNS_FILE = PROJECT_ROOT / "data" / "daily_returns.csv"
RESULTS_FILE = PROJECT_ROOT / "data" / "classical_cvar_results.csv"
SUMMARY_FILE = PROJECT_ROOT / "data" / "classical_cvar_summary.txt"

PORTFOLIO_SIZE = 4
CONFIDENCE_LEVEL = 0.95


# ------------------------------- Data loading -------------------------------

def load_returns(path: Path) -> tuple[list[str], np.ndarray]:
    """Load a date-indexed table of daily asset returns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Returns file not found: {path}\n"
            "Expected data/daily_returns.csv in your project."
        )

    frame = pd.read_csv(path, index_col=0)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="any")

    if frame.empty or frame.shape[1] < PORTFOLIO_SIZE:
        raise ValueError(
            f"Need at least {PORTFOLIO_SIZE} assets and one complete row of returns."
        )

    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Returns data contains non-finite values.")

    names = [str(column) for column in frame.columns]
    return names, values


# ------------------------------- Risk metrics -------------------------------

def historical_var_cvar(
    losses: np.ndarray,
    confidence: float = CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """
    Calculate historical VaR and CVaR from a sample of portfolio losses.

    VaR is the linear empirical quantile at the confidence level.
    CVaR here is the mean of observations with loss >= VaR. If there are ties
    at the VaR threshold, this can include more than exactly 1-confidence
    fraction of observations; this convention is documented deliberately.
    """
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a non-empty one-dimensional array.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    var = float(np.quantile(losses, confidence, method="linear"))
    tail_losses = losses[losses >= var]

    if tail_losses.size == 0:
        raise ValueError("Could not determine the VaR tail.")

    cvar = float(tail_losses.mean())
    return var, cvar


# ------------------------------- Main search --------------------------------

def main() -> None:
    names, returns = load_returns(RETURNS_FILE)
    n_assets = len(names)
    n_observations = len(returns)

    portfolios = list(combinations(range(n_assets), PORTFOLIO_SIZE))
    rows: list[dict] = []

    for portfolio in portfolios:
        weights = np.zeros(n_assets, dtype=float)
        weights[list(portfolio)] = 1.0 / PORTFOLIO_SIZE

        portfolio_returns = returns @ weights
        losses = -portfolio_returns

        var_loss, cvar_loss = historical_var_cvar(losses)

        rows.append(
            {
                "portfolio": "|".join(names[i] for i in portfolio),
                "assets": ", ".join(names[i] for i in portfolio),
                "confidence_level": CONFIDENCE_LEVEL,
                "observations": n_observations,
                "historical_var_loss": var_loss,
                "historical_cvar_loss": cvar_loss,
                "mean_daily_return": float(portfolio_returns.mean()),
                "daily_variance": float(np.var(portfolio_returns, ddof=1)),
                "worst_daily_loss": float(losses.max()),
            }
        )

    results = pd.DataFrame(rows)

    # Determine the minimum-CVaR and minimum-variance portfolios independently.
    cvar_best_position = int(
        results["historical_cvar_loss"].to_numpy().argmin()
    )
    variance_best_position = int(results["daily_variance"].to_numpy().argmin())

    results["is_cvar_optimum"] = False
    results.loc[cvar_best_position, "is_cvar_optimum"] = True
    results["is_variance_optimum"] = False
    results.loc[variance_best_position, "is_variance_optimum"] = True

    results = results.sort_values(
        ["historical_cvar_loss", "historical_var_loss"],
        ascending=True,
    ).reset_index(drop=True)
    results.insert(0, "rank_by_cvar", np.arange(1, len(results) + 1))

    cvar_best = results.loc[results["is_cvar_optimum"]].iloc[0]
    variance_best = results.loc[results["is_variance_optimum"]].iloc[0]

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_FILE, index=False)

    summary_lines = [
        "QuantumRisk — Classical Historical CVaR Baseline",
        "=" * 52,
        f"Assets: {n_assets}",
        f"Observations: {n_observations}",
        f"Portfolio size: {PORTFOLIO_SIZE} (equal weights)",
        f"Feasible portfolios evaluated: {len(portfolios):,}",
        f"Confidence level: {CONFIDENCE_LEVEL:.0%}",
        "Loss convention: loss = -daily portfolio return; positive values mean losses.",
        "VaR: linear empirical quantile at the confidence level.",
        "CVaR convention: mean of historical losses greater than or equal to VaR.",
        "Ties at VaR may cause the tail to contain more than exactly 5% of observations.",
        "",
        "Minimum-CVaR portfolio:",
        f"  Assets: {cvar_best['assets']}",
        f"  CVaR loss: {cvar_best['historical_cvar_loss']:.10g}",
        f"  VaR loss: {cvar_best['historical_var_loss']:.10g}",
        f"  Mean daily return: {cvar_best['mean_daily_return']:.10g}",
        f"  Daily variance: {cvar_best['daily_variance']:.10g}",
        f"  Worst daily loss: {cvar_best['worst_daily_loss']:.10g}",
        "",
        "Minimum-variance portfolio:",
        f"  Assets: {variance_best['assets']}",
        f"  Daily variance: {variance_best['daily_variance']:.10g}",
        f"  CVaR loss: {variance_best['historical_cvar_loss']:.10g}",
        f"  VaR loss: {variance_best['historical_var_loss']:.10g}",
        "",
        "Interpretation: this is an in-sample historical estimate, not a guarantee of future risk.",
    ]
    summary_text = "\n".join(summary_lines)
    SUMMARY_FILE.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"\nSaved all portfolio results to: {RESULTS_FILE}")
    print(f"Saved summary to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
