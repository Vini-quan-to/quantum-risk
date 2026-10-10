from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# Project defaults based on the existing QuantumRisk experiment.
DATA_PATH = Path("data/daily_returns.csv")
PORTFOLIO_SIZE = 4
CONFIDENCE_LEVEL = 0.95

# Previously reported baseline, used only as a comparison reference.
EXPECTED_ASSETS = {"XLP", "HYG", "LQD", "AGG"}
EXPECTED_CVAR = 0.008108726493
EXPECTED_VAR = 0.005884533208
EXPECTED_MEAN_RETURN = 0.0002762062889
EXPECTED_DAILY_VARIANCE = 1.347280908e-05

# Prefer these known asset columns if present. If not, infer numeric return columns.
KNOWN_ASSETS = [
    "AGG", "EEM", "EFA", "GLD", "HYG", "LQD", "TLT", "USO", "VNQ",
    "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "QQQ",
    "IWM", "SPY",
]


def load_returns(path):
    """Load daily returns and identify the asset columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run this script from the project root."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{path} contains no rows.")

    df.columns = [str(col).strip() for col in df.columns]

    known_present = [asset for asset in KNOWN_ASSETS if asset in df.columns]

    if known_present:
        asset_columns = known_present
    else:
        # Fallback for datasets with different asset names.
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        excluded = {
            "date", "timestamp", "time", "year", "month", "day",
            "unnamed: 0",
        }
        asset_columns = [
            col for col in numeric
            if str(col).strip().lower() not in excluded
        ]

    if len(asset_columns) < PORTFOLIO_SIZE:
        raise ValueError(
            "Could not identify enough numeric asset-return columns. "
            f"Found: {asset_columns}. Check the CSV headers."
        )

    returns_df = df[asset_columns].apply(pd.to_numeric, errors="coerce")

    # Avoid silently changing the observation window.
    if returns_df.isna().any().any():
        bad_columns = returns_df.columns[
            returns_df.isna().any()
        ].tolist()
        raise ValueError(
            "Missing or non-numeric values found in return columns: "
            f"{bad_columns}. Resolve these before comparing results."
        )

    returns = returns_df.to_numpy(dtype=float)

    if not np.isfinite(returns).all():
        raise ValueError("Return data contains infinite or non-finite values.")

    return returns, list(returns_df.columns)


def portfolio_metrics(returns, indices, confidence):
    """Calculate metrics using equal weights and the project's loss convention."""
    portfolio_returns = returns[:, indices].mean(axis=1)
    losses = -portfolio_returns

    var_loss = float(np.quantile(losses, confidence, method="linear"))
    tail_losses = losses[losses >= var_loss]

    if len(tail_losses) == 0:
        raise ValueError("No observations found in the CVaR tail.")

    cvar_loss = float(tail_losses.mean())

    return {
        "cvar": cvar_loss,
        "var": var_loss,
        "mean_return": float(np.mean(portfolio_returns)),
        "daily_variance": float(np.var(portfolio_returns, ddof=1)),
        "worst_daily_loss": float(np.max(losses)),
        "tail_observations": int(len(tail_losses)),
        "tail_fraction": float(len(tail_losses) / len(losses)),
    }


def main():
    returns, assets = load_returns(DATA_PATH)
    n_observations, n_assets = returns.shape

    if not 0 < CONFIDENCE_LEVEL < 1:
        raise ValueError("CONFIDENCE_LEVEL must be between 0 and 1.")

    total_portfolios = 0
    best_cvar = None
    best_variance = None
    ranked_cvar = []

    print("=== Independent QuantumRisk CVaR Baseline Validation ===")
    print(f"Data file: {DATA_PATH}")
    print(f"Observations: {n_observations}")
    print(f"Assets found: {n_assets}")
    print(f"Portfolio size: {PORTFOLIO_SIZE}")
    print(f"Confidence level: {CONFIDENCE_LEVEL:.1%}")
    print("Portfolio weights: equal weight")
    print("Loss convention: loss = -portfolio return")
    print("CVaR convention: mean losses >= empirical linear-quantile VaR")
    print()

    for selected_indices in combinations(range(n_assets), PORTFOLIO_SIZE):
        selected_assets = tuple(assets[i] for i in selected_indices)
        metrics = portfolio_metrics(
            returns, selected_indices, CONFIDENCE_LEVEL
        )
        total_portfolios += 1

        row = {
            "assets": selected_assets,
            **metrics,
        }
        ranked_cvar.append(row)

        if best_cvar is None or metrics["cvar"] < best_cvar["cvar"]:
            best_cvar = row

        if (
            best_variance is None
            or metrics["daily_variance"] < best_variance["daily_variance"]
        ):
            best_variance = row

    ranked_cvar.sort(key=lambda row: row["cvar"])

    def show_result(title, row):
        print(title)
        print(f"  Assets: {', '.join(row['assets'])}")
        print(f"  CVaR loss: {row['cvar']:.12f} ({row['cvar']:.6%})")
        print(f"  VaR loss: {row['var']:.12f} ({row['var']:.6%})")
        print(f"  Mean daily return: {row['mean_return']:.12f}")
        print(f"  Daily variance: {row['daily_variance']:.12e}")
        print(f"  Worst daily loss: {row['worst_daily_loss']:.12f}")
        print(
            f"  Tail observations: {row['tail_observations']} / "
            f"{n_observations} ({row['tail_fraction']:.2%})"
        )
        print()

    print(f"Portfolios evaluated: {total_portfolios}")
    show_result("=== Minimum-CVaR Portfolio ===", best_cvar)
    show_result("=== Minimum-Variance Portfolio ===", best_variance)

    print("=== Five Lowest-CVaR Portfolios ===")
    for rank, row in enumerate(ranked_cvar[:5], start=1):
        print(
            f"{rank}. {', '.join(row['assets'])} | "
            f"CVaR={row['cvar']:.12f} | VaR={row['var']:.12f}"
        )
    print()

    print("=== Comparison With Previously Reported Baseline ===")
    assets_match = set(best_cvar["assets"]) == EXPECTED_ASSETS
    cvar_match = np.isclose(
        best_cvar["cvar"], EXPECTED_CVAR, atol=1e-9, rtol=1e-6
    )
    var_match = np.isclose(
        best_cvar["var"], EXPECTED_VAR, atol=1e-9, rtol=1e-6
    )
    mean_match = np.isclose(
        best_cvar["mean_return"], EXPECTED_MEAN_RETURN,
        atol=1e-9, rtol=1e-6
    )
    variance_match = np.isclose(
        best_cvar["daily_variance"], EXPECTED_DAILY_VARIANCE,
        atol=1e-9, rtol=1e-6
    )

    print(f"Expected assets: {', '.join(sorted(EXPECTED_ASSETS))}")
    print(f"Assets match: {assets_match}")
    print(
        f"CVaR match: {cvar_match} "
        f"(new={best_cvar['cvar']:.12f}, expected={EXPECTED_CVAR:.12f})"
    )
    print(
        f"VaR match: {var_match} "
        f"(new={best_cvar['var']:.12f}, expected={EXPECTED_VAR:.12f})"
    )
    print(
        f"Mean return match: {mean_match} "
        f"(new={best_cvar['mean_return']:.12f}, "
        f"expected={EXPECTED_MEAN_RETURN:.12f})"
    )
    print(
        f"Daily variance match: {variance_match} "
        f"(new={best_cvar['daily_variance']:.12e}, "
        f"expected={EXPECTED_DAILY_VARIANCE:.12e})"
    )

    if all((assets_match, cvar_match, var_match, mean_match, variance_match)):
        print("\nPASS: Independent baseline matches the previously reported result.")
    else:
        print(
            "\nREVIEW REQUIRED: At least one value differs from the previous "
            "baseline. Check the data file, asset columns, confidence level, "
            "and CVaR convention before drawing conclusions."
        )


if __name__ == "__main__":
    main()
