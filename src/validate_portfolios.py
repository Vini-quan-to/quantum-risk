
from pathlib import Path
from itertools import combinations
import time

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RETURNS_FILE = DATA_DIR / "daily_returns.csv"
QAOA_FILE = DATA_DIR / "quantum_portfolio_results.csv"

VALIDATION_FILE = DATA_DIR / "portfolio_validation.csv"
CLASSICAL_FILE = DATA_DIR / "exhaustive_classical_k4.csv"

PORTFOLIO_SIZE = 4
RANDOM_PORTFOLIOS = 1000
RANDOM_SEED = 2026
TRADING_DAYS = 252


# ============================================================
# LOAD DATA
# ============================================================

def load_returns():
    if not RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Returns file not found: {RETURNS_FILE}"
        )

    returns = pd.read_csv(RETURNS_FILE)

    date_columns = [
        col for col in returns.columns
        if str(col).strip().lower()
        in {"date", "datetime", "timestamp"}
    ]

    returns = returns.drop(columns=date_columns)
    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(axis=0, how="any")

    if returns.empty or returns.shape[0] < 2:
        raise ValueError("Insufficient valid return observations.")

    if returns.columns.duplicated().any():
        raise ValueError("Duplicate asset names found.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns contain non-finite values.")

    if returns.shape[1] < PORTFOLIO_SIZE:
        raise ValueError("Not enough assets for the requested portfolio.")

    return returns


# ============================================================
# LOAD QAOA PORTFOLIO
# ============================================================

def load_qaoa_assets(available_assets):
    if not QAOA_FILE.exists():
        raise FileNotFoundError(
            f"QAOA results not found: {QAOA_FILE}\n"
            "Run src/quantum_optimizer.py first."
        )

    saved = pd.read_csv(QAOA_FILE)

    if "asset" not in saved.columns:
        raise ValueError(
            "Expected an 'asset' column in quantum_portfolio_results.csv."
        )

    selected = saved["asset"].dropna().astype(str).tolist()

    if len(selected) != PORTFOLIO_SIZE:
        raise ValueError(
            f"Expected {PORTFOLIO_SIZE} selected assets, "
            f"but found {len(selected)}."
        )

    if len(set(selected)) != len(selected):
        raise ValueError("QAOA output contains duplicate assets.")

    unknown = sorted(set(selected) - set(available_assets))

    if unknown:
        raise ValueError(
            f"QAOA output contains assets absent from returns data: {unknown}"
        )

    return selected


# ============================================================
# PORTFOLIO METRICS
# ============================================================

def calculate_metrics(returns, selected_assets):
    portfolio_returns = returns[selected_assets].mean(axis=1)

    daily_variance = float(
        portfolio_returns.var(ddof=1)
    )

    daily_volatility = float(
        portfolio_returns.std(ddof=1)
    )

    daily_mean = float(
        portfolio_returns.mean()
    )

    losses = -portfolio_returns.to_numpy()

    var_95 = float(
        np.quantile(losses, 0.95)
    )

    tail_losses = losses[losses >= var_95]

    cvar_95 = float(
        tail_losses.mean()
    )

    wealth = (1.0 + portfolio_returns).cumprod()

    drawdown = 1.0 - wealth / wealth.cummax()

    return {
        "assets": ", ".join(selected_assets),
        "portfolio_size": len(selected_assets),
        "daily_variance": daily_variance,
        "daily_volatility": daily_volatility,
        "annualized_volatility": (
            daily_volatility * np.sqrt(TRADING_DAYS)
        ),
        "daily_mean_return": daily_mean,
        "annualized_mean_return": daily_mean * TRADING_DAYS,
        "historical_var_95": var_95,
        "historical_cvar_95": cvar_95,
        "maximum_drawdown": float(drawdown.max()),
    }


# ============================================================
# EXHAUSTIVE CLASSICAL SEARCH
# ============================================================

def exhaustive_search(returns):
    """
    Enumerate every portfolio of exactly four assets.

    All assets are equally weighted. This finds the exact
    minimum-variance portfolio among the combinations tested.
    """

    assets = list(returns.columns)
    covariance = returns.cov().to_numpy(dtype=np.float64)

    results = []

    start = time.perf_counter()

    for selection in combinations(
        range(len(assets)),
        PORTFOLIO_SIZE,
    ):
        names = [assets[i] for i in selection]

        sub_covariance = covariance[np.ix_(selection, selection)]

        weights = np.full(
            PORTFOLIO_SIZE,
            1.0 / PORTFOLIO_SIZE,
        )

        variance = float(
            weights @ sub_covariance @ weights
        )

        results.append({
            "assets": ", ".join(names),
            "asset_tuple": names,
            "daily_variance": variance,
        })

    elapsed = time.perf_counter() - start

    results.sort(key=lambda item: item["daily_variance"])

    best = results[0]

    classical_metrics = calculate_metrics(
        returns,
        best["asset_tuple"],
    )

    output = pd.DataFrame([
        {
            "rank": rank,
            "assets": item["assets"],
            "daily_variance": item["daily_variance"],
            "annualized_volatility": np.sqrt(
                item["daily_variance"] * TRADING_DAYS
            ),
        }
        for rank, item in enumerate(results, start=1)
    ])

    output.to_csv(CLASSICAL_FILE, index=False)

    return results, classical_metrics, elapsed


# ============================================================
# RANDOM FEASIBLE PORTFOLIO BASELINE
# ============================================================

def random_baseline(returns, number_of_samples):
    """
    Sample random four-asset portfolios without replacement.
    These are benchmark samples, not an optimization algorithm.
    """

    rng = np.random.default_rng(RANDOM_SEED)
    assets = list(returns.columns)

    if number_of_samples <= 0:
        raise ValueError("Number of random samples must be positive.")

    covariance = returns.cov().to_numpy(dtype=np.float64)

    variances = []

    for _ in range(number_of_samples):
        selection = rng.choice(
            len(assets),
            size=PORTFOLIO_SIZE,
            replace=False,
        )

        weights = np.full(
            PORTFOLIO_SIZE,
            1.0 / PORTFOLIO_SIZE,
        )

        sub_covariance = covariance[np.ix_(selection, selection)]

        variance = float(
            weights @ sub_covariance @ weights
        )

        variances.append(variance)

    return np.asarray(variances)


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():
    print("=" * 72)
    print("QUANTUMRISK: PORTFOLIO VALIDATION")
    print("=" * 72)

    returns = load_returns()
    assets = list(returns.columns)

    print(f"Assets: {len(assets)}")
    print(f"Observations: {len(returns)}")
    print(f"Portfolio size: {PORTFOLIO_SIZE}")

    qaoa_assets = load_qaoa_assets(assets)

    print("\nQAOA portfolio:")
    print(qaoa_assets)

    qaoa_metrics = calculate_metrics(
        returns,
        qaoa_assets,
    )

    print("\nRunning exhaustive classical search...")

    all_portfolios, classical_metrics, search_time = (
        exhaustive_search(returns)
    )

    qaoa_variance = qaoa_metrics["daily_variance"]
    classical_variance = classical_metrics["daily_variance"]

    if classical_variance <= 0:
        raise ValueError(
            "Classical minimum variance is non-positive; "
            "cannot calculate a relative gap."
        )

    variance_gap_percent = (
        (qaoa_variance - classical_variance)
        / classical_variance
        * 100.0
    )

    qaoa_rank = next(
        (
            rank
            for rank, item in enumerate(all_portfolios, start=1)
            if set(item["asset_tuple"]) == set(qaoa_assets)
        ),
        None,
    )

    random_variances = random_baseline(
        returns,
        RANDOM_PORTFOLIOS,
    )

    random_median = float(np.median(random_variances))
    random_best = float(np.min(random_variances))

    summary = pd.DataFrame([
        {
            "method": "QAOA statevector candidate",
            **qaoa_metrics,
        },
        {
            "method": "Exact classical minimum variance",
            **classical_metrics,
        },
    ])

    summary.to_csv(VALIDATION_FILE, index=False)

    print("\n" + "=" * 72)
    print("COMPARISON")
    print("=" * 72)

    comparison = pd.DataFrame([
        {
            "method": "QAOA candidate",
            "assets": qaoa_metrics["assets"],
            "daily_variance": qaoa_variance,
            "annualized_volatility": (
                qaoa_metrics["annualized_volatility"]
            ),
        },
        {
            "method": "Exact classical optimum",
            "assets": classical_metrics["assets"],
            "daily_variance": classical_variance,
            "annualized_volatility": (
                classical_metrics["annualized_volatility"]
            ),
        },
        {
            "method": "Random sample median",
            "assets": "Varies by sample",
            "daily_variance": random_median,
            "annualized_volatility": np.sqrt(
                random_median * TRADING_DAYS
            ),
        },
        {
            "method": "Best of random samples",
            "assets": "Varies by sample",
            "daily_variance": random_best,
            "annualized_volatility": np.sqrt(
                random_best * TRADING_DAYS
            ),
        },
    ])

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.10f}",
        )
    )

    print("\n" + "=" * 72)
    print("VALIDATION DETAILS")
    print("=" * 72)

    print(f"Feasible portfolios enumerated: {len(all_portfolios):,}")
    print(f"Classical search runtime: {search_time:.4f} seconds")
    print(f"QAOA portfolio rank: {qaoa_rank:,} of {len(all_portfolios):,}")
    print(f"Variance gap above classical optimum: {variance_gap_percent:.2f}%")
    print(f"Random portfolios sampled: {RANDOM_PORTFOLIOS:,}")

    if set(qaoa_assets) == set(classical_metrics["assets"].split(", ")):
        print("\nQAOA selected the same portfolio as the classical optimum.")
    else:
        print("\nQAOA did not select the classical minimum-variance portfolio.")

    print("\nSaved files:")
    print(VALIDATION_FILE)
    print(CLASSICAL_FILE)

    print(
        "\nInterpretation: the exhaustive search is an exact optimum "
        "for equal-weight portfolios of four assets within this dataset. "
        "It does not prove a global optimum for variable weights, "
        "different portfolio sizes, or future returns."
    )


if __name__ == "__main__":
    main()
