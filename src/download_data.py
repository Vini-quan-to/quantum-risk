
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# QuantumRisk: market data configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

PRICES_FILE = DATA_DIR / "asset_prices.csv"
RETURNS_FILE = DATA_DIR / "daily_returns.csv"

START_DATE = "2023-10-01"

# Exactly 20 ETFs.
# These span multiple market sectors and asset classes.
TICKERS = [
    "SPY",  # US broad-market equities
    "QQQ",  # Nasdaq-100 equities
    "IWM",  # US small-cap equities
    "EFA",  # Developed international equities
    "EEM",  # Emerging-market equities
    "TLT",  # Long-term US Treasury bonds
    "GLD",  # Gold
    "USO",  # Oil
    "XLK",  # Technology sector
    "XLF",  # Financial sector
    "XLE",  # Energy sector
    "XLV",  # Healthcare sector
    "XLI",  # Industrial sector
    "XLP",  # Consumer staples sector
    "XLY",  # Consumer discretionary sector
    "XLU",  # Utilities sector
    "VNQ",  # US real estate
    "HYG",  # High-yield corporate bonds
    "LQD",  # Investment-grade corporate bonds
    "AGG",  # US aggregate bonds
]

MINIMUM_OBSERVATIONS = 100


# ============================================================
# Download and validate data
# ============================================================

def download_prices():
    """Download adjusted daily closing prices for all configured ETFs."""

    print("=" * 68)
    print("QUANTUMRISK: 20-ASSET DATA DOWNLOAD")
    print("=" * 68)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if len(TICKERS) != 20:
        raise ValueError(
            f"Expected exactly 20 tickers, found {len(TICKERS)}."
        )

    if len(set(TICKERS)) != len(TICKERS):
        raise ValueError("The ticker list contains duplicates.")

    print(f"\nNumber of assets: {len(TICKERS)}")
    print(f"Start date: {START_DATE}")
    print(f"Output directory: {DATA_DIR}")

    print("\nDownloading daily adjusted prices...")

    downloaded = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )

    if downloaded.empty:
        raise RuntimeError(
            "Yahoo Finance returned no data. "
            "Check your internet connection and try again."
        )

    # With multiple tickers, yfinance normally returns
    # MultiIndex columns, with price fields and ticker symbols.
    if isinstance(downloaded.columns, pd.MultiIndex):
        available_fields = downloaded.columns.get_level_values(0)

        if "Close" in available_fields:
            prices = downloaded["Close"].copy()
        else:
            raise RuntimeError(
                "Could not find the adjusted Close price field. "
                f"Received columns: {downloaded.columns.tolist()}"
            )
    else:
        # Defensive handling for a single-level result.
        if "Close" not in downloaded.columns:
            raise RuntimeError(
                "Could not find the Close price field in downloaded data."
            )

        prices = downloaded[["Close"]].copy()

    # Normalize column labels and preserve the configured ticker order.
    prices.columns = [str(column) for column in prices.columns]

    missing_tickers = [
        ticker for ticker in TICKERS
        if ticker not in prices.columns
    ]

    if missing_tickers:
        raise RuntimeError(
            "The following ETFs could not be downloaded: "
            f"{missing_tickers}"
        )

    prices = prices[TICKERS].sort_index()

    # Convert any invalid prices to missing values.
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan)

    print("\nMissing prices before cleaning:")
    print(prices.isna().sum().to_string())

    # Use common dates so every return observation covers
    # the same set of assets.
    original_rows = len(prices)
    prices = prices.dropna(axis=0, how="any")

    prices = prices[
        (prices > 0).all(axis=1)
    ]

    if len(prices) < MINIMUM_OBSERVATIONS:
        raise RuntimeError(
            f"Only {len(prices)} complete price rows remain; "
            f"at least {MINIMUM_OBSERVATIONS} are required."
        )

    returns = prices.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(axis=0, how="any")

    if returns.empty:
        raise RuntimeError("No valid daily returns were generated.")

    if not np.isfinite(returns.to_numpy()).all():
        raise RuntimeError("Daily returns contain non-finite values.")

    if returns.shape[1] != 20:
        raise RuntimeError(
            f"Expected 20 return columns, got {returns.shape[1]}."
        )

    # Save the same files used by the existing project.
    prices.to_csv(PRICES_FILE, index_label="Date")
    returns.to_csv(RETURNS_FILE, index_label="Date")

    # ========================================================
    # Report
    # ========================================================

    print("\n" + "=" * 68)
    print("DOWNLOAD SUMMARY")
    print("=" * 68)

    print(f"Requested tickers: {len(TICKERS)}")
    print(f"Successfully downloaded: {len(prices.columns)}")
    print(f"Original price rows: {original_rows}")
    print(f"Complete price rows: {len(prices)}")
    print(f"Daily return rows: {len(returns)}")
    print(f"Date range: {prices.index.min().date()} to "
          f"{prices.index.max().date()}")

    print("\nAssets:")
    for index, ticker in enumerate(prices.columns, start=1):
        print(f"{index:2}. {ticker}")

    annualized_volatility = (
        returns.std(ddof=1) * np.sqrt(252) * 100
    )

    print("\nAnnualized historical volatility:")
    print(annualized_volatility.round(2).astype(str).add("%").to_string())

    print("\nDaily returns preview:")
    print(returns.head().round(5).to_string())

    print("\nSaved files:")
    print(f"Prices:  {PRICES_FILE}")
    print(f"Returns: {RETURNS_FILE}")

    print("\n20-asset data download completed successfully.")


if __name__ == "__main__":
    download_prices()
