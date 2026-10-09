
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "GLD",
    "USO",
]

START_DATE = "2023-10-01"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_close_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract adjusted Close prices across yfinance versions."""

    if raw.empty:
        raise ValueError("No market data was downloaded.")

    if not isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns:
            raise ValueError("Close prices are missing.")
        return raw[["Close"]].rename(
            columns={"Close": TICKERS[0]}
        )

    # yfinance may arrange columns as:
    # (Price field, Ticker) or (Ticker, Price field).
    for level in range(raw.columns.nlevels):
        level_values = raw.columns.get_level_values(level)

        if "Close" in level_values:
            prices = raw.xs(
                "Close", axis=1, level=level
            )
            break
    else:
        raise ValueError("Could not locate Close prices.")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=TICKERS[0])

    return prices


def main():
    print("Downloading historical asset prices...")
    print(f"Assets: {', '.join(TICKERS)}")
    print(f"Start date: {START_DATE}")

    raw = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        multi_level_index=True,
        progress=False,
        threads=True,
    )

    prices = extract_close_prices(raw)

    # Retain only the requested assets, in fixed order.
    missing_tickers = [
        ticker for ticker in TICKERS
        if ticker not in prices.columns
    ]

    if missing_tickers:
        raise ValueError(
            f"Missing asset data: {missing_tickers}"
        )

    prices = prices[TICKERS].sort_index()

    # Remove rows where at least one asset has no price.
    # Do not forward-fill missing market observations.
    missing_before = prices.isna().sum()
    prices = prices.dropna(how="any")

    if len(prices) < 100:
        raise ValueError(
            "Too few complete observations. Check the "
            "download and ticker availability."
        )

    if (prices <= 0).any().any():
        raise ValueError("Non-positive prices detected.")

    # Daily simple returns.
    returns = prices.pct_change(fill_method=None)
    returns = returns.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(how="any")

    if returns.empty:
        raise ValueError("No valid returns calculated.")

    # Save clean data.
    prices_path = DATA_DIR / "asset_prices.csv"
    returns_path = DATA_DIR / "daily_returns.csv"

    prices.to_csv(prices_path, index_label="Date")
    returns.to_csv(returns_path, index_label="Date")

    # Basic data-quality and descriptive report.
    print("\nDownload successful.")
    print(f"Price observations: {len(prices)}")
    print(f"Return observations: {len(returns)}")
    print(f"Date range: {prices.index.min().date()} "
          f"to {prices.index.max().date()}")

    print("\nMissing prices before cleaning:")
    print(missing_before.to_string())

    print("\nAnnualized volatility estimates:")
    print(
        (returns.std() * np.sqrt(252))
        .sort_values(ascending=False)
        .round(4)
        .to_string()
    )

    print("\nSaved files:")
    print(prices_path)
    print(returns_path)

    print("\nPreview of daily returns:")
    print(returns.head().round(5).to_string())


if __name__ == "__main__":
    main()
