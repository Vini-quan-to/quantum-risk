
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RISK_FILE = DATA_DIR / "risk_comparison.csv"
QAOA_FILE = DATA_DIR / "qaoa_aer_results.csv"

FIGURE_DIR = PROJECT_ROOT / "figures"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    if not RISK_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {RISK_FILE}"
        )

    if not QAOA_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {QAOA_FILE}"
        )

    risk = pd.read_csv(RISK_FILE)
    qaoa = pd.read_csv(QAOA_FILE)

    qaoa["matched_exact_optimum"] = (
        qaoa["matched_exact_optimum"]
        .astype(str).str.lower().eq("true")
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Figure 1: Tail risk
    # --------------------------------------------------

    labels = [
        "QAOA minimum variance",
        "Classical minimum CVaR",
    ]

    qaoa_row = risk[
        risk["method"] == "QAOA minimum-variance portfolio"
    ].iloc[0]

    classical_row = risk[
        risk["method"] == "Classical minimum-CVaR portfolio"
    ].iloc[0]

    var_values = [
        qaoa_row["VaR_95"] * 100,
        classical_row["VaR_95"] * 100,
    ]

    cvar_values = [
        qaoa_row["CVaR_95"] * 100,
        classical_row["CVaR_95"] * 100,
    ]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        var_values,
        width,
        label="VaR 95%",
    )

    ax.bar(
        x + width / 2,
        cvar_values,
        width,
        label="CVaR 95%",
    )

    ax.set_ylabel("Historical loss (%)")
    ax.set_title("Portfolio Tail-Risk Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "risk_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------
    # Figure 2: Return and volatility
    # --------------------------------------------------

    returns = np.array([
        qaoa_row["annualized_return_estimate"] * 100,
        classical_row["annualized_return_estimate"] * 100,
    ])

    volatility = np.array([
        qaoa_row["annualized_volatility"] * 100,
        classical_row["annualized_volatility"] * 100,
    ])

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, label in enumerate(labels):
        ax.scatter(
            volatility[i],
            returns[i],
            s=120,
            label=label,
        )
        ax.annotate(
            label,
            (volatility[i], returns[i]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Annualized volatility (%)")
    ax.set_ylabel("Annualized return estimate (%)")
    ax.set_title("Historical Return–Volatility Comparison")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "return_volatility.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------
    # Figure 3: QAOA run outcomes
    # --------------------------------------------------

    exact_count = int(
        qaoa["matched_exact_optimum"].sum()
    )

    other_count = len(qaoa) - exact_count

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        ["Exact optimum", "Other result"],
        [exact_count, other_count],
    )

    ax.set_ylabel("Number of runs")
    ax.set_title(
        f"QAOA Exact-Optimum Frequency "
        f"({exact_count}/{len(qaoa)} runs)"
    )
    ax.set_ylim(0, max(1, len(qaoa)) + 1)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "qaoa_reliability.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("=" * 60)
    print("QUANTUMRISK VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Risk comparison: {FIGURE_DIR / 'risk_comparison.png'}")
    print(f"Return-volatility plot: {FIGURE_DIR / 'return_volatility.png'}")
    print(f"QAOA reliability: {FIGURE_DIR / 'qaoa_reliability.png'}")
    print("\nAll figures saved successfully.")


if __name__ == "__main__":
    main()
