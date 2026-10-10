"""
QuantumRisk — Variance vs CVaR comparison report.

Run from the project root:
    python src/compare_variance_cvar.py

The script reads existing CSVs when present and writes:
    data/variance_cvar_comparison.csv
    data/variance_cvar_comparison.txt

It does not modify any existing experiment outputs.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Prefer the most recent refined variance experiment, then fall back to older runs.
CANDIDATES = {
    "classical_variance": [
        "classical_k4_results.csv",
        "classical_variance_results.csv",
    ],
    "classical_cvar": [
        "classical_cvar_results.csv",
    ],
    "quantum_variance": [
        "quantum_portfolio_results_p2_refined.csv",
        "quantum_portfolio_results_p2_multistart.csv",
        "quantum_portfolio_results.csv",
    ],
    "quantum_cvar": [
        "quantum_cvar_portfolio_results.csv",
    ],
}

OUT_CSV = DATA_DIR / "variance_cvar_comparison.csv"
OUT_TXT = DATA_DIR / "variance_cvar_comparison.txt"


def first_existing(key):
    for filename in CANDIDATES[key]:
        path = DATA_DIR / filename
        if path.exists():
            return path
    return None


def find_column(frame, names):
    lower_map = {str(c).strip().lower(): c for c in frame.columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    # Allow partial matching for minor naming variations.
    for col in frame.columns:
        normalized = str(col).strip().lower()
        if any(name.lower() in normalized for name in names):
            return col
    return None


def parse_portfolio(value):
    if pd.isna(value):
        return ""
    return str(value).replace("|", ", ").strip()


def load_csv(key):
    path = first_existing(key)
    if path is None:
        return None, None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: could not read {path.name}: {exc}")
        return None, path
    if frame.empty:
        print(f"WARNING: {path.name} is empty.")
        return None, path
    return frame, path


def metric_row(label, frame, path, objective):
    """Extract optimum or distribution summary from a result CSV."""
    portfolio_col = find_column(
        frame, ["assets", "portfolio", "portfolio_assets", "selected_assets"]
    )
    probability_col = find_column(
        frame, ["probability", "state_probability", "portfolio_probability"]
    )
    variance_col = find_column(
        frame, ["daily_variance", "variance", "portfolio_variance"]
    )
    # Prefer exact metric names. Broad substring matching for "var" is unsafe
    # because it can match "variance" and produce a false VaR value.
    exact_columns = {str(c).strip().lower(): c for c in frame.columns}
    cvar_col = next(
        (
            exact_columns[name]
            for name in (
                "historical_cvar_loss",
                "cvar_95",
                "cvar_loss",
                "portfolio_cvar",
            )
            if name in exact_columns
        ),
        None,
    )
    var_col = next(
        (
            exact_columns[name]
            for name in (
                "historical_var_loss",
                "var_95",
                "var_loss",
                "portfolio_var",
            )
            if name in exact_columns
        ),
        None,
    )
    rank_col = find_column(frame, ["rank_by_cvar", "rank"])

    # Classical full-enumeration results are sorted by their objective;
    # quantum result files contain a probability distribution.
    is_quantum = label.startswith("Quantum")
    row = {
        "experiment": label,
        "source_file": path.name if path else "",
        "portfolio": "",
        "probability": np.nan,
        "daily_variance": np.nan,
        "historical_var_loss": np.nan,
        "historical_cvar_loss": np.nan,
        "portfolio_rank": np.nan,
        "expected_daily_variance": np.nan,
        "expected_cvar_loss": np.nan,
        "notes": "",
    }

    if portfolio_col is None:
        row["notes"] = "Could not identify a portfolio/assets column."
        return row

    # Use the relevant objective for classical optimum, and highest-probability
    # row for quantum modal portfolio.
    if is_quantum and probability_col is not None:
        probs = pd.to_numeric(frame[probability_col], errors="coerce").fillna(0)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        modal_idx = probs.idxmax()
        chosen = frame.loc[modal_idx]
        row["portfolio"] = parse_portfolio(chosen[portfolio_col])
        row["probability"] = float(probs.loc[modal_idx])
        if variance_col:
            vals = pd.to_numeric(frame[variance_col], errors="coerce")
            row["daily_variance"] = float(vals.loc[modal_idx])
            row["expected_daily_variance"] = float(
                (vals.fillna(0) * probs).sum()
            )
        if cvar_col:
            vals = pd.to_numeric(frame[cvar_col], errors="coerce")
            row["historical_cvar_loss"] = float(vals.loc[modal_idx])
            row["expected_cvar_loss"] = float(
                (vals.fillna(0) * probs).sum()
            )
        if var_col:
            vals = pd.to_numeric(frame[var_col], errors="coerce")
            row["historical_var_loss"] = float(vals.loc[modal_idx])
        row["notes"] = "Modal portfolio; expected metrics use normalized CSV probabilities."
        return row

    objective_col = cvar_col if objective == "cvar" else variance_col
    if objective_col is None:
        row["notes"] = f"Could not identify the {objective} objective column."
        return row

    values = pd.to_numeric(frame[objective_col], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        row["notes"] = f"No numeric values in {objective_col}."
        return row
    chosen_idx = valid.idxmin()
    chosen = frame.loc[chosen_idx]
    row["portfolio"] = parse_portfolio(chosen[portfolio_col])
    if variance_col:
        row["daily_variance"] = pd.to_numeric(
            pd.Series([chosen[variance_col]]), errors="coerce"
        ).iloc[0]
    if cvar_col:
        row["historical_cvar_loss"] = pd.to_numeric(
            pd.Series([chosen[cvar_col]]), errors="coerce"
        ).iloc[0]
    if var_col:
        row["historical_var_loss"] = pd.to_numeric(
            pd.Series([chosen[var_col]]), errors="coerce"
        ).iloc[0]
    if rank_col:
        row["portfolio_rank"] = chosen[rank_col]
    row["notes"] = f"Minimum {objective} in {path.name}."
    return row


def fmt(value, percent=False):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.4%}" if percent else f"{x:.10g}"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("Classical minimum variance", "classical_variance", "variance"),
        ("Classical minimum CVaR", "classical_cvar", "cvar"),
        ("Quantum variance simulation (modal)", "quantum_variance", "variance"),
        ("Quantum CVaR simulation (modal)", "quantum_cvar", "cvar"),
    ]

    rows = []
    missing = []
    for label, key, objective in specs:
        frame, path = load_csv(key)
        if frame is None:
            missing.append((label, [str(DATA_DIR / f) for f in CANDIDATES[key]]))
            continue
        rows.append(metric_row(label, frame, path, objective))

    if not rows:
        print(f"No recognized result CSVs found in {DATA_DIR}")
        print("Expected at least one of:")
        for candidates in CANDIDATES.values():
            for filename in candidates:
                print(f"  data/{filename}")
        sys.exit(1)

    report = pd.DataFrame(rows)
    report.to_csv(OUT_CSV, index=False)

    lines = [
        "QuantumRisk — Variance vs CVaR comparison",
        "=" * 48,
        "",
        "This report summarizes existing CSV outputs. It does not rerun optimizers.",
        "Quantum entries refer to classical statevector/QAOA-style simulations,",
        "not execution on quantum hardware.",
        "",
    ]
    for _, row in report.iterrows():
        lines.extend([
            str(row["experiment"]),
            "-" * len(str(row["experiment"])),
            f"Source file: {row['source_file']}",
            f"Portfolio: {row['portfolio'] or 'n/a'}",
            f"Modal probability: {fmt(row['probability'], percent=True)}",
            f"Daily variance: {fmt(row['daily_variance'])}",
            f"Historical VaR loss: {fmt(row['historical_var_loss'], percent=True)}",
            f"Historical CVaR loss: {fmt(row['historical_cvar_loss'], percent=True)}",
            f"Expected daily variance: {fmt(row['expected_daily_variance'])}",
            f"Expected CVaR loss: {fmt(row['expected_cvar_loss'], percent=True)}",
            f"Notes: {row['notes']}",
            "",
        ])

    if missing:
        lines.extend([
            "Missing result files",
            "--------------------",
            "Some experiments were skipped because no candidate CSV was found:",
        ])
        for label, candidates in missing:
            lines.append(f"- {label}:")
            lines.extend([f"    {p}" for p in candidates])
        lines.append("")

    lines.extend([
        "Interpretation notes",
        "--------------------",
        "- Classical rows show the minimum objective found in the corresponding full-result CSV.",
        "- Quantum rows show the highest-probability (modal) portfolio and expected metrics if available.",
        "- Compare CVaR only with CVaR and variance only with variance; their numerical scales differ.",
        "- Historical estimates are in-sample and do not guarantee future performance.",
    ])

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved comparison CSV: {OUT_CSV}")
    print(f"Saved comparison summary: {OUT_TXT}")


if __name__ == "__main__":
    main()
