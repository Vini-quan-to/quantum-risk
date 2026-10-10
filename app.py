from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# QuantumRisk: Interactive portfolio-risk dashboard
# Save as app.py in the project root.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Current main.py writes these CSVs in the project root.
# Older experiment scripts may write CSVs inside data/.
RESULT_CANDIDATES = {
    "Classical portfolios": [
        PROJECT_ROOT / "classical_k4_results.csv",
        DATA_DIR / "classical_k4_results.csv",
    ],
    "Classical summary": [
        PROJECT_ROOT / "classical_k4_summary.csv",
        DATA_DIR / "classical_k4_summary.csv",
    ],
    "Quantum portfolio results": [
        DATA_DIR / "quantum_portfolio_results.csv",
        PROJECT_ROOT / "quantum_portfolio_results.csv",
    ],
    "Portfolio-size comparison": [
        DATA_DIR / "portfolio_size_comparison.csv",
        PROJECT_ROOT / "portfolio_size_comparison.csv",
    ],
}

# Values from the previous local QAOA-style statevector simulation.
# This app displays them as historical reference values; it does not rerun QAOA.
QAOA_REFERENCE = {
    "assets": "EFA, TLT, USO, XLP",
    "daily_variance": 4.44507648303e-05,
    "annualized_volatility": 0.10583758,
    "selected_bitstring_probability": 0.00273088,
    "rank": 1039,
    "total_portfolios": 4845,
}

st.set_page_config(
    page_title="QuantumRisk | Portfolio Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; padding-bottom: 2.5rem;}
    [data-testid="stMetric"] {
        background: rgba(128,128,128,0.07);
        border: 1px solid rgba(128,128,128,0.20);
        padding: 14px 16px;
        border-radius: 12px;
    }
    .small-note {font-size: 0.88rem; opacity: 0.8;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 QuantumRisk")
st.caption(
    "Portfolio-risk optimization · Exact classical benchmark · QAOA simulation reference"
)

# Resolve existing result paths dynamically.
available_files = {}
for label, candidates in RESULT_CANDIDATES.items():
    for candidate in candidates:
        if candidate.exists():
            available_files[label] = candidate
            break


def read_csv_safely(path):
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


def parse_assets(value):
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


classical_results_path = available_files.get("Classical portfolios")
classical_summary_path = available_files.get("Classical summary")
classical_df = (
    read_csv_safely(classical_results_path)
    if classical_results_path is not None
    else pd.DataFrame()
)
summary_df = (
    read_csv_safely(classical_summary_path)
    if classical_summary_path is not None
    else pd.DataFrame()
)

# Select the minimum-variance row from the full result table when possible.
classical_best = None
if not classical_df.empty and "daily_variance" in classical_df.columns:
    classical_best = classical_df.loc[classical_df["daily_variance"].idxmin()].to_dict()
elif not summary_df.empty:
    subset = summary_df
    if "objective" in subset.columns:
        matching = subset[subset["objective"].astype(str).str.contains("variance", case=False)]
        if not matching.empty:
            subset = matching
    if not subset.empty:
        classical_best = subset.iloc[0].to_dict()

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Dashboard section",
        ["Overview", "Portfolio explorer", "Experiment runner", "Project files"],
        label_visibility="collapsed",
    )
    st.divider()
    st.subheader("Benchmark settings")
    st.metric("Assets in dataset", "20" if classical_best else "—")
    st.metric("Portfolio size (k)", "4")
    st.caption("The classical benchmark checks every feasible 4-asset combination.")
    st.divider()
    st.caption("Project root")
    st.code(str(PROJECT_ROOT), language="text")

if page == "Overview":
    st.subheader("Optimization overview")

    if classical_best:
        assets = str(classical_best.get("assets", "—"))
        variance = float(classical_best.get("daily_variance", np.nan))
        volatility = float(classical_best.get("annualized_volatility", np.nan))
        cvar = float(classical_best.get("CVaR_95", np.nan))
        ret = float(classical_best.get("annualized_return_estimate", np.nan))
        runtime = float(classical_best.get("runtime_seconds", np.nan))
        if not np.isfinite(runtime) and not summary_df.empty and "runtime_seconds" in summary_df:
            runtime = float(summary_df["runtime_seconds"].iloc[0])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Annualized volatility", f"{volatility:.2%}" if np.isfinite(volatility) else "—")
        m2.metric("Annualized return estimate", f"{ret:.2%}" if np.isfinite(ret) else "—")
        m3.metric("95% CVaR", f"{cvar:.3%}" if np.isfinite(cvar) else "—")
        m4.metric("Portfolios evaluated", f"{len(classical_df):,}" if not classical_df.empty else "4,845")

        st.markdown("#### Exact classical minimum-variance portfolio")
        st.success(f"Selected assets: **{assets}**")
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("**Classical result details**")
            details = {
                "Daily variance": variance,
                "Annualized volatility": volatility,
                "Annualized return estimate": ret,
                "VaR 95%": float(classical_best.get("VaR_95", np.nan)),
                "CVaR 95%": cvar,
                "Maximum drawdown": float(classical_best.get("max_drawdown", np.nan)),
            }
            detail_df = pd.DataFrame(
                [{"Metric": key, "Value": value} for key, value in details.items()]
            )
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Previous QAOA-style simulation reference**")
            st.metric("Reference assets", QAOA_REFERENCE["assets"])
            st.metric("Daily variance", f"{QAOA_REFERENCE['daily_variance']:.3e}")
            st.metric("Annualized volatility", f"{QAOA_REFERENCE['annualized_volatility']:.2%}")
            st.caption(
                f"Selected bitstring probability: "
                f"{QAOA_REFERENCE['selected_bitstring_probability']:.6f} · "
                f"Rank {QAOA_REFERENCE['rank']:,}/{QAOA_REFERENCE['total_portfolios']:,}"
            )

        st.markdown("#### Risk comparison")
        chart_data = pd.DataFrame({
            "Method": ["Classical exact optimum", "Previous QAOA-style simulation"],
            "Daily variance": [variance, QAOA_REFERENCE["daily_variance"]],
            "Annualized volatility (%)": [
                volatility * 100 if np.isfinite(volatility) else np.nan,
                QAOA_REFERENCE["annualized_volatility"] * 100,
            ],
        })
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Daily variance**")
            st.bar_chart(chart_data.set_index("Method")[["Daily variance"]])
        with c2:
            st.markdown("**Annualized volatility (%)**")
            st.bar_chart(chart_data.set_index("Method")[["Annualized volatility (%)"]])

        if np.isfinite(variance) and variance > 0:
            gap = (QAOA_REFERENCE["daily_variance"] / variance - 1) * 100
            st.info(
                f"The previous QAOA-style reference portfolio has **{gap:.2f}% higher daily "
                "variance** than the exact classical optimum on the reported comparison. "
                "This is a result for these runs, not evidence of quantum advantage."
            )
    else:
        st.warning(
            "No classical result CSV was found. Run `python main.py` from the project root "
            "first, then refresh this page."
        )

    st.divider()
    st.caption(
        "Method note: the QAOA figures are reference values from a previous local "
        "classical statevector simulation. This dashboard does not run QAOA or use "
        "quantum hardware."
    )

elif page == "Portfolio explorer":
    st.subheader("Portfolio explorer")
    if classical_df.empty:
        st.warning("No classical portfolio results found. Run `python main.py` first.")
    else:
        numeric_cols = [
            c for c in ["daily_variance", "annualized_volatility", "CVaR_95", "VaR_95", "mean_daily_return"]
            if c in classical_df.columns
        ]
        if "daily_variance" in classical_df.columns:
            min_var = float(classical_df["daily_variance"].min())
            max_var = float(classical_df["daily_variance"].max())
            if min_var < max_var:
                low, high = st.slider(
                    "Filter daily variance range",
                    min_value=min_var,
                    max_value=max_var,
                    value=(min_var, max_var),
                    format="%.6g",
                )
                view_df = classical_df[
                    classical_df["daily_variance"].between(low, high)
                ].copy()
            else:
                view_df = classical_df.copy()
        else:
            view_df = classical_df.copy()

        if "daily_variance" in view_df.columns:
            view_df = view_df.sort_values("daily_variance")
        st.caption(f"Showing {len(view_df):,} of {len(classical_df):,} portfolios")
        st.dataframe(view_df, use_container_width=True, hide_index=True, height=460)
        st.download_button(
            "Download filtered portfolios CSV",
            data=view_df.to_csv(index=False).encode("utf-8"),
            file_name="quantumrisk_filtered_portfolios.csv",
            mime="text/csv",
        )

        if numeric_cols and not view_df.empty:
            chosen_metric = st.selectbox("Rank portfolios by", numeric_cols, index=0)
            ascending = chosen_metric not in ("mean_daily_return",)
            ranked = view_df.sort_values(chosen_metric, ascending=ascending).head(15)
            st.markdown("#### Top 15 portfolios by selected metric")
            st.dataframe(ranked, use_container_width=True, hide_index=True)

elif page == "Experiment runner":
    st.subheader("Experiment runner")
    st.write(
        "Run the root-level benchmark or an existing script from `src/`. "
        "The script must be compatible with your local Python environment."
    )

    script_options = {"Current dashboard benchmark (main.py)": PROJECT_ROOT / "main.py"}
    existing_scripts = {
        "Classical validator (src/validate_portfolios_k4.py)": PROJECT_ROOT / "src" / "validate_portfolios_k4.py",
        "Quantum optimizer (src/quantum_optimizer.py)": PROJECT_ROOT / "src" / "quantum_optimizer.py",
    }
    script_options.update({label: path for label, path in existing_scripts.items() if path.exists()})
    selected_script = st.selectbox("Experiment", list(script_options.keys()))
    if st.button("▶ Run selected experiment", type="primary"):
        script_path = script_options[selected_script]
        with st.status(f"Running {script_path.name}...", expanded=True) as status:
            try:
                result = subprocess.run(
                    [sys.executable, "-u", str(script_path)],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=600,
                )
                combined = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
                st.code(combined[-30000:] or "(No output)", language="text")
                if result.returncode == 0:
                    status.update(label="Experiment completed", state="complete")
                else:
                    status.update(label=f"Experiment failed (exit {result.returncode})", state="error")
            except subprocess.TimeoutExpired:
                status.update(label="Experiment timed out after 10 minutes", state="error")
            except Exception as exc:
                status.update(label=f"Could not run experiment: {exc}", state="error")

elif page == "Project files":
    st.subheader("Available result files")
    if not available_files:
        st.info("No result CSVs found yet. Run `python main.py` first.")
    else:
        for label, path in available_files.items():
            with st.expander(f"{label} — {path.relative_to(PROJECT_ROOT)}"):
                file_df = read_csv_safely(path)
                if not file_df.empty:
                    st.dataframe(file_df.head(100), use_container_width=True, hide_index=True)
                st.download_button(
                    f"Download {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="text/csv",
                    key=f"download_{label}",
                )

st.divider()
st.caption(
    "QuantumRisk research prototype · Equal-weight portfolios of four assets · "
    "Classical exhaustive search is the exact benchmark for the supplied dataset. "
    "No quantum advantage is claimed."
)
