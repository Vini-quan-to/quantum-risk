"""
QuantumRisk — Portfolio Risk Dashboard
Run from the project root:
    streamlit run dashboard.py

Expected files:
    data/daily_returns.csv
    data/quantum_portfolio_results.csv
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Page configuration and visual theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="QuantumRisk | Portfolio Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
      :root {
        --qr-bg: #0b1120;
        --qr-panel: #111b2e;
        --qr-panel-2: #17243a;
        --qr-border: #293750;
        --qr-text: #edf4ff;
        --qr-muted: #91a2bd;
        --qr-teal: #5eead4;
        --qr-blue: #60a5fa;
        --qr-amber: #fbbf24;
      }
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: radial-gradient(ellipse at 15% 0%, #172b43 0%, #0b1120 48%, #080d18 100%); color: var(--qr-text); }
      [data-testid="stHeader"] { background: rgba(11,17,32,0.85); }
      [data-testid="stSidebar"] { background: #0e1728; border-right: 1px solid var(--qr-border); }
      [data-testid="stSidebar"] * { color: var(--qr-text); }
      h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.035em; }
      h1 { font-size: 2.45rem !important; }
      h2 { font-size: 1.4rem !important; }
      h3 { font-size: 1.05rem !important; }
      p, label, .stCaption { color: var(--qr-muted); }
      div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #17243a, #101a2b);
        border: 1px solid var(--qr-border);
        border-radius: 15px;
        padding: 17px 18px;
        min-height: 112px;
      }
      div[data-testid="stMetricLabel"] p { color: #a9bad2 !important; font-size: 0.82rem; }
      div[data-testid="stMetricValue"] { color: #edf4ff; font-family: 'Space Grotesk', sans-serif; }
      div[data-testid="stMetricDelta"] { color: var(--qr-amber) !important; }
      div[data-testid="stDataFrame"], div[data-testid="stTable"] { border: 1px solid var(--qr-border); border-radius: 12px; overflow: hidden; }
      div[data-testid="stExpander"] { background: #111b2e; border: 1px solid var(--qr-border); border-radius: 12px; }
      .qr-eyebrow { color: var(--qr-teal); font-size: .76rem; letter-spacing: .17em; font-weight: 700; text-transform: uppercase; margin-bottom: .4rem; }
      .qr-subtitle { color: #91a2bd; margin-top: -.45rem; }
      .qr-panel { background: linear-gradient(145deg, rgba(23,36,58,.95), rgba(15,24,41,.95)); border: 1px solid #293750; border-radius: 16px; padding: 18px 20px; }
      .qr-pill { display:inline-block; padding: 4px 9px; border-radius: 999px; font-size: .72rem; font-weight: 700; background: #153c3b; color: #5eead4; border: 1px solid #22615b; }
      .qr-muted { color: #91a2bd; font-size: .85rem; }
      .qr-portfolio { font-family: 'Space Grotesk', sans-serif; font-size: 1.12rem; font-weight: 600; color: #edf4ff; margin: .35rem 0; }
      hr { border-color: #293750 !important; }
      button[kind="primary"] { background: #0f766e; border: 1px solid #2dd4bf; }
      [data-testid="stPlotlyChart"] { background: rgba(17,27,46,.58); border: 1px solid #293750; border-radius: 14px; padding: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RETURNS_PATH = DATA_DIR / "daily_returns.csv"
RESULTS_PATH = DATA_DIR / "quantum_portfolio_results.csv"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#c7d5e8"),
    margin=dict(l=18, r=18, t=55, b=18),
    xaxis=dict(gridcolor="#293750", zerolinecolor="#293750", linecolor="#293750"),
    yaxis=dict(gridcolor="#293750", zerolinecolor="#293750", linecolor="#293750"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


# ---------------------------------------------------------------------------
# Data loading and calculations
# ---------------------------------------------------------------------------
@st.cache_data
def load_returns(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="any")
    return frame


@st.cache_data
def load_results(path: str, modified_time: float) -> pd.DataFrame:
    # modified_time invalidates Streamlit cache when optimizer rewrites the CSV.
    return pd.read_csv(path)


def exact_minimum_variance(returns: pd.DataFrame, k: int) -> tuple[list[str], float, int]:
    names = list(returns.columns)
    if k > len(names):
        raise ValueError("Portfolio size cannot exceed the number of assets.")
    cov = returns.cov().to_numpy()
    best_combo: tuple[int, ...] | None = None
    best_var = float("inf")
    count = 0
    for combo in combinations(range(len(names)), k):
        sub_cov = cov[np.ix_(combo, combo)]
        variance = float(sub_cov.sum() / (k * k))
        count += 1
        if variance < best_var:
            best_var = variance
            best_combo = combo
    return [names[i] for i in (best_combo or ())], best_var, count


def portfolio_label(value: str) -> str:
    return str(value).replace("|", " · ")


def format_scientific(value: float) -> str:
    return f"{value:.4e}"


def set_chart_theme(fig, height: int = 440):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


# ---------------------------------------------------------------------------
# Load source data
# ---------------------------------------------------------------------------
if not RETURNS_PATH.exists():
    st.error(f"Missing data file: `{RETURNS_PATH}`")
    st.info("Place `daily_returns.csv` in the project's `data` folder.")
    st.stop()

returns = load_returns(str(RETURNS_PATH))
if returns.empty:
    st.error("The returns file has no usable numeric data.")
    st.stop()

if RESULTS_PATH.exists():
    results = load_results(str(RESULTS_PATH), RESULTS_PATH.stat().st_mtime)
else:
    results = pd.DataFrame()

asset_names = list(returns.columns)
asset_count = len(asset_names)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="qr-eyebrow">QuantumRisk</div>', unsafe_allow_html=True)
    st.markdown("## Control center")
    st.caption("Configure the benchmark and explore the saved simulation.")
    portfolio_size = st.slider(
        "Assets per portfolio",
        min_value=1,
        max_value=asset_count,
        value=min(4, asset_count),
        help="Changes the exact classical benchmark. Saved simulation results retain the k used by the optimizer.",
    )
    depths = sorted(results["depth"].dropna().unique().tolist()) if (
        not results.empty and "depth" in results.columns
    ) else []
    selected_depth = st.selectbox("Simulation depth", depths) if depths else None
    if not results.empty and "probability" in results.columns:
        top_limit = min(100, len(results))
        default_top = min(20, top_limit)
        top_n = st.slider(
            "Portfolios in chart/table",
            min_value=min(5, top_limit),
            max_value=top_limit,
            value=max(min(20, top_limit), min(5, top_limit)),
            step=5 if top_limit >= 10 else 1,
        )
    else:
        top_n = 20
    st.divider()
    st.markdown("### Data sources")
    st.caption("Historical daily returns")
    st.code("data/daily_returns.csv", language="text")
    st.caption("Optimizer probability output")
    st.code("data/quantum_portfolio_results.csv", language="text")
    st.caption("Research prototype · no live trades")


# ---------------------------------------------------------------------------
# Benchmark and simulation metrics
# ---------------------------------------------------------------------------
with st.spinner("Computing the exact classical benchmark..."):
    classical_assets, classical_variance, feasible_count = exact_minimum_variance(
        returns, portfolio_size
    )

sim = results.copy()
if not sim.empty and selected_depth is not None:
    sim = sim[pd.to_numeric(sim["depth"], errors="coerce") == selected_depth].copy()

if not sim.empty:
    sim["probability"] = pd.to_numeric(sim["probability"], errors="coerce")
    sim["daily_variance"] = pd.to_numeric(sim["daily_variance"], errors="coerce")
    sim = sim.dropna(subset=["probability", "daily_variance"])
    sim = sim.sort_values("probability", ascending=False)
    sim["portfolio_display"] = sim["portfolio"].map(portfolio_label)
else:
    sim = pd.DataFrame()

modal = sim.iloc[0] if not sim.empty else None
modal_assets = str(modal["portfolio"]).split("|") if modal is not None else []
modal_variance = float(modal["daily_variance"]) if modal is not None else np.nan
modal_probability = float(modal["probability"]) if modal is not None else np.nan
expected_variance = (
    float(np.dot(sim["probability"], sim["daily_variance"]) / sim["probability"].sum())
    if not sim.empty and sim["probability"].sum() > 0
    else np.nan
)

exact_optimum_probability = 0.0
if not sim.empty and "is_exact_optimum" in sim.columns:
    flag = sim["is_exact_optimum"].astype(str).str.lower().isin(["true", "1"])
    if flag.any():
        exact_optimum_probability = float(sim.loc[flag, "probability"].iloc[0])

variance_gap = (
    (modal_variance / classical_variance - 1) * 100
    if modal is not None and classical_variance != 0
    else np.nan
)

# ---------------------------------------------------------------------------
# Header and KPI cards
# ---------------------------------------------------------------------------
head_l, head_r = st.columns([3, 1])
with head_l:
    st.markdown('<div class="qr-eyebrow">Quantitative research platform</div>', unsafe_allow_html=True)
    st.title("QuantumRisk")
    st.markdown(
        '<div class="qr-subtitle">Portfolio risk analytics · QAOA-style simulation vs. exact classical benchmark</div>',
        unsafe_allow_html=True,
    )
with head_r:
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:right"><span class="qr-pill">● LOCAL DATA CONNECTED</span></div>',
        unsafe_allow_html=True,
    )

st.write("")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Assets in dataset", f"{asset_count}")
k2.metric("Daily observations", f"{len(returns):,}")
k3.metric("Classical minimum variance", format_scientific(classical_variance))
k4.metric("Feasible portfolios", f"{feasible_count:,}")
st.write("")

# ---------------------------------------------------------------------------
# Main comparison section
# ---------------------------------------------------------------------------
col_benchmark, col_sim = st.columns([1.05, 0.95], gap="large")

with col_benchmark:
    st.markdown("## Benchmark portfolios")
    st.markdown('<div class="qr-muted">Equal-weighted portfolios · daily variance</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="qr-panel">
          <span class="qr-pill">EXACT CLASSICAL OPTIMUM</span>
          <div class="qr-portfolio">{' · '.join(classical_assets)}</div>
          <div class="qr-muted">Lowest variance across {feasible_count:,} portfolios for k={portfolio_size}</div>
          <h2 style="color:#5eead4;margin-bottom:0.2rem">{format_scientific(classical_variance)}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if modal is not None:
        st.write("")
        st.markdown(
            f"""
            <div class="qr-panel">
              <span class="qr-pill" style="background:#3b2d12;color:#fbbf24;border-color:#765a1b">MOST PROBABLE SIMULATED OUTCOME</span>
              <div class="qr-portfolio">{' · '.join(modal_assets)}</div>
              <div class="qr-muted">Probability: {modal_probability:.5%}</div>
              <h2 style="color:#fbbf24;margin-bottom:0.2rem">{format_scientific(modal_variance)}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Modal variance gap vs. current classical benchmark", f"{variance_gap:.2f}%")

with col_sim:
    st.markdown("## Simulation snapshot")
    st.markdown('<div class="qr-muted">Metrics from the saved optimizer output</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    s1.metric("Modal probability", f"{modal_probability:.5%}" if modal is not None else "—")
    s2.metric("Exact optimum probability", f"{exact_optimum_probability:.5%}" if modal is not None else "—")
    s3, s4 = st.columns(2)
    s3.metric("Expected daily variance", format_scientific(expected_variance) if np.isfinite(expected_variance) else "—")
    s4.metric("Simulation depth", str(selected_depth) if selected_depth is not None else "Not available")
    st.markdown(
        """
        <div class="qr-panel">
          <div style="font-weight:600;color:#edf4ff;margin-bottom:8px">How to read this</div>
          <div class="qr-muted">The classical result is the minimum found by exhaustive enumeration. The simulation probabilities describe the optimizer's measured distribution, not a forecast of future returns.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Charts: readable, non-log variance scale and risk-probability tradeoff
# ---------------------------------------------------------------------------
if not sim.empty:
    st.markdown("## Probability landscape")
    st.markdown('<div class="qr-muted">Explore which feasible portfolios receive the most probability mass.</div>', unsafe_allow_html=True)
    chart_l, chart_r = st.columns(2, gap="large")

    top = sim.head(top_n).copy()
    top = top.sort_values("probability", ascending=True)
    fig_prob = go.Figure(
        go.Bar(
            x=top["probability"] * 100,
            y=top["portfolio_display"],
            orientation="h",
            marker=dict(color="#2dd4bf"),
            hovertemplate="<b>%{y}</b><br>Probability: %{x:.5f}%<extra></extra>",
        )
    )
    fig_prob.update_layout(
        title=f"Top {len(top)} portfolios by probability",
        xaxis_title="Probability (%)",
        yaxis_title="",
        **PLOT_LAYOUT,
        height=max(460, min(760, 24 * len(top) + 140)),
    )
    with chart_l:
        st.plotly_chart(fig_prob, use_container_width=True)

    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=sim["daily_variance"],
            y=sim["probability"] * 100,
            mode="markers",
            marker=dict(size=7, color="#60a5fa", opacity=0.68),
            text=sim["portfolio_display"],
            hovertemplate="<b>%{text}</b><br>Daily variance: %{x:.4e}<br>Probability: %{y:.5f}%<extra></extra>",
            name="Simulated portfolios",
        )
    )
    fig_scatter.add_vline(
        x=classical_variance,
        line_dash="dash",
        line_color="#5eead4",
        annotation_text="Classical minimum",
        annotation_font_color="#5eead4",
    )
    fig_scatter.update_layout(
        title="Probability vs. daily variance",
        xaxis_title="Daily portfolio variance",
        yaxis_title="Probability (%)",
        **PLOT_LAYOUT,
        height=max(460, min(760, 24 * len(top) + 140)),
    )
    with chart_r:
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("## Portfolio explorer")
    st.markdown('<div class="qr-muted">Ranked by simulated probability. Use the table to compare probability and historical risk.</div>', unsafe_allow_html=True)
    display = sim.head(top_n).copy()
    display["portfolio"] = display["portfolio"].map(portfolio_label)
    columns = [c for c in ["rank_by_probability", "portfolio", "probability", "daily_variance", "is_exact_optimum"] if c in display.columns]
    display = display[columns].rename(
        columns={
            "rank_by_probability": "Probability rank",
            "portfolio": "Portfolio assets",
            "probability": "Probability",
            "daily_variance": "Daily variance",
            "is_exact_optimum": "Exact optimum",
        }
    )
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=min(560, 38 * len(display) + 42),
        column_config={
            "Probability rank": st.column_config.NumberColumn(width="small"),
            "Portfolio assets": st.column_config.TextColumn(width="large"),
            "Probability": st.column_config.NumberColumn(format="%.7f", width="medium"),
            "Daily variance": st.column_config.NumberColumn(format="%.6e", width="medium"),
            "Exact optimum": st.column_config.CheckboxColumn(width="small"),
        },
    )
else:
    st.warning("Simulation output is not available. Run `python src/quantum_optimizer.py` to generate the results CSV.")

st.divider()

# ---------------------------------------------------------------------------
# Asset-level risk and methodology
# ---------------------------------------------------------------------------
st.markdown("## Asset-level risk")
st.markdown('<div class="qr-muted">Annualized historical volatility, calculated from daily returns.</div>', unsafe_allow_html=True)

annual_vol = returns.std(ddof=1) * np.sqrt(252) * 100
annual_mean = returns.mean() * 252 * 100
asset_stats = pd.DataFrame(
    {
        "Asset": asset_names,
        "Annualized volatility (%)": annual_vol.reindex(asset_names).to_numpy(),
        "Annualized mean return (%)": annual_mean.reindex(asset_names).to_numpy(),
    }
).sort_values("Annualized volatility (%)", ascending=False)

fig_vol = go.Figure(
    go.Bar(
        x=asset_stats["Asset"],
        y=asset_stats["Annualized volatility (%)"],
        marker=dict(color="#60a5fa"),
        customdata=asset_stats[["Annualized mean return (%)"]].to_numpy(),
        hovertemplate=(
            "<b>%{x}</b><br>Annualized volatility: %{y:.2f}%"
            "<br>Annualized mean return: %{customdata[0]:.2f}%<extra></extra>"
        ),
    )
)
fig_vol.update_layout(
    title="Historical volatility by asset",
    xaxis_title="Asset",
    yaxis_title="Annualized volatility (%)",
    **PLOT_LAYOUT,
    height=400,
)
st.plotly_chart(fig_vol, use_container_width=True)

with st.expander("Methodology, assumptions and limitations"):
    st.markdown(
        """
        - **Portfolio construction:** equal weights across the selected assets.
        - **Risk metric:** sample covariance-based daily portfolio variance.
        - **Classical benchmark:** exhaustive enumeration for the selected portfolio size.
        - **Simulation:** results loaded from the optimizer CSV; this dashboard does not run a quantum circuit.
        - **No quantum advantage claim:** this prototype does not establish quantum speedup or hardware advantage.
        - **Annualization:** daily standard deviation multiplied by the square root of 252 trading days.
        - **Historical data:** historical risk statistics do not guarantee future performance.
        """
    )

st.markdown(
    '<div style="padding:16px 0 6px;color:#71829c;font-size:.78rem;border-top:1px solid #293750;margin-top:18px">QUANTUMRISK · RESEARCH PROTOTYPE · HISTORICAL DATA, NOT INVESTMENT ADVICE</div>',
    unsafe_allow_html=True,
)
