"""
app.py — Monte Carlo Risk Engine Dashboard
==========================================
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from data import (
    fetch_prices,
    compute_log_returns,
    compute_covariance_matrix,
    compute_correlation_matrix,
    annualised_stats,
    validate_covariance_matrix,
)
from risk_engine import (
    simulate_correlated_returns,
    portfolio_returns,
    compute_var_cvar_mc,
    compute_var_cvar_historical,
    run_stress_test,
    crisis_correlation_matrix,
    normal_correlation_matrix,
    build_stress_covariance,
    get_simulation_data,
    regularise_covariance,
    rolling_var_backtest,
    kupiec_pof_test,
    run_full_backtest,
)

# ──────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Monte Carlo Risk Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #060912;
    color: #b8cce0;
}

/* Main content area — target every Streamlit container layer */
.main .block-container,
.main,
section[data-testid="stMain"],
section[data-testid="stMain"] > div,
div[data-testid="stAppViewContainer"],
div[data-testid="stAppViewContainer"] > section,
div[data-testid="stAppViewBlockContainer"],
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"],
.stApp,
.stApp > header,
header[data-testid="stHeader"] {
    background-color: #060912 !important;
    padding-top: 2rem;
}

/* Tab content panels */
div[data-testid="stTabsContent"],
div[role="tabpanel"] {
    background-color: #060912 !important;
}

/* Expanders, info boxes, spinners */
div[data-testid="stExpander"],
div[data-testid="stAlert"],
div[data-testid="stInfo"] {
    background-color: #080d1a !important;
    border-color: #0f1e30 !important;
}

.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: #e2eaf4;
    letter-spacing: -0.5px;
    margin-bottom: 0.15rem;
}
.sub-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #3d5a7a;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #3a6a9a;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    border-bottom: 1px solid #0f1e30;
    padding-bottom: 0.4rem;
    margin: 1.8rem 0 1rem 0;
}

/* Metric cards */
.metric-card {
    background: #080d1a;
    border: 1px solid #0f1e30;
    border-radius: 6px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.6rem;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #3d5a7a;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e05060;
    margin-top: 0.15rem;
}
.metric-value-pass {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #00c47a;
    margin-top: 0.15rem;
}
.metric-value-fail {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #e05060;
    margin-top: 0.15rem;
}
.metric-detail {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #3d5a7a;
    margin-top: 0.5rem;
    line-height: 1.8;
}

/* Sidebar */
div[data-testid="stSidebar"] {
    background-color: #040710;
    border-right: 1px solid #0f1e30;
}
div[data-testid="stSidebar"] .block-container {
    background-color: #040710;
}

/* Tabs — remove emojis handled in Python, style the tab bar */
.stTabs [data-baseweb="tab-list"] {
    background-color: #060912;
    border-bottom: 1px solid #0f1e30;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #3d5a7a;
    background-color: transparent;
    border: none;
    padding: 0.6rem 1.4rem;
}
.stTabs [aria-selected="true"] {
    color: #7ab8d4 !important;
    border-bottom: 2px solid #7ab8d4 !important;
    background-color: transparent !important;
}

/* Inputs */
.stSlider > div > div > div { background: #3a6a9a !important; }

.stSelectbox label, .stSlider label, .stMultiSelect label,
.stTextInput label, .stNumberInput label, .stCheckbox label {
    color: #4a6a8a !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Run button */
.stButton > button {
    background: #0d2040;
    color: #7ab8d4;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    border: 1px solid #1a3a5c;
    border-radius: 3px;
    padding: 0.45rem 1.4rem;
    letter-spacing: 1px;
    font-size: 0.72rem;
    text-transform: uppercase;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #122840;
    border-color: #3a6a9a;
    color: #a8d4e8;
}

/* Clean table styling */
.risk-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    margin-top: 0.5rem;
}
.risk-table th {
    background: #080d1a;
    color: #3d5a7a;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #0f1e30;
    text-align: left;
    font-weight: 400;
}
.risk-table td {
    padding: 0.55rem 1rem;
    border-bottom: 1px solid #0a1525;
    color: #8aaccc;
}
.risk-table tr:hover td { background: #080d1a; }
.risk-table .pass { color: #00c47a; }
.risk-table .fail { color: #e05060; }
.risk-table .num  { color: #b8cce0; font-variant-numeric: tabular-nums; }

hr { border-color: #0f1e30; }

/* Catch-all — force white backgrounds out of any remaining Streamlit wrappers */
div[class*="st-"], div[class*="css-"] {
    background-color: transparent;
}
iframe { background-color: #060912 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Plotly theme
# ──────────────────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor="#060912",
    plot_bgcolor="#080d1a",
    font=dict(family="IBM Plex Mono", color="#8aaccc", size=10),
    xaxis=dict(gridcolor="#0f1e30", zerolinecolor="#0f1e30", linecolor="#0f1e30"),
    yaxis=dict(gridcolor="#0f1e30", zerolinecolor="#0f1e30", linecolor="#0f1e30"),
    margin=dict(l=50, r=30, t=50, b=50),
)

ACCENT = ["#4ab8d4", "#e05060", "#00c47a", "#d4a84b", "#7a6abf", "#d47a40"]
C95, C99 = "#d4a84b", "#e05060"


def layout(**overrides):
    """PLOT_LAYOUT with specific keys overridden — avoids duplicate-kwarg errors."""
    return {**{k: v for k, v in PLOT_LAYOUT.items() if k not in overrides}, **overrides}


def table_html(df: pd.DataFrame, col_classes: dict = None) -> str:
    """Render a DataFrame as a styled HTML table."""
    col_classes = col_classes or {}
    rows = ""
    for _, row in df.iterrows():
        cells = ""
        for col in df.columns:
            cls = col_classes.get(col, "")
            cells += f'<td class="{cls}">{row[col]}</td>'
        rows += f"<tr>{cells}</tr>"
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f'<table class="risk-table"><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>'


# ──────────────────────────────────────────────────────────────────
# Caching
# ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(tickers, start, end):
    prices  = fetch_prices(tickers, start=start, end=end)
    returns = compute_log_returns(prices)
    cov     = compute_covariance_matrix(returns)
    corr    = compute_correlation_matrix(returns)
    stats   = annualised_stats(returns)
    return prices, returns, cov, corr, stats


@st.cache_data(ttl=600, show_spinner=False)
def run_simulation(mu, Sigma, weights, horizon, n_sims, conf_levels, seed):
    return get_simulation_data(
        np.array(mu), np.array(Sigma), np.array(weights),
        horizon, n_sims, conf_levels, seed,
    )


# ──────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="section-header">Portfolio</div>', unsafe_allow_html=True)

    tickers_input = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, GLD, TLT, SPY")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.Timestamp("2018-01-01"))
    with col2:
        end_date = st.date_input("End", value=pd.Timestamp("2024-01-01"))

    st.markdown('<div class="section-header">Weights</div>', unsafe_allow_html=True)
    st.caption("Auto-normalised to sum = 1")

    raw_weights = []
    for t in tickers:
        w = st.slider(t, 0.0, 1.0, 1.0 / len(tickers), 0.01, key=f"w_{t}")
        raw_weights.append(w)

    total   = sum(raw_weights)
    weights = [w / total for w in raw_weights] if total > 0 else [1 / len(tickers)] * len(tickers)

    st.markdown('<div class="section-header">Simulation</div>', unsafe_allow_html=True)

    horizon = st.selectbox("Horizon (days)", [1, 5, 10, 21, 63], index=0,
                           help="1=daily, 10=Basel 2-week, 21=monthly, 63=quarterly")
    n_sims  = st.select_slider("Simulations", [10_000, 50_000, 100_000, 500_000], value=100_000)
    conf_95 = st.checkbox("95% confidence", value=True)
    conf_99 = st.checkbox("99% confidence", value=True)
    seed    = st.number_input("Random seed", value=42, step=1)

    conf_levels = []
    if conf_95: conf_levels.append(0.95)
    if conf_99: conf_levels.append(0.99)
    if not conf_levels: conf_levels = [0.95]

    run_btn = st.button("Run Engine", use_container_width=True)

    st.markdown('<div class="section-header">Stress Test</div>', unsafe_allow_html=True)
    crisis_corr_val = st.slider("Crisis correlation", 0.5, 0.99, 0.80, 0.01)
    calm_corr_val   = st.slider("Calm correlation",   0.0, 0.5,  0.20, 0.01)


# ──────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">Monte Carlo Risk Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">VaR &middot; CVaR &middot; Cholesky Simulation &middot; Stress Testing &middot; Backtesting</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────

with st.spinner("Fetching market data..."):
    try:
        prices, returns, cov_df, corr_df, stats_df = load_market_data(
            tickers, start=str(start_date), end=str(end_date),
        )
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        st.stop()

available = list(cov_df.columns)
if len(available) < len(tickers):
    missing = set(tickers) - set(available)
    st.warning(f"Could not fetch data for: {missing}. Using available tickers.")
    tickers = available
    weights = [1 / len(tickers)] * len(tickers)

N           = len(tickers)
weights     = np.array(weights[:N])
weights     = weights / weights.sum()
mu          = returns[tickers].mean().values
Sigma       = cov_df.loc[tickers, tickers].values
returns_arr = returns[tickers].values

# ──────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "Overview",
    "MC Simulation",
    "VaR & CVaR",
    "Stress Test",
    "Backtesting",
    "Diagnostics",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown('<div class="section-header">Portfolio Composition</div>', unsafe_allow_html=True)

    col_pie, col_stats = st.columns([1, 2])

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=tickers,
            values=weights,
            hole=0.55,
            marker=dict(colors=ACCENT[:N], line=dict(color="#060912", width=2)),
            textfont=dict(family="IBM Plex Mono", size=10, color="#8aaccc"),
        ))
        fig_pie.update_layout(
            **PLOT_LAYOUT,
            title=dict(text="Portfolio Weights", font=dict(size=12, color="#8aaccc")),
            showlegend=True,
            legend=dict(font=dict(family="IBM Plex Mono", size=10, color="#8aaccc")),
            height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="fig_pie")

    with col_stats:
        display_stats = stats_df.loc[tickers].copy()
        display_stats.columns = ["Ann. Return", "Ann. Vol", "Sharpe"]
        display_stats.insert(0, "Weight", weights)
        rows_html = ""
        for tkr, row in display_stats.iterrows():
            ret_col = "pass" if row["Ann. Return"] > 0 else "fail"
            rows_html += (
                f"<tr>"
                f"<td>{tkr}</td>"
                f"<td class='num'>{row['Weight']:.2%}</td>"
                f"<td class='num {ret_col}'>{row['Ann. Return']:.2%}</td>"
                f"<td class='num'>{row['Ann. Vol']:.2%}</td>"
                f"<td class='num'>{row['Sharpe']:.3f}</td>"
                f"</tr>"
            )
        st.markdown(f"""
        <table class="risk-table">
            <thead><tr>
                <th>Ticker</th><th>Weight</th>
                <th>Ann. Return</th><th>Ann. Vol</th><th>Sharpe</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Price History</div>', unsafe_allow_html=True)

    norm_prices = prices[tickers] / prices[tickers].iloc[0] * 100
    fig_price = go.Figure()
    for i, t in enumerate(tickers):
        fig_price.add_trace(go.Scatter(
            x=norm_prices.index, y=norm_prices[t], name=t,
            line=dict(color=ACCENT[i % len(ACCENT)], width=1.4),
        ))
    fig_price.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Normalised Price (base = 100)", font=dict(color="#8aaccc")),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=0.98,
                    font=dict(size=10, color="#8aaccc")),
    )
    st.plotly_chart(fig_price, use_container_width=True, key="fig_price")

    st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)

    corr_vals = corr_df.loc[tickers, tickers].values
    fig_corr = go.Figure(go.Heatmap(
        z=corr_vals, x=tickers, y=tickers,
        colorscale=[[0, "#081828"], [0.5, "#080d1a"], [1, "#2a6a8a"]],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_vals, 2), texttemplate="%{text}",
        textfont=dict(family="IBM Plex Mono", size=11, color="#8aaccc"),
        showscale=True,
    ))
    fig_corr.update_layout(**PLOT_LAYOUT, title="Pairwise Correlation", height=360)
    st.plotly_chart(fig_corr, use_container_width=True, key="fig_corr")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — MC SIMULATION
# ══════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown('<div class="section-header">Cholesky-Correlated Return Simulation</div>',
                unsafe_allow_html=True)

    if run_btn or "sim_data" not in st.session_state:
        with st.spinner(f"Running {n_sims:,} simulations..."):
            sim_data = run_simulation(
                mu.tolist(), Sigma.tolist(), weights.tolist(),
                horizon, n_sims, conf_levels, int(seed),
            )
        st.session_state["sim_data"] = sim_data
    else:
        sim_data = st.session_state["sim_data"]

    sim_port  = sim_data["sim_port_returns"]
    sim_asset = sim_data["sim_asset_returns"]

    col_a, col_b = st.columns(2)

    with col_a:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=sim_port, nbinsx=120,
            marker=dict(color="#2a6a8a", opacity=0.85, line=dict(width=0)),
            name="MC Portfolio",
        ))
        fig_dist.update_layout(
            **PLOT_LAYOUT,
            title=f"MC Portfolio Returns  ({horizon}d horizon, {n_sims:,} sims)",
            xaxis_title="Return", yaxis_title="Count",
            height=340, showlegend=False,
        )
        st.plotly_chart(fig_dist, use_container_width=True, key="fig_dist")

    with col_b:
        sim_vols  = np.std(sim_asset, axis=0) * np.sqrt(252)
        hist_vols = returns[tickers].std().values * np.sqrt(252)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=tickers, y=hist_vols, name="Historical",
                                 marker_color="#d4a84b"))
        fig_vol.add_trace(go.Bar(x=tickers, y=sim_vols, name="Simulated",
                                 marker_color="#2a6a8a", opacity=0.85))
        fig_vol.update_layout(
            **PLOT_LAYOUT,
            barmode="group",
            title="Annual Volatility: Simulated vs Historical",
            yaxis_title="Annualised Vol", height=340,
        )
        st.plotly_chart(fig_vol, use_container_width=True, key="fig_vol")

    if N >= 2:
        st.markdown('<div class="section-header">Simulated Correlation Structure</div>',
                    unsafe_allow_html=True)
        pair_a, pair_b = st.columns(2)
        pairs     = [(0, 1), (0, 2)] if N >= 3 else [(0, 1)]
        cols_pair = [pair_a, pair_b]  if N >= 3 else [pair_a]

        for (i, j), col in zip(pairs, cols_pair):
            with col:
                sample_idx = np.random.choice(len(sim_asset), 5000, replace=False)
                rho_sim  = np.corrcoef(sim_asset[:, i], sim_asset[:, j])[0, 1]
                rho_hist = corr_df.loc[tickers[i], tickers[j]]
                fig_sc = go.Figure(go.Scattergl(
                    x=sim_asset[sample_idx, i], y=sim_asset[sample_idx, j],
                    mode="markers",
                    marker=dict(size=2, color="#2a6a8a", opacity=0.25),
                ))
                fig_sc.update_layout(
                    **PLOT_LAYOUT,
                    title=f"{tickers[i]} vs {tickers[j]}"
                          f"<br><sup>rho_sim={rho_sim:.3f}  rho_hist={rho_hist:.3f}</sup>",
                    xaxis_title=tickers[i], yaxis_title=tickers[j], height=320,
                )
                st.plotly_chart(fig_sc, use_container_width=True, key=f"fig_sc_{i}_{j}")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — VaR & CVaR
# ══════════════════════════════════════════════════════════════════

with tabs[2]:
    if "sim_data" not in st.session_state:
        st.info("Run the engine first (Run Engine in sidebar).")
        st.stop()

    sim_data     = st.session_state["sim_data"]
    sim_port     = sim_data["sim_port_returns"]
    mc_metrics   = sim_data["mc_metrics"]
    hist_metrics = compute_var_cvar_historical(returns_arr, weights, conf_levels)
    all_metrics  = {**mc_metrics, **hist_metrics}

    st.markdown('<div class="section-header">Risk Metrics</div>', unsafe_allow_html=True)

    card_cols = st.columns(len(all_metrics))
    for idx, (name, val) in enumerate(sorted(all_metrics.items())):
        with card_cols[idx]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{name.replace("_", " ")}</div>
                <div class="metric-value">{val*100:.3f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Loss Distribution with VaR / CVaR Overlays</div>',
                unsafe_allow_html=True)

    colours_alpha = {"0.95": C95, "0.99": C99}
    fig_var = go.Figure()
    fig_var.add_trace(go.Histogram(
        x=sim_port, nbinsx=150,
        marker=dict(color="#2a6a8a", opacity=0.6, line=dict(width=0)),
        name="MC Returns",
    ))

    annotation_positions = [0.95, 0.80, 0.65, 0.50]
    ann_idx = 0

    for alpha in conf_levels:
        mc_var  = mc_metrics.get(f"VaR_{alpha:.2f}_MC",  0)
        mc_cvar = mc_metrics.get(f"CVaR_{alpha:.2f}_MC", 0)
        col     = colours_alpha.get(f"{alpha:.2f}", C99)
        label   = f"{int(alpha*100)}%"

        fig_var.add_vline(
            x=-mc_var, line_width=1.5, line_dash="dash", line_color=col,
            annotation_text=f"VaR {label}",
            annotation_position="top",
            annotation=dict(
                yref="paper",
                y=annotation_positions[ann_idx % len(annotation_positions)],
                font=dict(family="IBM Plex Mono", size=9, color=col),
                bgcolor="#060912", borderpad=3,
            ),
        )
        ann_idx += 1

        fig_var.add_vline(
            x=-mc_cvar, line_width=1.5, line_dash="dot", line_color=col,
            annotation_text=f"CVaR {label}",
            annotation_position="top",
            annotation=dict(
                yref="paper",
                y=annotation_positions[ann_idx % len(annotation_positions)],
                font=dict(family="IBM Plex Mono", size=9, color=col),
                bgcolor="#060912", borderpad=3,
            ),
        )
        ann_idx += 1

    fig_var.update_layout(
        **PLOT_LAYOUT,
        title=f"Portfolio Return Distribution — {horizon}d Horizon",
        xaxis_title="Return", yaxis_title="Frequency", height=440,
    )
    st.plotly_chart(fig_var, use_container_width=True, key="fig_var")

    st.markdown('<div class="section-header">MC vs Historical Return Density</div>',
                unsafe_allow_html=True)

    port_hist_rets = returns_arr @ weights
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Histogram(
        x=sim_port, nbinsx=120, histnorm="probability density",
        marker=dict(color="#2a6a8a", opacity=0.55), name="MC (Gaussian)",
    ))
    fig_comp.add_trace(go.Histogram(
        x=port_hist_rets, nbinsx=80, histnorm="probability density",
        marker=dict(color="#d4a84b", opacity=0.55), name="Historical",
    ))
    fig_comp.update_layout(
        **layout(margin=dict(l=50, r=30, t=80, b=50)),
        barmode="overlay",
        title=dict(text="MC vs Historical Return Density", y=0.97, x=0, xanchor="left"),
        xaxis_title="Daily Return", yaxis_title="Probability Density",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, font=dict(size=10)),
    )
    st.plotly_chart(fig_comp, use_container_width=True, key="fig_comp")

    st.markdown('<div class="section-header">Numerical Comparison</div>', unsafe_allow_html=True)

    comp_rows = []
    for alpha in conf_levels:
        comp_rows.append({
            "Confidence":  f"{int(alpha*100)}%",
            "VaR (MC)":    f"{mc_metrics.get(f'VaR_{alpha:.2f}_MC',   0)*100:.4f}%",
            "CVaR (MC)":   f"{mc_metrics.get(f'CVaR_{alpha:.2f}_MC',  0)*100:.4f}%",
            "VaR (Hist)":  f"{hist_metrics.get(f'VaR_{alpha:.2f}_Hist',  0)*100:.4f}%",
            "CVaR (Hist)": f"{hist_metrics.get(f'CVaR_{alpha:.2f}_Hist', 0)*100:.4f}%",
        })
    st.markdown(table_html(pd.DataFrame(comp_rows),
                           col_classes={c: "num" for c in
                                        ["VaR (MC)", "CVaR (MC)", "VaR (Hist)", "CVaR (Hist)"]}),
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — STRESS TEST
# ══════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown('<div class="section-header">Correlation Regime Stress Test</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Stress testing replaces the empirical correlation structure with stylised regimes
    while preserving each asset's historical volatility.
    The stressed covariance is reconstructed as **&Sigma;_stress = D &middot; &rho;_stress &middot; D**
    where D = diag(&sigma;), isolating the correlation channel of risk.
    """)

    hist_corr   = corr_df.loc[tickers, tickers].values
    crisis_corr = crisis_correlation_matrix(N, crisis_corr_val)
    calm_corr   = normal_correlation_matrix(N, calm_corr_val)

    stress_scenarios = {
        "Historical":                          hist_corr,
        f"Crisis  (rho = {crisis_corr_val:.2f})": crisis_corr,
        f"Calm  (rho = {calm_corr_val:.2f})":    calm_corr,
    }

    with st.spinner("Running stress scenarios..."):
        stress_df = run_stress_test(
            mu, Sigma, weights, stress_scenarios,
            horizon=horizon,
            n_simulations=min(n_sims, 200_000),
            confidence_levels=conf_levels,
            random_state=int(seed),
        )

    fig_stress = go.Figure()
    metric_cols = [c for c in stress_df.columns if "VaR" in c or "CVaR" in c]
    bar_colours = ACCENT[:3]

    for i, scenario in enumerate(stress_df.index):
        fig_stress.add_trace(go.Bar(
            name=scenario,
            x=metric_cols,
            y=stress_df.loc[scenario, metric_cols].values * 100,
            marker_color=bar_colours[i % len(bar_colours)],
        ))

    fig_stress.update_layout(
        **layout(margin=dict(l=50, r=30, t=80, b=50)),
        barmode="group",
        title=dict(text="VaR & CVaR by Correlation Regime", y=0.97, x=0, xanchor="left"),
        xaxis_title="Metric", yaxis_title="Value (%)", height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, font=dict(size=10)),
    )
    st.plotly_chart(fig_stress, use_container_width=True, key="fig_stress")

    st.markdown('<div class="section-header">Stressed Correlation Matrices</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for hm_idx, (col, (name, corr_mat)) in enumerate(
        zip([c1, c2, c3], stress_scenarios.items())
    ):
        with col:
            fig_hm = go.Figure(go.Heatmap(
                z=corr_mat, x=tickers[:N], y=tickers[:N],
                colorscale=[[0, "#081828"], [0.5, "#080d1a"], [1, "#2a6a8a"]],
                zmid=0, zmin=-1, zmax=1,
                text=np.round(corr_mat, 2), texttemplate="%{text}",
                textfont=dict(family="IBM Plex Mono", size=9, color="#8aaccc"),
                showscale=False,
            ))
            fig_hm.update_layout(
                **layout(margin=dict(l=30, r=10, t=45, b=30)),
                title=dict(text=name, font=dict(size=10, color="#8aaccc")),
                height=270,
            )
            st.plotly_chart(fig_hm, use_container_width=True, key=f"fig_hm_{hm_idx}")

    st.markdown('<div class="section-header">Stress Test Results</div>', unsafe_allow_html=True)

    stress_display = (stress_df * 100).copy()
    stress_display.index.name = "Scenario"
    stress_display = stress_display.reset_index()
    for c in stress_display.columns[1:]:
        stress_display[c] = stress_display[c].map(lambda x: f"{x:.4f}%")
    st.markdown(table_html(stress_display,
                           col_classes={c: "num" for c in stress_display.columns[1:]}),
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 5 — BACKTESTING
# ══════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown('<div class="section-header">Rolling VaR Backtest</div>', unsafe_allow_html=True)

    st.markdown("""
    For each day **t**, VaR is estimated using only the prior **W** days of data
    (no lookahead), then compared to the actual return on day **t**.
    A **violation** occurs when the actual loss exceeds the forecast VaR.
    The **Kupiec POF test** checks whether the observed violation rate is
    statistically consistent with the theoretical rate (1 &minus; &alpha;).
    """)

    bt_col1, _ = st.columns([1, 3])
    with bt_col1:
        bt_window = st.selectbox(
            "Estimation window (days)", [126, 252, 504], index=1,
            help="Basel III standard = 250 days",
        )

    if len(returns_arr) < bt_window + 20:
        st.warning(f"Need at least {bt_window + 20} days of data. Extend your date range.")
        st.stop()

    with st.spinner(f"Running {bt_window}-day rolling backtest..."):
        bt_df, kupiec_results = run_full_backtest(
            returns_arr, weights, window=bt_window, confidence_levels=conf_levels,
        )

    st.markdown('<div class="section-header">Kupiec POF Test Results</div>', unsafe_allow_html=True)

    kup_cols = st.columns(len(kupiec_results))
    for col, (label, res) in zip(kup_cols, kupiec_results.items()):
        with col:
            val_class = "metric-value-pass" if res["result"] == "PASS" else "metric-value-fail"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="{val_class}">{res['result']}</div>
                <div class="metric-detail">
                    T = {res['T']} days<br>
                    Violations: {res['Violations']}
                    ({res['p_observed']*100:.2f}% observed
                    vs {res['p_theoretical']*100:.1f}% expected)<br>
                    LR = {res['LR_statistic']:.3f} &nbsp;&middot;&nbsp;
                    p-value = {res['p_value']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Actual Returns vs VaR Forecast</div>',
                unsafe_allow_html=True)

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["actual_return"],
        mode="lines", name="Actual Return",
        line=dict(color="#4a6a8a", width=0.8),
        opacity=0.7,
    ))

    bt_colours = {"0.95": C95, "0.99": C99}
    for alpha in conf_levels:
        key   = f"{alpha:.2f}"
        col   = bt_colours.get(key, C99)
        label = f"{int(alpha*100)}%"

        fig_bt.add_trace(go.Scatter(
            x=bt_df.index, y=-bt_df[f"VaR_{key}"],
            mode="lines", name=f"−VaR {label}",
            line=dict(color=col, width=1.4, dash="dash"),
        ))

        mask = bt_df[f"violation_{key}"] == 1
        fig_bt.add_trace(go.Scatter(
            x=bt_df.index[mask], y=bt_df["actual_return"][mask],
            mode="markers", name=f"Violation {label}",
            marker=dict(color=col, size=5, symbol="x",
                        line=dict(width=1.5, color=col)),
        ))

    fig_bt.update_layout(
        **PLOT_LAYOUT,
        title=f"Rolling {bt_window}-day VaR Backtest",
        xaxis_title="Day", yaxis_title="Return", height=460,
        legend=dict(orientation="h", yanchor="bottom", y=0.98, font=dict(size=10)),
    )
    st.plotly_chart(fig_bt, use_container_width=True, key="fig_bt")

    st.markdown('<div class="section-header">Rolling Violation Rate (63-day window)</div>',
                unsafe_allow_html=True)

    fig_vrate = go.Figure()
    for alpha in conf_levels:
        key       = f"{alpha:.2f}"
        col       = bt_colours.get(key, C99)
        label     = f"{int(alpha*100)}%"
        roll_rate = bt_df[f"violation_{key}"].rolling(63).mean() * 100

        fig_vrate.add_trace(go.Scatter(
            x=bt_df.index, y=roll_rate, mode="lines",
            name=f"Observed {label}", line=dict(color=col, width=1.8),
        ))
        fig_vrate.add_hline(
            y=(1 - alpha) * 100, line_dash="dot", line_color=col, line_width=1,
            annotation_text=f"Expected {(1-alpha)*100:.1f}%",
            annotation=dict(
                font=dict(family="IBM Plex Mono", size=9, color=col),
                bgcolor="#060912", borderpad=3,
            ),
        )

    fig_vrate.update_layout(
        **PLOT_LAYOUT,
        title="Rolling 63-day Violation Rate vs Expected",
        xaxis_title="Day", yaxis_title="Violation Rate (%)", height=360,
        legend=dict(orientation="h", yanchor="bottom", y=0.98, font=dict(size=10)),
    )
    st.plotly_chart(fig_vrate, use_container_width=True, key="fig_vrate")

    st.markdown('<div class="section-header">Backtest Summary</div>', unsafe_allow_html=True)

    summary_rows = []
    for alpha in conf_levels:
        key = f"{alpha:.2f}"
        res = kupiec_results[f"alpha={key}"]
        summary_rows.append({
            "Confidence": f"{int(alpha*100)}%",
            "Window":     f"{bt_window}d",
            "Days":       str(res["T"]),
            "Violations": str(res["Violations"]),
            "Obs. Rate":  f"{res['p_observed']*100:.2f}%",
            "Exp. Rate":  f"{res['p_theoretical']*100:.1f}%",
            "LR Stat":    f"{res['LR_statistic']:.3f}",
            "p-value":    f"{res['p_value']:.3f}",
            "Kupiec":     res["result"],
        })

    col_classes = {c: "num" for c in ["Days", "Violations", "Obs. Rate",
                                       "Exp. Rate", "LR Stat", "p-value"]}
    col_classes["Kupiec"] = ""  # will be overridden per-row below

    # Custom render to colour-code Kupiec column
    df_sum = pd.DataFrame(summary_rows)
    rows_html = ""
    for _, row in df_sum.iterrows():
        kup_cls = "pass" if row["Kupiec"] == "PASS" else "fail"
        cells = (
            f"<td>{row['Confidence']}</td>"
            f"<td>{row['Window']}</td>"
            f"<td class='num'>{row['Days']}</td>"
            f"<td class='num'>{row['Violations']}</td>"
            f"<td class='num'>{row['Obs. Rate']}</td>"
            f"<td class='num'>{row['Exp. Rate']}</td>"
            f"<td class='num'>{row['LR Stat']}</td>"
            f"<td class='num'>{row['p-value']}</td>"
            f"<td class='{kup_cls}'>{row['Kupiec']}</td>"
        )
        rows_html += f"<tr>{cells}</tr>"

    headers = "".join(f"<th>{c}</th>" for c in df_sum.columns)
    st.markdown(
        f'<table class="risk-table"><thead><tr>{headers}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# TAB 6 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown('<div class="section-header">Covariance Matrix Diagnostics</div>',
                unsafe_allow_html=True)

    checks = validate_covariance_matrix(Sigma)

    d1, d2, d3, d4 = st.columns(4)
    for dcol, label, val in [
        (d1, "Symmetric",      checks["symmetric"]),
        (d2, "PSD",            checks["psd"]),
        (d3, "No NaN / Inf",   checks["no_nan_inf"]),
    ]:
        dcol.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="{'metric-value-pass' if val else 'metric-value-fail'}">
                {'Pass' if val else 'Fail'}
            </div>
        </div>""", unsafe_allow_html=True)
    d4.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Min Eigenvalue</div>
        <div class="metric-value">{checks['min_eigenvalue']:.2e}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Eigenspectrum (PCA)</div>', unsafe_allow_html=True)

    eigenvalues = np.linalg.eigvalsh(Sigma)[::-1]
    explained   = eigenvalues / eigenvalues.sum() * 100
    cumulative  = np.cumsum(explained)

    fig_eigen = make_subplots(specs=[[{"secondary_y": True}]])
    fig_eigen.add_trace(go.Bar(
        x=[f"PC{i+1}" for i in range(N)], y=explained,
        marker_color="#2a6a8a", name="% Variance",
    ), secondary_y=False)
    fig_eigen.add_trace(go.Scatter(
        x=[f"PC{i+1}" for i in range(N)], y=cumulative,
        mode="lines+markers",
        line=dict(color=C99, width=2), marker=dict(size=5),
        name="Cumulative %",
    ), secondary_y=True)
    fig_eigen.update_layout(**PLOT_LAYOUT, title="Covariance Eigenspectrum", height=360)
    fig_eigen.update_yaxes(title_text="% Variance Explained", secondary_y=False,
                           gridcolor="#0f1e30")
    fig_eigen.update_yaxes(title_text="Cumulative %", secondary_y=True,
                           gridcolor="#0f1e30")
    st.plotly_chart(fig_eigen, use_container_width=True, key="fig_eigen")

    st.markdown('<div class="section-header">Cholesky Factor</div>', unsafe_allow_html=True)

    from risk_engine import cholesky_decompose, regularise_covariance
    L        = cholesky_decompose(regularise_covariance(Sigma))
    residual = np.max(np.abs(L @ L.T - regularise_covariance(Sigma)))

    st.code(
        f"Cholesky factor L  (shape {L.shape})\n"
        f"Max reconstruction error |LLt - Sigma_reg| = {residual:.2e}\n\n"
        + pd.DataFrame(L, index=tickers, columns=tickers).round(6).to_string(),
        language="text",
    )

    st.markdown('<div class="section-header">Rolling Volatility</div>', unsafe_allow_html=True)

    roll_window = st.slider("Rolling window (days)", 10, 126, 21)
    roll_vol    = returns[tickers].rolling(roll_window).std() * np.sqrt(252)

    fig_roll = go.Figure()
    for i, t in enumerate(tickers):
        fig_roll.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol[t], name=t,
            line=dict(color=ACCENT[i % len(ACCENT)], width=1.2),
        ))
    fig_roll.update_layout(
        **PLOT_LAYOUT,
        title=f"{roll_window}-day Rolling Annualised Volatility",
        yaxis_title="Vol", height=340,
        legend=dict(orientation="h", yanchor="bottom", y=0.98, font=dict(size=10)),
    )
    st.plotly_chart(fig_roll, use_container_width=True, key="fig_roll")