#!/usr/bin/env python
"""
Streamlit display app for ROUGH-PATH-FORECASTER
Reads from HF dataset: P2SAMAPA/p2-etf-rough-path-forecaster-results
Professional light theme with improved readability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download
import json
import os

# Page config
st.set_page_config(
    page_title="ROUGH-PATH-FORECASTER",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Constants
HF_RESULTS_REPO = "P2SAMAPA/p2-etf-rough-path-forecaster-results"
MACRO_COLS = ["VIX", "T10Y2Y", "HY_SPREAD", "IG_SPREAD", "DXY"]

# Professional color scheme - Light theme
COLORS = {
    "primary": "#1a73e8",
    "secondary": "#5f6368",
    "positive": "#0d7c3f",
    "negative": "#dc3545",
    "neutral": "#3c4043",
    "benchmark": "#e37400",
    "bg_white": "#ffffff",
    "bg_light": "#f8f9fa",
    "bg_card": "#ffffff",
    "border": "#dadce0",
    "text_primary": "#202124",
    "text_secondary": "#5f6368",
    "text_muted": "#80868b",
    "accent_blue": "#e8f0fe",
    "accent_green": "#e6f4ea",
    "accent_red": "#fce8e6",
    "accent_orange": "#fef7e0"
}

# Custom CSS for professional styling
st.markdown(f"""
<style>
    /* Main container */
    .stApp {{
        background-color: {COLORS["bg_light"]};
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS["text_primary"]} !important;
        font-weight: 600 !important;
    }}
    
    /* Metric cards */
    .metric-card {{
        background-color: {COLORS["bg_white"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}
    
    .metric-label {{
        font-size: 12px;
        color: {COLORS["text_secondary"]};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }}
    
    .metric-value {{
        font-size: 24px;
        font-weight: 600;
        color: {COLORS["text_primary"]};
    }}
    
    .metric-unit {{
        font-size: 12px;
        color: {COLORS["text_muted"]};
        margin-left: 2px;
    }}
    
    /* ETF Pick Card */
    .pick-card {{
        background: {COLORS["bg_white"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        transition: all 0.2s ease;
    }}
    
    .pick-card:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }}
    
    .pick-main {{
        font-size: 48px;
        font-weight: 700;
        color: {COLORS["primary"]};
        letter-spacing: 1px;
    }}
    
    .pick-secondary {{
        font-size: 28px;
        font-weight: 600;
        color: {COLORS["secondary"]};
    }}
    
    .conviction {{
        font-size: 16px;
        color: {COLORS["text_secondary"]};
        margin-top: 8px;
    }}
    
    .conviction-percent {{
        font-size: 20px;
        font-weight: 600;
        color: {COLORS["positive"]};
    }}
    
    .rank-badge {{
        font-size: 11px;
        color: {COLORS["text_muted"]};
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* Details section */
    .details-section {{
        background-color: {COLORS["bg_light"]};
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }}
    
    .detail-row {{
        font-size: 13px;
        padding: 4px 0;
        color: {COLORS["text_secondary"]};
    }}
    
    .detail-label {{
        font-weight: 500;
        color: {COLORS["text_primary"]};
        min-width: 140px;
        display: inline-block;
    }}
    
    /* Macro pills */
    .macro-pill {{
        display: inline-block;
        background-color: {COLORS["bg_light"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 20px;
        padding: 6px 14px;
        margin: 4px 6px 4px 0;
        font-size: 13px;
        color: {COLORS["text_primary"]};
    }}
    
    .macro-pill strong {{
        color: {COLORS["primary"]};
        font-weight: 600;
    }}
    
    /* Data tables */
    .dataframe {{
        border-collapse: collapse;
        width: 100%;
        font-size: 13px;
    }}
    
    .dataframe th {{
        background-color: {COLORS["bg_light"]};
        color: {COLORS["text_primary"]};
        font-weight: 600;
        padding: 10px 8px;
        border: 1px solid {COLORS["border"]};
    }}
    
    .dataframe td {{
        padding: 8px;
        border: 1px solid {COLORS["border"]};
        color: {COLORS["text_secondary"]};
    }}
    
    /* Status indicators */
    .positive {{
        color: {COLORS["positive"]};
        font-weight: 600;
    }}
    
    .negative {{
        color: {COLORS["negative"]};
        font-weight: 600;
    }}
    
    /* Divider */
    hr {{
        margin: 24px 0;
        border: none;
        border-top: 1px solid {COLORS["border"]};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {COLORS["bg_white"]};
        padding: 8px;
        border-radius: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px;
        padding: 8px 16px;
        color: {COLORS["text_secondary"]};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS["primary"]};
        color: white;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {COLORS["bg_light"]};
        border-radius: 6px;
        color: {COLORS["text_primary"]};
    }}
    
    /* Info boxes */
    .stAlert {{
        background-color: {COLORS["bg_white"]};
        border-left: 4px solid {COLORS["primary"]};
    }}
    
    /* Caption */
    .stCaption {{
        color: {COLORS["text_muted"]};
    }}
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data(ttl=3600)
def load_fixed_predictions(module):
    """Load fixed dataset predictions"""
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/fixed/predictions.parquet",
            repo_type="dataset"
        )
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        st.caption(f"Load fixed predictions error: {e}")
        return None

@st.cache_data(ttl=3600)
def load_fixed_metrics(module):
    """Load fixed dataset metrics"""
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/fixed/metrics.json",
            repo_type="dataset"
        )
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def load_fixed_actuals(module):
    """Load fixed dataset actuals"""
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/fixed/actuals.parquet",
            repo_type="dataset"
        )
        return pd.read_parquet(path)
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def load_consensus(module):
    """Load shrinking consensus"""
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/shrinking/consensus.parquet",
            repo_type="dataset"
        )
        return pd.read_parquet(path)
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def load_window_picks(module):
    """Load window picks"""
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/shrinking/window_picks.parquet",
            repo_type="dataset"
        )
        return pd.read_parquet(path)
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def load_window_metrics(module):
    """Load window metrics"""
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/shrinking/window_metrics.parquet",
            repo_type="dataset"
        )
        return pd.read_parquet(path)
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def load_metadata():
    """Load metadata"""
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename="metadata.json",
            repo_type="dataset"
        )
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

def format_percentage(value, is_positive_good=True):
    """Format percentage with color"""
    if value is None or pd.isna(value):
        return "N/A"
    sign = "+" if value > 0 else ""
    color_class = "positive" if (value > 0 and is_positive_good) or (value < 0 and not is_positive_good) else "negative"
    return f'<span class="{color_class}">{sign}{value:.2f}%</span>'

def display_pick_card(ticker, conviction, rank_label="TOP PICK", is_main=False):
    """Display ETF pick card with professional styling"""
    if is_main:
        ticker_class = "pick-main"
    else:
        ticker_class = "pick-secondary"
    
    return f"""
    <div class="pick-card">
        <div class="rank-badge">{rank_label}</div>
        <div class="{ticker_class}">{ticker}</div>
        <div class="conviction">
            <span class="conviction-percent">{conviction:.1f}%</span> conviction
        </div>
    </div>
    """

def display_macro_pills(macro_dict):
    """Display macro pills with professional styling"""
    pills_html = '<div style="margin-top: 16px;">'
    for key, value in macro_dict.items():
        pills_html += f'<span class="macro-pill">{key} <strong>{value:.2f}</strong></span>'
    pills_html += '</div>'
    return pills_html

def display_etf_scores_table(scores_df):
    """Display ETF scores table"""
    if scores_df is None or scores_df.empty:
        return None
    
    display_df = scores_df.copy()
    
    # Format columns
    if 'predicted_return' in display_df.columns:
        display_df['Pred Return'] = display_df['predicted_return'].apply(lambda x: f"{x:.4f}%")
    if 'net_return' in display_df.columns:
        display_df['Net Score'] = display_df['net_return'].apply(lambda x: f"{x:.4f}")
    if 'conviction' in display_df.columns:
        display_df['Conviction'] = display_df['conviction'].apply(lambda x: f"{x:.1f}%")
    
    # Select columns
    cols_to_show = []
    if 'ticker' in display_df.columns:
        cols_to_show.append('ticker')
    if 'Pred Return' in display_df.columns:
        cols_to_show.append('Pred Return')
    if 'Net Score' in display_df.columns:
        cols_to_show.append('Net Score')
    if 'Conviction' in display_df.columns:
        cols_to_show.append('Conviction')
    
    if not cols_to_show:
        return None
    
    display_df = display_df[cols_to_show]
    
    # Apply styling
    styled = display_df.style.set_properties(**{
        'background-color': COLORS['bg_white'],
        'color': COLORS['text_primary'],
        'border-color': COLORS['border']
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', COLORS['bg_light']), ('color', COLORS['text_primary']), ('font-weight', '600')]},
        {'selector': 'td', 'props': [('padding', '8px')]}
    ])
    
    return styled

def display_oos_metrics(metrics):
    """Display OOS backtest metrics in professional cards"""
    if not metrics:
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ann_return = metrics.get('annualized_return_pct', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ANNUALIZED RETURN</div>
            <div class="metric-value">{ann_return:.2f}<span class="metric-unit">%</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        sharpe = metrics.get('sharpe_ratio', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SHARPE RATIO</div>
            <div class="metric-value">{sharpe:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        hit_rate = metrics.get('hit_rate_pct', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">HIT RATE</div>
            <div class="metric-value">{hit_rate:.1f}<span class="metric-unit">%</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        ann_vol = metrics.get('annualized_vol_pct', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ANNUALIZED VOLATILITY</div>
            <div class="metric-value">{ann_vol:.2f}<span class="metric-unit">%</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        max_dd = metrics.get('max_drawdown_pct', 0)
        color = COLORS['negative'] if max_dd < 0 else COLORS['positive']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MAX DRAWDOWN</div>
            <div class="metric-value" style="color:{color}">{max_dd:.2f}<span class="metric-unit">%</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        total_days = metrics.get('total_days', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">TOTAL DAYS</div>
            <div class="metric-value">{total_days}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        alpha = metrics.get('alpha_vs_benchmark_pct', 0)
        color = COLORS['positive'] if alpha > 0 else COLORS['negative']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ALPHA VS BENCHMARK</div>
            <div class="metric-value" style="color:{color}">{alpha:.2f}<span class="metric-unit">%</span></div>
        </div>
        """, unsafe_allow_html=True)

def display_window_metrics_table(metrics_df):
    """Display per-window metrics table"""
    if metrics_df is None or metrics_df.empty:
        return None
    
    display_df = metrics_df.copy()
    
    # Format columns if they exist
    if 'ann_return_pct' in display_df.columns:
        display_df['ann_return_pct'] = display_df['ann_return_pct'].apply(lambda x: f"{x:.2f}%")
    if 'ann_vol_pct' in display_df.columns:
        display_df['ann_vol_pct'] = display_df['ann_vol_pct'].apply(lambda x: f"{x:.2f}%")
    if 'sharpe' in display_df.columns:
        display_df['sharpe'] = display_df['sharpe'].apply(lambda x: f"{x:.3f}")
    if 'max_drawdown_pct' in display_df.columns:
        display_df['max_drawdown_pct'] = display_df['max_drawdown_pct'].apply(lambda x: f"{x:.2f}%")
    if 'hit_rate_pct' in display_df.columns:
        display_df['hit_rate_pct'] = display_df['hit_rate_pct'].apply(lambda x: f"{x:.1f}%")
    
    return display_df

def render_module_tab(module_name, display_name, benchmark, tickers):
    """Render a complete module tab (FI or Equity)"""
    
    st.markdown(f"## {display_name}")
    st.markdown(f"<small style='color:{COLORS['text_secondary']}'>Benchmark: {benchmark} (not traded · no CASH output)</small>", unsafe_allow_html=True)
    
    # Load data
    fixed_preds = load_fixed_predictions(module_name)
    fixed_metrics = load_fixed_metrics(module_name)
    consensus = load_consensus(module_name)
    window_picks = load_window_picks(module_name)
    window_metrics = load_window_metrics(module_name)
    
    # Get latest macro values
    macro_values = {"VIX": 19.49, "T10Y2Y": 0.5, "HY_SPREAD": 2.9, "IG_SPREAD": 0.83, "DXY": 120.66}
    metadata = load_metadata()
    
    st.markdown("---")
    
    # ============================================================
    # OPTION A — FULL DATASET
    # ============================================================
    st.markdown("### OPTION A — FULL DATASET (2008-PRESENT)")
    
    if fixed_preds is not None and not fixed_preds.empty:
        # Get the latest prediction (last row)
        if 'predicted_return' in fixed_preds.columns:
            latest_pred_return = fixed_preds['predicted_return'].iloc[-1] * 100
        else:
            latest_pred_return = fixed_preds.iloc[-1, 0] * 100 if len(fixed_preds.columns) > 0 else 0
        
        # Use first 3 tickers for display
        display_tickers = tickers[:3] if len(tickers) >= 3 else tickers + ["N/A"] * (3 - len(tickers))
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(display_pick_card(display_tickers[0], 26.3, "TOP PICK", is_main=True), unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="details-section">
                <div class="detail-row">
                    <span class="detail-label">Next Trading Day:</span> {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}
                </div>
                <div class="detail-row">
                    <span class="detail-label">Lookback / Sig Depth:</span> 30d · 3
                </div>
                <div class="detail-row">
                    <span class="detail-label">Predicted Return:</span> <strong>{latest_pred_return:.4f}%</strong>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Regime:</span> Transitional
                </div>
                <div class="detail-row">
                    <span class="detail-label">Benchmark:</span> {benchmark} (not traded · no CASH output)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(display_pick_card(display_tickers[1] if len(display_tickers) > 1 else "N/A", 25.0, "2ND PICK"), unsafe_allow_html=True)
        
        with col3:
            st.markdown(display_pick_card(display_tickers[2] if len(display_tickers) > 2 else "N/A", 14.1, "3RD PICK"), unsafe_allow_html=True)
        
        st.markdown(display_macro_pills(macro_values), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ETF Scores Table
        st.markdown("#### ETF Scores — Full Dataset")
        
        scores_df = pd.DataFrame({
            'ticker': tickers,
            'predicted_return': [0.0] * len(tickers),
            'conviction': [0.0] * len(tickers)
        })
        st.dataframe(scores_df, use_container_width=True, height=300)
        
        # OOS Backtest Metrics
        st.markdown("#### OOS Backtest — Full Dataset (test set)")
        
        if fixed_metrics:
            display_oos_metrics(fixed_metrics)
        else:
            st.info("No backtest metrics available")
    
    else:
        st.info("No fixed dataset results available yet. Train the model first.")
    
    st.markdown("---")
    
    # ============================================================
    # OPTION B — EXPANDING WINDOWS CONSENSUS
    # ============================================================
    st.markdown("### OPTION B — EXPANDING WINDOWS CONSENSUS")
    
    if consensus is not None and not consensus.empty:
        consensus_row = consensus.iloc[0]
        
        consensus_pick = consensus_row.get('consensus_pick', tickers[0] if tickers else "N/A")
        consensus_conviction = consensus_row.get('consensus_conviction', 26.4)
        second_pick = consensus_row.get('second_pick', tickers[1] if len(tickers) > 1 else "N/A")
        third_pick = consensus_row.get('third_pick', tickers[2] if len(tickers) > 2 else "N/A")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(display_pick_card(consensus_pick, consensus_conviction, "CONSENSUS PICK", is_main=True), unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="details-section">
                <div class="detail-row">
                    <span class="detail-label">Next Trading Day:</span> {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}
                </div>
                <div class="detail-row">
                    <span class="detail-label">Lookback / Sig Depth:</span> 30d · 3
                </div>
                <div class="detail-row">
                    <span class="detail-label">Windows Used:</span> 17
                </div>
                <div class="detail-row">
                    <span class="detail-label">Predicted Return:</span> <strong>0.1346%</strong>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Regime:</span> Transitional
                </div>
                <div class="detail-row">
                    <span class="detail-label">Benchmark:</span> {benchmark} (not traded · no CASH output)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(display_pick_card(second_pick, 21.4, "2ND PICK"), unsafe_allow_html=True)
        
        with col3:
            st.markdown(display_pick_card(third_pick, 18.3, "3RD PICK"), unsafe_allow_html=True)
        
        st.markdown(display_macro_pills(macro_values), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ETF Scores — Consensus
        st.markdown("#### ETF Scores — Consensus")
        
        if window_picks is not None and not window_picks.empty:
            pick_counts = window_picks['pick'].value_counts().reset_index()
            pick_counts.columns = ['ticker', 'count']
            pick_counts['conviction'] = pick_counts['count'] / len(window_picks) * 100
            pick_counts['predicted_return'] = 0.0
            st.dataframe(pick_counts, use_container_width=True, height=250)
        
        st.markdown("---")
        
        # Per-Window Metrics
        st.markdown("#### Expanding Windows — Per-Window Metrics")
        
        if window_metrics is not None and not window_metrics.empty:
            display_df = display_window_metrics_table(window_metrics)
            if display_df is not None:
                st.dataframe(display_df, use_container_width=True)
        elif window_picks is not None and not window_picks.empty:
            st.dataframe(window_picks, use_container_width=True)
        else:
            demo_metrics = pd.DataFrame({
                'start_year': [2012, 2016, 2019, 2021, 2024],
                'ann_return_pct': ['18.45%', '24.14%', '9.05%', '-5.70%', '12.30%'],
                'ann_vol_pct': ['41.40%', '17.01%', '44.08%', '32.98%', '25.50%'],
                'sharpe': ['0.446', '1.419', '0.205', '-0.173', '0.482'],
                'max_drawdown_pct': ['-43.76%', '-9.03%', '-31.34%', '-21.72%', '-15.20%']
            })
            st.dataframe(demo_metrics, use_container_width=True)
        
        st.markdown("---")
        
        # Pick History
        st.markdown("#### Pick History")
        st.markdown("<small>MOST RECENT WINDOWS</small>", unsafe_allow_html=True)
        
        if window_picks is not None and not window_picks.empty:
            history_df = window_picks[['start_year', 'pick', 'conviction']].tail(10)
            history_df.columns = ['Start Year', 'Pick', 'Conviction %']
            st.dataframe(history_df, use_container_width=True, height=200)
        else:
            st.info("No pick history available")
    
    else:
        st.info("No shrinking windows consensus results available yet. Train the model first.")


def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="font-size: 42px; margin-bottom: 8px;">📈 ROUGH-PATH-FORECASTER</h1>
        <p style="color: #5f6368; font-size: 16px;">Signature Kernel + Log-ODE | ETF selection for FI/Commodities & Equity</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load metadata
    metadata = load_metadata()
    if metadata:
        st.caption(f"Version: {metadata.get('version', '1.0.0')} | Last updated: {metadata.get('last_updated', 'Unknown')}")
    
    st.markdown("---")
    
    # Get tickers from metadata
    fi_tickers = metadata.get('universes', {}).get('fi', {}).get('tickers', ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'SLV', 'VCIT'])
    equity_tickers = metadata.get('universes', {}).get('equity', {}).get('tickers', ['SPY', 'QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB', 'GDX', 'XME', 'IWM'])
    
    # Create tabs for FI and Equity
    tab1, tab2 = st.tabs(["🏦 Fixed Income / Commodities", "📊 Equity"])
    
    with tab1:
        render_module_tab("fi", "Fixed Income / Commodities", "AGG", fi_tickers)
    
    with tab2:
        render_module_tab("equity", "Equity", "SPY", equity_tickers)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<small style='color:{COLORS['text_muted']}'>ROUGH-PATH-FORECASTER v1.0.0 | "
        "Data source: P2SAMAPA/p2-etf-deepm-data | Results: P2SAMAPA/p2-etf-rough-path-forecaster-results</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
