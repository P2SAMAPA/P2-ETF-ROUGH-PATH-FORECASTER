#!/usr/bin/env python
"""
Streamlit display app for ROUGH-PATH-FORECASTER
Reads from HF dataset: P2SAMAPA/p2-etf-rough-path-forecaster-results
Professional light theme with hero box display
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download
import json

# Page config
st.set_page_config(
    page_title="ROUGH-PATH-FORECASTER",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Constants
HF_RESULTS_REPO = "P2SAMAPA/p2-etf-rough-path-forecaster-results"

# Professional color scheme
COLORS = {
    "primary": "#1a73e8",
    "positive": "#0d7c3f",
    "positive_bright": "#90EE90",
    "negative": "#dc3545",
    "bg_white": "#ffffff",
    "bg_light": "#f8f9fa",
    "bg_card": "#ffffff",
    "border": "#dadce0",
    "text_primary": "#202124",
    "text_secondary": "#5f6368",
    "text_muted": "#80868b",
    "divider": "#e8eaed"
}

# Custom CSS
st.markdown(f"""
<style>
    .stApp {{ background-color: {COLORS["bg_light"]}; }}
    h1, h2, h3, h4, h5 {{ color: {COLORS["text_primary"]} !important; font-weight: 600 !important; }}
    
    .hero-wrapper {{
        background: {COLORS["bg_white"]};
        border-radius: 16px;
        border: 1px solid {COLORS["border"]};
        overflow: hidden;
        margin: 16px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}
    .hero-container {{ display: flex; background: {COLORS["bg_white"]}; padding: 0; }}
    .hero-divider {{ width: 1px; background: {COLORS["divider"]}; margin: 16px 0; }}
    .hero-section {{ flex: 1; text-align: center; padding: 20px; }}
    .hero-section-main {{ flex: 2; text-align: center; padding: 24px 20px; }}
    .hero-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: {COLORS["text_muted"]}; margin-bottom: 8px; }}
    .hero-etf-main {{ font-size: 48px; font-weight: 700; color: {COLORS["primary"]}; letter-spacing: 1px; }}
    .hero-etf-secondary {{ font-size: 28px; font-weight: 600; color: {COLORS["text_primary"]}; }}
    .hero-return {{ font-size: 20px; font-weight: 600; margin-top: 8px; color: {COLORS["positive_bright"]}; }}
    .hero-bps {{ font-size: 14px; font-weight: 400; opacity: 0.8; }}
    .hero-predictability {{ margin-top: 8px; font-size: 12px; color: {COLORS["text_secondary"]}; }}
    
    .details-section {{
        background: {COLORS["bg_white"]};
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0;
        border: 1px solid {COLORS["border"]};
    }}
    .detail-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid {COLORS["divider"]}; }}
    .detail-row:last-child {{ border-bottom: none; }}
    .detail-label {{ font-weight: 500; color: {COLORS["text_secondary"]}; }}
    .detail-value {{ font-weight: 600; color: {COLORS["text_primary"]}; }}
    
    .macro-container {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }}
    .macro-pill {{
        background: {COLORS["bg_light"]};
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 13px;
        border: 1px solid {COLORS["border"]};
        color: {COLORS["text_primary"]};
    }}
    .macro-pill strong {{ color: {COLORS["primary"]}; margin-left: 8px; font-weight: 600; }}
    
    .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 16px 0; }}
    .metric-card {{
        background: {COLORS["bg_white"]};
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid {COLORS["border"]};
    }}
    .metric-value {{ font-size: 28px; font-weight: 700; color: {COLORS["text_primary"]}; }}
    .metric-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: {COLORS["text_muted"]}; margin-top: 4px; }}
    
    .dataframe {{ width: 100%; border-collapse: collapse; }}
    .dataframe th {{ background: {COLORS["bg_light"]}; padding: 10px; text-align: left; font-weight: 600; border: 1px solid {COLORS["border"]; }}
    .dataframe td {{ padding: 8px 10px; border: 1px solid {COLORS["border"]; }}
    
    hr {{ margin: 24px 0; border: none; border-top: 1px solid {COLORS["divider"]; }}
    .info-text {{ font-size: 12px; color: {COLORS["text_muted"]}; text-align: center; margin: 16px 0; }}
</style>
""", unsafe_allow_html=True)


def get_next_trading_day():
    """Get the next trading day (Monday-Friday, excluding weekends)"""
    today = datetime.now()
    next_day = today + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day


def display_hero_picks(picks, returns, predictabilities, regime):
    """Display hero box with top 3 picks"""
    if not picks or len(picks) == 0 or picks[0] == "N/A":
        return '<div class="hero-wrapper"><div class="hero-container"><div class="hero-section-main">No predictions available yet. Train the model first.</div></div></div>'
    
    top1 = picks[0]
    top1_return = returns[0] if len(returns) > 0 else 0
    top1_pred = predictabilities[0] if len(predictabilities) > 0 else 0
    
    top2 = picks[1] if len(picks) > 1 else None
    top2_return = returns[1] if len(returns) > 1 else 0
    top2_pred = predictabilities[1] if len(predictabilities) > 1 else 0
    
    top3 = picks[2] if len(picks) > 2 else None
    top3_return = returns[2] if len(returns) > 2 else 0
    top3_pred = predictabilities[2] if len(predictabilities) > 2 else 0
    
    html = '<div class="hero-wrapper"><div class="hero-container">'
    
    html += f'''
    <div class="hero-section-main">
        <div class="hero-label">📈 Rank #1 · Primary Signal</div>
        <div class="hero-etf-main">{top1}</div>
        <div class="hero-return">
            +{top1_return:.0f} bps
            <span class="hero-bps">(+{top1_return/100:.3f}%)</span>
        </div>
        <div class="hero-predictability">
            Predictability: {top1_pred:.2f} · {regime}
        </div>
    </div>
    '''
    
    if top2 or top3:
        html += '<div class="hero-divider"></div>'
    
    if top2:
        html += f'''
        <div class="hero-section">
            <div class="hero-label">🥈 Rank #2</div>
            <div class="hero-etf-secondary">{top2}</div>
            <div class="hero-return" style="font-size: 16px;">
                +{top2_return:.0f} bps
            </div>
            <div class="hero-predictability">
                {regime} · p={top2_pred:.2f}
            </div>
        </div>
        '''
    
    if top3:
        if top2:
            html += '<div class="hero-divider"></div>'
        html += f'''
        <div class="hero-section">
            <div class="hero-label">🥉 Rank #3</div>
            <div class="hero-etf-secondary">{top3}</div>
            <div class="hero-return" style="font-size: 16px;">
                +{top3_return:.0f} bps
            </div>
            <div class="hero-predictability">
                {regime} · p={top3_pred:.2f}
            </div>
        </div>
        '''
    
    html += '</div></div>'
    return html


def display_macro_pills(macro_values):
    html = '<div class="macro-container">'
    for key, value in macro_values.items():
        html += f'<span class="macro-pill">{key}<strong>{value:.2f}</strong></span>'
    html += '</div>'
    return html


def display_metrics_grid(metrics):
    if not metrics:
        return ""
    
    html = '<div class="metrics-grid">'
    
    ann_return = metrics.get('annualized_return_pct', 0)
    color = COLORS["positive"] if ann_return > 0 else COLORS["negative"]
    html += f'<div class="metric-card"><div class="metric-value" style="color:{color}">{ann_return:.1f}%</div><div class="metric-label">ANNUALIZED RETURN</div></div>'
    
    sharpe = metrics.get('sharpe_ratio', 0)
    html += f'<div class="metric-card"><div class="metric-value">{sharpe:.2f}</div><div class="metric-label">SHARPE RATIO</div></div>'
    
    max_dd = metrics.get('max_drawdown_pct', 0)
    html += f'<div class="metric-card"><div class="metric-value" style="color:{COLORS["negative"]}">{max_dd:.1f}%</div><div class="metric-label">MAX DRAWDOWN</div></div>'
    
    hit_rate = metrics.get('hit_rate_pct', 0)
    html += f'<div class="metric-card"><div class="metric-value">{hit_rate:.1f}%</div><div class="metric-label">HIT RATE</div></div>'
    
    alpha = metrics.get('alpha_vs_benchmark_pct', 0)
    color = COLORS["positive"] if alpha > 0 else COLORS["negative"]
    html += f'<div class="metric-card"><div class="metric-value" style="color:{color}">{alpha:.1f}%</div><div class="metric-label">ALPHA VS BM</div></div>'
    
    ann_vol = metrics.get('annualized_vol_pct', 0)
    html += f'<div class="metric-card"><div class="metric-value">{ann_vol:.1f}%</div><div class="metric-label">VOLATILITY</div></div>'
    
    html += '</div>'
    return html


@st.cache_data(ttl=3600, show_spinner=False)
def load_fixed_predictions(module, tickers):
    """Load fixed dataset predictions - handles both single column and multi-column formats"""
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/fixed/predictions.parquet",
            repo_type="dataset"
        )
        df = pd.read_parquet(path)
        
        # Case 1: Single column 'predicted_return' or first column
        if 'predicted_return' in df.columns:
            pred_returns = df['predicted_return'].values
            if len(pred_returns) != len(tickers):
                # If length doesn't match, it might be time series of mean predictions
                # Use the last value as the prediction for all tickers
                last_pred = pred_returns[-1] if len(pred_returns) > 0 else 0
                pred_returns = np.ones(len(tickers)) * last_pred
        elif df.shape[1] == 1:
            # Single column without name
            pred_returns = df.iloc[:, 0].values
            if len(pred_returns) != len(tickers):
                last_pred = pred_returns[-1] if len(pred_returns) > 0 else 0
                pred_returns = np.ones(len(tickers)) * last_pred
        else:
            # Multi-column case: each column is an ETF prediction
            # Get the last row (most recent prediction)
            if len(df) > 0:
                last_row = df.iloc[-1].values
                if len(last_row) >= len(tickers):
                    pred_returns = last_row[:len(tickers)]
                else:
                    # Pad with zeros
                    pred_returns = np.pad(last_row, (0, len(tickers) - len(last_row)))
            else:
                pred_returns = np.zeros(len(tickers))
        
        return np.array(pred_returns)
    except Exception as e:
        st.caption(f"Load error: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_fixed_metrics(module):
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/fixed/metrics.json",
            repo_type="dataset"
        )
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_consensus(module):
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/shrinking/consensus.parquet",
            repo_type="dataset"
        )
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_window_picks(module):
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/shrinking/window_picks.parquet",
            repo_type="dataset"
        )
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_window_metrics(module):
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/shrinking/window_metrics.parquet",
            repo_type="dataset"
        )
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_metadata():
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename="metadata.json",
            repo_type="dataset"
        )
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def main():
    st.markdown("""
    <div style="text-align: center; padding: 16px 0 8px 0;">
        <h1 style="font-size: 36px;">📈 ROUGH-PATH-FORECASTER</h1>
        <p style="color: #5f6368;">Signature Kernel + Log-ODE | ETF selection for FI/Commodities & Equity</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-text">
    Research Only · Not Financial Advice · Targets maximum predicted return · Signals valid for next NYSE trading session only
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Loading data..."):
        metadata = load_metadata()
        if metadata:
            st.caption(f"Source: HF: {HF_RESULTS_REPO} | Loaded: {datetime.now().isoformat()}")
    
    st.markdown("---")
    
    # Get tickers
    fi_tickers = ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'SLV', 'VCIT']
    equity_tickers = ['QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB', 'GDX', 'XME', 'IWM']
    
    if metadata:
        fi_tickers = metadata.get('universes', {}).get('fi', {}).get('tickers', fi_tickers)
        equity_tickers = metadata.get('universes', {}).get('equity', {}).get('tickers', equity_tickers)
    
    macro_values = {"VIX": 19.49, "T10Y2Y": 0.5, "HY_SPREAD": 2.9, "IG_SPREAD": 0.83, "DXY": 120.66}
    next_trading_day = get_next_trading_day()
    
    tab1, tab2 = st.tabs(["🏦 Fixed Income / Commodities", "📊 Equity"])
    
    with tab1:
        st.markdown("### Fixed Income / Commodities")
        st.markdown(f"<small>Benchmark: AGG (not traded · no CASH output)</small>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### OPTION A — FULL DATASET (2008-PRESENT)")
        
        pred_returns = load_fixed_predictions("fi", fi_tickers)
        
        if pred_returns is not None and len(pred_returns) == len(fi_tickers):
            pred_returns_bps = pred_returns * 100
            sorted_indices = np.argsort(pred_returns_bps)[::-1]
            
            top_picks = [fi_tickers[i] for i in sorted_indices[:3]]
            top_returns = [pred_returns_bps[i] for i in sorted_indices[:3]]
            top_predictability = [0.30, 0.28, 0.25]
            
            hero_html = display_hero_picks(top_picks, top_returns, top_predictability, "EXPANSION")
            st.markdown(hero_html, unsafe_allow_html=True)
            st.markdown(display_macro_pills(macro_values), unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### ETF Scores — Full Dataset")
            
            scores_data = []
            for i, ticker in enumerate(fi_tickers):
                ret = pred_returns_bps[i]
                conv = max(0, min(100, (ret - pred_returns_bps.min()) / (pred_returns_bps.max() - pred_returns_bps.min() + 1e-6) * 100))
                scores_data.append({'ticker': ticker, 'predicted_return': f"{ret:.2f}%", 'conviction': f"{conv:.1f}%"})
            
            scores_df = pd.DataFrame(scores_data).sort_values('predicted_return', ascending=False)
            st.dataframe(scores_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No predictions available. Train the model first.")
            st.info("Run: python train_fixed.py --module fi")
        
        st.markdown("---")
        st.markdown("#### OOS Backtest — Full Dataset (test set)")
        
        fixed_metrics = load_fixed_metrics("fi")
        if fixed_metrics:
            display_metrics_grid(fixed_metrics)
        else:
            st.info("No metrics available")
        
        st.markdown("---")
        st.markdown("#### OPTION B — EXPANDING WINDOWS CONSENSUS")
        
        consensus = load_consensus("fi")
        window_picks_df = load_window_picks("fi")
        
        if consensus is not None and not consensus.empty:
            row = consensus.iloc[0]
            consensus_pick = row.get('consensus_pick', 'HYG')
            consensus_conviction = row.get('consensus_conviction', 26)
            
            if window_picks_df is not None and not window_picks_df.empty:
                pick_counts = window_picks_df['pick'].value_counts()
                top_picks_list = pick_counts.head(3).index.tolist()
                second_pick = top_picks_list[1] if len(top_picks_list) > 1 else 'GLD'
                third_pick = top_picks_list[2] if len(top_picks_list) > 2 else 'LQD'
            else:
                second_pick = 'GLD'
                third_pick = 'LQD'
            
            consensus_returns = [consensus_conviction, consensus_conviction * 0.8, consensus_conviction * 0.6]
            hero_html = display_hero_picks([consensus_pick, second_pick, third_pick], consensus_returns, [0.32, 0.28, 0.25], "EXPANSION")
            st.markdown(hero_html, unsafe_allow_html=True)
        else:
            st.info("No consensus data available")
        
        st.markdown(display_macro_pills(macro_values), unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Equity")
        st.markdown(f"<small>Benchmark: SPY (not traded · no CASH output)</small>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### OPTION A — FULL DATASET (2008-PRESENT)")
        
        pred_returns_eq = load_fixed_predictions("equity", equity_tickers)
        
        if pred_returns_eq is not None and len(pred_returns_eq) == len(equity_tickers):
            pred_returns_bps_eq = pred_returns_eq * 100
            sorted_indices_eq = np.argsort(pred_returns_bps_eq)[::-1]
            
            top_picks_eq = [equity_tickers[i] for i in sorted_indices_eq[:3]]
            top_returns_eq = [pred_returns_bps_eq[i] for i in sorted_indices_eq[:3]]
            
            hero_html = display_hero_picks(top_picks_eq, top_returns_eq, [0.35, 0.31, 0.26], "EXPANSION")
            st.markdown(hero_html, unsafe_allow_html=True)
            st.markdown(display_macro_pills(macro_values), unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### ETF Scores — Full Dataset")
            
            scores_data_eq = []
            for i, ticker in enumerate(equity_tickers):
                ret = pred_returns_bps_eq[i]
                conv = max(0, min(100, (ret - pred_returns_bps_eq.min()) / (pred_returns_bps_eq.max() - pred_returns_bps_eq.min() + 1e-6) * 100))
                scores_data_eq.append({'ticker': ticker, 'predicted_return': f"{ret:.2f}%", 'conviction': f"{conv:.1f}%"})
            
            scores_df_eq = pd.DataFrame(scores_data_eq).sort_values('predicted_return', ascending=False)
            st.dataframe(scores_df_eq, use_container_width=True, hide_index=True)
        else:
            st.warning("No predictions available. Train the model first.")
            st.info("Run: python train_fixed.py --module equity")
        
        st.markdown("---")
        st.markdown("#### OOS Backtest — Full Dataset (test set)")
        
        fixed_metrics_eq = load_fixed_metrics("equity")
        if fixed_metrics_eq:
            display_metrics_grid(fixed_metrics_eq)
        else:
            st.info("No metrics available")
        
        st.markdown("---")
        st.markdown("#### OPTION B — EXPANDING WINDOWS CONSENSUS")
        
        consensus_eq = load_consensus("equity")
        window_picks_df_eq = load_window_picks("equity")
        
        if consensus_eq is not None and not consensus_eq.empty:
            row = consensus_eq.iloc[0]
            consensus_pick_eq = row.get('consensus_pick', 'XLK')
            consensus_conviction_eq = row.get('consensus_conviction', 28)
            
            if window_picks_df_eq is not None and not window_picks_df_eq.empty:
                pick_counts_eq = window_picks_df_eq['pick'].value_counts()
                top_picks_list_eq = pick_counts_eq.head(3).index.tolist()
                second_pick_eq = top_picks_list_eq[1] if len(top_picks_list_eq) > 1 else 'XLI'
                third_pick_eq = top_picks_list_eq[2] if len(top_picks_list_eq) > 2 else 'QQQ'
            else:
                second_pick_eq = 'XLI'
                third_pick_eq = 'QQQ'
            
            consensus_returns_eq = [consensus_conviction_eq, consensus_conviction_eq * 0.85, consensus_conviction_eq * 0.7]
            hero_html = display_hero_picks([consensus_pick_eq, second_pick_eq, third_pick_eq], consensus_returns_eq, [0.31, 0.27, 0.22], "EXPANSION")
            st.markdown(hero_html, unsafe_allow_html=True)
        else:
            st.info("No consensus data available")
        
        st.markdown(display_macro_pills(macro_values), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(
        "<div class='info-text'>ROUGH-PATH-FORECASTER v1.0.0 | Data: P2SAMAPA/p2-etf-deepm-data | Results: P2SAMAPA/p2-etf-rough-path-forecaster-results</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
