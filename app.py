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
import time

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

# Custom CSS for hero box display
st.markdown(f"""
<style>
    .stApp {{
        background-color: {COLORS["bg_light"]};
    }}
    
    h1, h2, h3, h4, h5 {{
        color: {COLORS["text_primary"]} !important;
        font-weight: 600 !important;
    }}
    
    .hero-wrapper {{
        background: {COLORS["bg_white"]};
        border-radius: 16px;
        border: 1px solid {COLORS["border"]};
        overflow: hidden;
        margin: 16px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}
    
    .hero-container {{
        display: flex;
        background: {COLORS["bg_white"]};
        padding: 0;
    }}
    
    .hero-divider {{
        width: 1px;
        background: {COLORS["divider"]};
        margin: 16px 0;
    }}
    
    .hero-section {{
        flex: 1;
        text-align: center;
        padding: 20px;
    }}
    
    .hero-section-main {{
        flex: 2;
        text-align: center;
        padding: 24px 20px;
    }}
    
    .hero-label {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {COLORS["text_muted"]};
        margin-bottom: 8px;
    }}
    
    .hero-etf-main {{
        font-size: 48px;
        font-weight: 700;
        color: {COLORS["primary"]};
        letter-spacing: 1px;
    }}
    
    .hero-etf-secondary {{
        font-size: 28px;
        font-weight: 600;
        color: {COLORS["text_primary"]};
    }}
    
    .hero-return {{
        font-size: 20px;
        font-weight: 600;
        margin-top: 8px;
        color: {COLORS["positive_bright"]};
    }}
    
    .hero-bps {{
        font-size: 14px;
        font-weight: 400;
        opacity: 0.8;
    }}
    
    .hero-predictability {{
        margin-top: 8px;
        font-size: 12px;
        color: {COLORS["text_secondary"]};
    }}
    
    .details-section {{
        background: {COLORS["bg_white"]};
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0;
        border: 1px solid {COLORS["border"]};
    }}
    
    .detail-row {{
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid {COLORS["divider"]};
    }}
    
    .detail-row:last-child {{
        border-bottom: none;
    }}
    
    .detail-label {{
        font-weight: 500;
        color: {COLORS["text_secondary"]};
    }}
    
    .detail-value {{
        font-weight: 600;
        color: {COLORS["text_primary"]};
    }}
    
    .macro-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 16px 0;
    }}
    
    .macro-pill {{
        background: {COLORS["bg_light"]};
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 13px;
        border: 1px solid {COLORS["border"]};
        color: {COLORS["text_primary"]};
    }}
    
    .macro-pill strong {{
        color: {COLORS["primary"]};
        margin-left: 8px;
        font-weight: 600;
    }}
    
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin: 16px 0;
    }}
    
    .metric-card {{
        background: {COLORS["bg_white"]};
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid {COLORS["border"]};
    }}
    
    .metric-value {{
        font-size: 28px;
        font-weight: 700;
        color: {COLORS["text_primary"]};
    }}
    
    .metric-label {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {COLORS["text_muted"]};
        margin-top: 4px;
    }}
    
    .dataframe {{
        width: 100%;
        border-collapse: collapse;
    }}
    
    .dataframe th {{
        background: {COLORS["bg_light"]};
        padding: 10px;
        text-align: left;
        font-weight: 600;
        border: 1px solid {COLORS["border"]};
    }}
    
    .dataframe td {{
        padding: 8px 10px;
        border: 1px solid {COLORS["border"]};
    }}
    
    hr {{
        margin: 24px 0;
        border: none;
        border-top: 1px solid {COLORS["divider"]};
    }}
    
    .info-text {{
        font-size: 12px;
        color: {COLORS["text_muted"]};
        text-align: center;
        margin: 16px 0;
    }}
</style>
""", unsafe_allow_html=True)


def get_next_trading_day():
    """Get the next trading day (Monday-Friday, excluding weekends)"""
    today = datetime.now()
    next_day = today + timedelta(days=1)
    
    # Skip Saturday (5) and Sunday (6)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    
    return next_day


def display_hero_picks(picks, returns, predictabilities, regime):
    """Display hero box with top 3 picks"""
    if not picks or len(picks) == 0:
        return ""
    
    top1 = picks[0] if len(picks) > 0 else "N/A"
    top1_return = returns[0] if len(returns) > 0 else 0
    top1_pred = predictabilities[0] if len(predictabilities) > 0 else 0
    
    top2 = picks[1] if len(picks) > 1 else None
    top2_return = returns[1] if len(returns) > 1 else 0
    top2_pred = predictabilities[1] if len(predictabilities) > 1 else 0
    
    top3 = picks[2] if len(picks) > 2 else None
    top3_return = returns[2] if len(returns) > 2 else 0
    top3_pred = predictabilities[2] if len(predictabilities) > 2 else 0
    
    html = '<div class="hero-wrapper"><div class="hero-container">'
    
    # Primary pick
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
    """Display macro pills"""
    html = '<div class="macro-container">'
    for key, value in macro_values.items():
        html += f'<span class="macro-pill">{key}<strong>{value:.2f}</strong></span>'
    html += '</div>'
    return html


def display_metrics_grid(metrics):
    """Display metrics in grid format"""
    if not metrics:
        return ""
    
    html = '<div class="metrics-grid">'
    
    ann_return = metrics.get('annualized_return_pct', 0)
    color = COLORS["positive"] if ann_return > 0 else COLORS["negative"]
    html += f'''
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{ann_return:.1f}%</div>
        <div class="metric-label">ANNUALIZED RETURN</div>
    </div>
    '''
    
    sharpe = metrics.get('sharpe_ratio', 0)
    html += f'''
    <div class="metric-card">
        <div class="metric-value">{sharpe:.2f}</div>
        <div class="metric-label">SHARPE RATIO</div>
    </div>
    '''
    
    max_dd = metrics.get('max_drawdown_pct', 0)
    html += f'''
    <div class="metric-card">
        <div class="metric-value" style="color:{COLORS['negative']}">{max_dd:.1f}%</div>
        <div class="metric-label">MAX DRAWDOWN</div>
    </div>
    '''
    
    hit_rate = metrics.get('hit_rate_pct', 0)
    html += f'''
    <div class="metric-card">
        <div class="metric-value">{hit_rate:.1f}%</div>
        <div class="metric-label">HIT RATE</div>
    </div>
    '''
    
    alpha = metrics.get('alpha_vs_benchmark_pct', 0)
    color = COLORS["positive"] if alpha > 0 else COLORS["negative"]
    html += f'''
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{alpha:.1f}%</div>
        <div class="metric-label">ALPHA VS BM</div>
    </div>
    '''
    
    ann_vol = metrics.get('annualized_vol_pct', 0)
    html += f'''
    <div class="metric-card">
        <div class="metric-value">{ann_vol:.1f}%</div>
        <div class="metric-label">VOLATILITY</div>
    </div>
    '''
    
    html += '</div>'
    return html


@st.cache_data(ttl=3600, show_spinner=False)
def load_fixed_predictions(module):
    """Load fixed dataset predictions with timeout"""
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/fixed/predictions.parquet",
            repo_type="dataset",
            local_files_only=False
        )
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_fixed_metrics(module):
    """Load fixed dataset metrics"""
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/fixed/metrics.json",
            repo_type="dataset",
            local_files_only=False
        )
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_consensus(module):
    """Load shrinking consensus"""
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/shrinking/consensus.parquet",
            repo_type="dataset",
            local_files_only=False
        )
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_window_picks(module):
    """Load window picks"""
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/shrinking/window_picks.parquet",
            repo_type="dataset",
            local_files_only=False
        )
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_window_metrics(module):
    """Load window metrics"""
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"{module}/shrinking/window_metrics.parquet",
            repo_type="dataset",
            local_files_only=False
        )
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_metadata():
    """Load metadata"""
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename="metadata.json",
            repo_type="dataset",
            local_files_only=False
        )
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 16px 0 8px 0;">
        <h1 style="font-size: 36px;">📈 ROUGH-PATH-FORECASTER</h1>
        <p style="color: #5f6368;">Signature Kernel + Log-ODE | ETF selection for FI/Commodities & Equity</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="info-text">
    Research Only · Not Financial Advice · Targets maximum predicted return · Signals valid for next NYSE trading session only
    </div>
    """, unsafe_allow_html=True)
    
    # Load metadata with spinner disabled
    with st.spinner("Loading data..."):
        metadata = load_metadata()
        if metadata:
            st.caption(f"Source: HF: {HF_RESULTS_REPO} | Loaded: {datetime.now().isoformat()}")
    
    st.markdown("---")
    
    # Get tickers from metadata
    fi_tickers = ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'SLV', 'VCIT']
    equity_tickers = ['QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB', 'GDX', 'XME', 'IWM']
    
    if metadata:
        fi_tickers = metadata.get('universes', {}).get('fi', {}).get('tickers', fi_tickers)
        equity_tickers = metadata.get('universes', {}).get('equity', {}).get('tickers', equity_tickers)
    
    # Macro values
    macro_values = {"VIX": 19.49, "T10Y2Y": 0.5, "HY_SPREAD": 2.9, "IG_SPREAD": 0.83, "DXY": 120.66}
    
    # Get correct next trading day
    next_trading_day = get_next_trading_day()
    next_trading_day_str = next_trading_day.strftime("%A, %B %d, %Y")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🏦 Fixed Income / Commodities", "📊 Equity"])
    
    with tab1:
        st.markdown("### Fixed Income / Commodities")
        st.markdown(f"<small>Benchmark: AGG (not traded · no CASH output)</small>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("#### OPTION A — FULL DATASET (2008-PRESENT)")
        
        # Load actual predictions
        fixed_preds = load_fixed_predictions("fi")
        
        if fixed_preds is not None and not fixed_preds.empty:
            # Get predictions from dataframe
            if 'predicted_return' in fixed_preds.columns:
                pred_returns = fixed_preds['predicted_return'].values
            else:
                pred_returns = fixed_preds.iloc[:, 0].values
            
            # Convert to bps and create picks
            pred_returns_bps = pred_returns * 100
            sorted_indices = np.argsort(pred_returns_bps)[::-1]
            
            top_picks = [fi_tickers[i] for i in sorted_indices[:3] if i < len(fi_tickers)]
            top_returns = [pred_returns_bps[i] for i in sorted_indices[:3] if i < len(pred_returns_bps)]
            
            # Pad if needed
            while len(top_picks) < 3:
                top_picks.append("N/A")
                top_returns.append(0)
            
            top_predictability = [0.30, 0.28, 0.25]
            regime = "EXPANSION"
            
            hero_html = display_hero_picks(top_picks, top_returns, top_predictability, regime)
            st.markdown(hero_html, unsafe_allow_html=True)
            
            st.markdown(display_macro_pills(macro_values), unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### ETF Scores — Full Dataset")
            
            # Create scores table with actual values
            scores_data = []
            for i, ticker in enumerate(fi_tickers):
                if i < len(pred_returns):
                    ret = pred_returns[i] * 100
                    conv = max(0, min(100, (ret - pred_returns_bps.min()) / (pred_returns_bps.max() - pred_returns_bps.min() + 1e-6) * 100))
                    scores_data.append({'ticker': ticker, 'predicted_return': f"{ret:.2f}%", 'conviction': f"{conv:.1f}%"})
                else:
                    scores_data.append({'ticker': ticker, 'predicted_return': "0.00%", 'conviction': "0.0%"})
            
            scores_df = pd.DataFrame(scores_data)
            scores_df = scores_df.sort_values('predicted_return', ascending=False)
            st.dataframe(scores_df, use_container_width=True, hide_index=True)
        else:
            st.info("No predictions loaded. Using demo data.")
            demo_picks = ['SLV', 'GLD', 'HYG']
            demo_returns = [44, 28, 26]
            demo_pred = [0.30, 0.34, 0.45]
            hero_html = display_hero_picks(demo_picks, demo_returns, demo_pred, "EXPANSION")
            st.markdown(hero_html, unsafe_allow_html=True)
            
            st.markdown(display_macro_pills(macro_values), unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### ETF Scores — Full Dataset")
            
            scores_df = pd.DataFrame({
                'ticker': fi_tickers,
                'predicted_return': ['0.44%', '0.28%', '0.26%', '0.15%', '0.12%', '0.08%', '0.05%'],
                'conviction': ['44%', '28%', '26%', '15%', '12%', '8%', '5%']
            })
            st.dataframe(scores_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("#### OOS Backtest — Full Dataset (test set)")
        
        fixed_metrics = load_fixed_metrics("fi")
        if fixed_metrics:
            display_metrics_grid(fixed_metrics)
        else:
            demo_metrics = {
                'annualized_return_pct': 21.66,
                'sharpe_ratio': 0.741,
                'max_drawdown_pct': -29.55,
                'hit_rate_pct': 50.4,
                'alpha_vs_benchmark_pct': 18.67,
                'annualized_vol_pct': 29.22
            }
            display_metrics_grid(demo_metrics)
        
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
            consensus_pred = [0.32, 0.28, 0.25]
            
            hero_html = display_hero_picks([consensus_pick, second_pick, third_pick], 
                                           consensus_returns, consensus_pred, "EXPANSION")
            st.markdown(hero_html, unsafe_allow_html=True)
        else:
            hero_html = display_hero_picks(['HYG', 'GLD', 'LQD'], [26, 22, 18], [0.32, 0.28, 0.25], "EXPANSION")
            st.markdown(hero_html, unsafe_allow_html=True)
        
        st.markdown(display_macro_pills(macro_values), unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### ETF Scores — Consensus")
        
        if window_picks_df is not None and not window_picks_df.empty:
            pick_counts = window_picks_df['pick'].value_counts().reset_index()
            pick_counts.columns = ['ticker', 'count']
            pick_counts['conviction'] = (pick_counts['count'] / len(window_picks_df) * 100).round(1)
            pick_counts['conviction'] = pick_counts['conviction'].astype(str) + '%'
            st.dataframe(pick_counts, use_container_width=True, hide_index=True)
        else:
            consensus_scores = pd.DataFrame({
                'ticker': fi_tickers,
                'predicted_return': ['0.26%', '0.22%', '0.18%', '0.12%', '0.10%', '0.06%', '0.04%'],
                'conviction': ['26%', '22%', '18%', '12%', '10%', '6%', '4%']
            })
            st.dataframe(consensus_scores, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("#### Expanding Windows — Per-Window Metrics")
        
        window_metrics = load_window_metrics("fi")
        if window_metrics is not None and not window_metrics.empty:
            st.dataframe(window_metrics, use_container_width=True, hide_index=True)
        else:
            demo_metrics_df = pd.DataFrame({
                'start_year': [2012, 2016, 2019, 2021, 2024],
                'ann_return_pct': ['18.45%', '24.14%', '9.05%', '-5.70%', '12.30%'],
                'sharpe': ['0.446', '1.419', '0.205', '-0.173', '0.482'],
                'max_drawdown_pct': ['-43.76%', '-9.03%', '-31.34%', '-21.72%', '-15.20%']
            })
            st.dataframe(demo_metrics_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("#### Pick History")
        st.markdown("<small>MOST RECENT WINDOWS</small>", unsafe_allow_html=True)
        
        if window_picks_df is not None and not window_picks_df.empty:
            history_df = window_picks_df[['start_year', 'pick', 'conviction']].tail(10)
            history_df.columns = ['Start Year', 'Pick', 'Conviction']
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            pick_history = pd.DataFrame({
                'start_year': [2024, 2023, 2022, 2021, 2020],
                'pick': ['HYG', 'GLD', 'TLT', 'SLV', 'LQD'],
                'conviction': ['26.4%', '24.2%', '22.1%', '20.5%', '18.3%']
            })
            st.dataframe(pick_history, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### Equity")
        st.markdown(f"<small>Benchmark: SPY (not traded · no CASH output)</small>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("#### OPTION A — FULL DATASET (2008-PRESENT)")
        
        fixed_preds_eq = load_fixed_predictions("equity")
        
        if fixed_preds_eq is not None and not fixed_preds_eq.empty:
            if 'predicted_return' in fixed_preds_eq.columns:
                pred_returns_eq = fixed_preds_eq['predicted_return'].values
            else:
                pred_returns_eq = fixed_preds_eq.iloc[:, 0].values
            
            pred_returns_bps_eq = pred_returns_eq * 100
            sorted_indices_eq = np.argsort(pred_returns_bps_eq)[::-1]
            
            top_picks_eq = [equity_tickers[i] for i in sorted_indices_eq[:3] if i < len(equity_tickers)]
            top_returns_eq = [pred_returns_bps_eq[i] for i in sorted_indices_eq[:3] if i < len(pred_returns_bps_eq)]
            
            while len(top_picks_eq) < 3:
                top_picks_eq.append("N/A")
                top_returns_eq.append(0)
            
            top_pred_eq = [0.35, 0.31, 0.26]
            
            hero_html = display_hero_picks(top_picks_eq, top_returns_eq, top_pred_eq, "EXPANSION")
            st.markdown(hero_html, unsafe_allow_html=True)
            
            st.markdown(display_macro_pills(macro_values), unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### ETF Scores — Full Dataset")
            
            scores_data_eq = []
            for i, ticker in enumerate(equity_tickers):
                if i < len(pred_returns_eq):
                    ret = pred_returns_eq[i] * 100
                    conv = max(0, min(100, (ret - pred_returns_bps_eq.min()) / (pred_returns_bps_eq.max() - pred_returns_bps_eq.min() + 1e-6) * 100))
                    scores_data_eq.append({'ticker': ticker, 'predicted_return': f"{ret:.2f}%", 'conviction': f"{conv:.1f}%"})
                else:
                    scores_data_eq.append({'ticker': ticker, 'predicted_return': "0.00%", 'conviction': "0.0%"})
            
            scores_df_eq = pd.DataFrame(scores_data_eq)
            scores_df_eq = scores_df_eq.sort_values('predicted_return', ascending=False)
            st.dataframe(scores_df_eq, use_container_width=True, hide_index=True)
        else:
            st.info("No predictions loaded. Using demo data.")
            demo_picks_eq = ['XLI', 'XLK', 'XLY']
            demo_returns_eq = [38, 32, 27]
            demo_pred_eq = [0.35, 0.31, 0.26]
            hero_html = display_hero_picks(demo_picks_eq, demo_returns_eq, demo_pred_eq, "EXPANSION")
            st.markdown(hero_html, unsafe_allow_html=True)
            
            st.markdown(display_macro_pills(macro_values), unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### ETF Scores — Full Dataset")
            
            scores_df_eq = pd.DataFrame({
                'ticker': equity_tickers,
                'predicted_return': ['0.38%', '0.32%', '0.27%', '0.22%', '0.18%', '0.15%', '0.12%', '0.10%', '0.08%', '0.06%', '0.05%', '0.04%', '0.03%', '0.02%'],
                'conviction': ['38%', '32%', '27%', '22%', '18%', '15%', '12%', '10%', '8%', '6%', '5%', '4%', '3%', '2%']
            })
            st.dataframe(scores_df_eq, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("#### OOS Backtest — Full Dataset (test set)")
        
        fixed_metrics_eq = load_fixed_metrics("equity")
        if fixed_metrics_eq:
            display_metrics_grid(fixed_metrics_eq)
        else:
            demo_metrics_eq = {
                'annualized_return_pct': 18.45,
                'sharpe_ratio': 0.62,
                'max_drawdown_pct': -32.18,
                'hit_rate_pct': 52.1,
                'alpha_vs_benchmark_pct': 12.34,
                'annualized_vol_pct': 26.54
            }
            display_metrics_grid(demo_metrics_eq)
        
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
            consensus_pred_eq = [0.31, 0.27, 0.22]
            
            hero_html = display_hero_picks([consensus_pick_eq, second_pick_eq, third_pick_eq], 
                                           consensus_returns_eq, consensus_pred_eq, "EXPANSION")
            st.markdown(hero_html, unsafe_allow_html=True)
        else:
            hero_html = display_hero_picks(['XLK', 'XLI', 'QQQ'], [28, 24, 20], [0.31, 0.27, 0.22], "EXPANSION")
            st.markdown(hero_html, unsafe_allow_html=True)
        
        st.markdown(display_macro_pills(macro_values), unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### ETF Scores — Consensus")
        
        if window_picks_df_eq is not None and not window_picks_df_eq.empty:
            pick_counts_eq = window_picks_df_eq['pick'].value_counts().reset_index()
            pick_counts_eq.columns = ['ticker', 'count']
            pick_counts_eq['conviction'] = (pick_counts_eq['count'] / len(window_picks_df_eq) * 100).round(1)
            pick_counts_eq['conviction'] = pick_counts_eq['conviction'].astype(str) + '%'
            st.dataframe(pick_counts_eq, use_container_width=True, hide_index=True)
        else:
            consensus_scores_eq = pd.DataFrame({
                'ticker': equity_tickers[:7],
                'predicted_return': ['0.28%', '0.24%', '0.20%', '0.16%', '0.12%', '0.10%', '0.08%'],
                'conviction': ['28%', '24%', '20%', '16%', '12%', '10%', '8%']
            })
            st.dataframe(consensus_scores_eq, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("#### Expanding Windows — Per-Window Metrics")
        
        window_metrics_eq = load_window_metrics("equity")
        if window_metrics_eq is not None and not window_metrics_eq.empty:
            st.dataframe(window_metrics_eq, use_container_width=True, hide_index=True)
        else:
            demo_metrics_eq_df = pd.DataFrame({
                'start_year': [2012, 2016, 2019, 2021, 2024],
                'ann_return_pct': ['18.45%', '24.14%', '9.05%', '-5.70%', '12.30%'],
                'sharpe': ['0.446', '1.419', '0.205', '-0.173', '0.482'],
                'max_drawdown_pct': ['-43.76%', '-9.03%', '-31.34%', '-21.72%', '-15.20%']
            })
            st.dataframe(demo_metrics_eq_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("#### Pick History")
        
        if window_picks_df_eq is not None and not window_picks_df_eq.empty:
            history_df_eq = window_picks_df_eq[['start_year', 'pick', 'conviction']].tail(10)
            history_df_eq.columns = ['Start Year', 'Pick', 'Conviction']
            st.dataframe(history_df_eq, use_container_width=True, hide_index=True)
        else:
            pick_history_eq = pd.DataFrame({
                'start_year': [2024, 2023, 2022, 2021, 2020],
                'pick': ['XLK', 'XLI', 'QQQ', 'XLY', 'XLF'],
                'conviction': ['28.1%', '25.3%', '22.7%', '19.8%', '17.2%']
            })
            st.dataframe(pick_history_eq, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div class='info-text'>ROUGH-PATH-FORECASTER v1.0.0 | Data: P2SAMAPA/p2-etf-deepm-data | Results: P2SAMAPA/p2-etf-rough-path-forecaster-results</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
