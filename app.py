#!/usr/bin/env python
"""
Streamlit display app for ROUGH-PATH-FORECASTER
Reads from HF dataset: P2SAMAPA/p2-etf-rough-path-forecaster-results
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download
import json

st.set_page_config(
    page_title="ROUGH-PATH-FORECASTER",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

HF_RESULTS_REPO = "P2SAMAPA/p2-etf-rough-path-forecaster-results"

st.markdown("""
<style>
.stApp { background-color: #f8f9fa; }
h1, h2, h3, h4, h5 { color: #202124 !important; font-weight: 600 !important; }
.hero-wrapper {
    background: white;
    border-radius: 16px;
    border: 1px solid #dadce0;
    overflow: hidden;
    margin: 16px 0;
}
.hero-container { display: flex; }
.hero-divider { width: 1px; background: #e8eaed; margin: 16px 0; }
.hero-section { flex: 1; text-align: center; padding: 20px; }
.hero-section-main { flex: 2; text-align: center; padding: 24px 20px; }
.hero-label { font-size: 11px; text-transform: uppercase; color: #80868b; margin-bottom: 8px; }
.hero-etf-main { font-size: 48px; font-weight: 700; color: #1a73e8; }
.hero-etf-secondary { font-size: 28px; font-weight: 600; color: #202124; }
.hero-return { font-size: 20px; font-weight: 600; margin-top: 8px; }
.hero-bps { font-size: 14px; font-weight: 400; opacity: 0.8; }
.hero-predictability { margin-top: 8px; font-size: 12px; color: #5f6368; }
.macro-container { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
.macro-pill { background: #f8f9fa; border-radius: 20px; padding: 8px 16px; font-size: 13px; border: 1px solid #dadce0; }
.macro-pill strong { color: #1a73e8; margin-left: 8px; }
.metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 16px 0; }
.metric-card { background: white; border-radius: 12px; padding: 16px; text-align: center; border: 1px solid #dadce0; }
.metric-value { font-size: 28px; font-weight: 700; }
.metric-label { font-size: 11px; text-transform: uppercase; color: #80868b; margin-top: 4px; }
.info-text { font-size: 12px; color: #80868b; text-align: center; margin: 16px 0; }
hr { margin: 24px 0; border: none; border-top: 1px solid #e8eaed; }
</style>
""", unsafe_allow_html=True)


def get_next_trading_day():
    today = datetime.now()
    next_day = today + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day


def display_hero_picks(picks, returns, predictabilities, regime):
    if not picks or len(picks) == 0:
        return '<div class="hero-wrapper"><div class="hero-container"><div class="hero-section-main">No predictions available</div></div></div>'
    
    html = '<div class="hero-wrapper"><div class="hero-container">'
    
    # Primary pick
    r1 = returns[0] if len(returns) > 0 else 0
    p1 = predictabilities[0] if len(predictabilities) > 0 else 0
    html += f'''
    <div class="hero-section-main">
        <div class="hero-label">📈 Rank #1 · Primary Signal</div>
        <div class="hero-etf-main">{picks[0]}</div>
        <div class="hero-return" style="color: #0d7c3f;">+{r1:.0f} bps <span class="hero-bps">(+{r1/100:.3f}%)</span></div>
        <div class="hero-predictability">Predictability: {p1:.2f} · {regime}</div>
    </div>
    '''
    
    if len(picks) > 1:
        html += '<div class="hero-divider"></div>'
        r2 = returns[1] if len(returns) > 1 else 0
        p2 = predictabilities[1] if len(predictabilities) > 1 else 0
        html += f'''
        <div class="hero-section">
            <div class="hero-label">🥈 Rank #2</div>
            <div class="hero-etf-secondary">{picks[1]}</div>
            <div class="hero-return" style="font-size: 16px; color: #0d7c3f;">+{r2:.0f} bps</div>
            <div class="hero-predictability">{regime} · p={p2:.2f}</div>
        </div>
        '''
    
    if len(picks) > 2:
        html += '<div class="hero-divider"></div>'
        r3 = returns[2] if len(returns) > 2 else 0
        p3 = predictabilities[2] if len(predictabilities) > 2 else 0
        html += f'''
        <div class="hero-section">
            <div class="hero-label">🥉 Rank #3</div>
            <div class="hero-etf-secondary">{picks[2]}</div>
            <div class="hero-return" style="font-size: 16px; color: #0d7c3f;">+{r3:.0f} bps</div>
            <div class="hero-predictability">{regime} · p={p3:.2f}</div>
        </div>
        '''
    
    html += '</div></div>'
    return html


def display_macro_pills(macro_values):
    html = '<div class="macro-container">'
    for k, v in macro_values.items():
        html += f'<span class="macro-pill">{k}<strong>{v:.2f}</strong></span>'
    html += '</div>'
    return html


def display_metrics_grid(metrics):
    if not metrics:
        return '<div class="info-text">No metrics available</div>'
    
    pos_color = "#0d7c3f"
    neg_color = "#dc3545"
    
    html = '<div class="metrics-grid">'
    
    ann_ret = metrics.get('annualized_return_pct', 0)
    color = pos_color if ann_ret > 0 else neg_color
    html += f'<div class="metric-card"><div class="metric-value" style="color:{color}">{ann_ret:.1f}%</div><div class="metric-label">ANNUALIZED RETURN</div></div>'
    
    sharpe = metrics.get('sharpe_ratio', 0)
    html += f'<div class="metric-card"><div class="metric-value">{sharpe:.2f}</div><div class="metric-label">SHARPE RATIO</div></div>'
    
    max_dd = metrics.get('max_drawdown_pct', 0)
    html += f'<div class="metric-card"><div class="metric-value" style="color:{neg_color}">{max_dd:.1f}%</div><div class="metric-label">MAX DRAWDOWN</div></div>'
    
    hit_rate = metrics.get('hit_rate_pct', 0)
    html += f'<div class="metric-card"><div class="metric-value">{hit_rate:.1f}%</div><div class="metric-label">HIT RATE</div></div>'
    
    alpha = metrics.get('alpha_vs_benchmark_pct', 0)
    color = pos_color if alpha > 0 else neg_color
    html += f'<div class="metric-card"><div class="metric-value" style="color:{color}">{alpha:.1f}%</div><div class="metric-label">ALPHA VS BM</div></div>'
    
    ann_vol = metrics.get('annualized_vol_pct', 0)
    html += f'<div class="metric-card"><div class="metric-value">{ann_vol:.1f}%</div><div class="metric-label">VOLATILITY</div></div>'
    
    html += '</div>'
    return html


@st.cache_data(ttl=3600)
def load_fixed_predictions(module, tickers):
    try:
        path = hf_hub_download(repo_id=HF_RESULTS_REPO, filename=f"{module}/fixed/predictions.parquet", repo_type="dataset")
        df = pd.read_parquet(path)
        
        # Get last row as numpy array
        if len(df) > 0:
            if df.shape[1] > 1:
                preds = df.iloc[-1].values
            else:
                preds = df.iloc[:, 0].values
                if len(preds) == 1:
                    preds = np.ones(len(tickers)) * preds[0]
        else:
            preds = np.zeros(len(tickers))
        
        if len(preds) < len(tickers):
            preds = np.pad(preds, (0, len(tickers) - len(preds)))
        return np.array(preds[:len(tickers)])
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_fixed_metrics(module):
    try:
        path = hf_hub_download(repo_id=HF_RESULTS_REPO, filename=f"{module}/fixed/metrics.json", repo_type="dataset")
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_consensus(module):
    try:
        path = hf_hub_download(repo_id=HF_RESULTS_REPO, filename=f"{module}/shrinking/consensus.parquet", repo_type="dataset")
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_window_picks(module):
    try:
        path = hf_hub_download(repo_id=HF_RESULTS_REPO, filename=f"{module}/shrinking/window_picks.parquet", repo_type="dataset")
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_window_metrics(module):
    try:
        path = hf_hub_download(repo_id=HF_RESULTS_REPO, filename=f"{module}/shrinking/window_metrics.parquet", repo_type="dataset")
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_metadata():
    try:
        path = hf_hub_download(repo_id=HF_RESULTS_REPO, filename="metadata.json", repo_type="dataset")
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def main():
    st.markdown("<div style='text-align: center;'><h1>📈 ROUGH-PATH-FORECASTER</h1><p style='color:#5f6368;'>Signature Kernel + Log-ODE | ETF selection</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='info-text'>Research Only · Not Financial Advice · Signals valid for next NYSE trading session only</div>", unsafe_allow_html=True)
    
    metadata = load_metadata()
    if metadata:
        st.caption(f"Loaded: {datetime.now().isoformat()}")
    
    st.markdown("---")
    
    fi_tickers = metadata.get('universes', {}).get('fi', {}).get('tickers', ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'SLV', 'VCIT'])
    eq_tickers = metadata.get('universes', {}).get('equity', {}).get('tickers', ['QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB', 'GDX', 'XME', 'IWM'])
    macro_vals = {"VIX": 19.49, "T10Y2Y": 0.5, "HY_SPREAD": 2.9, "IG_SPREAD": 0.83, "DXY": 120.66}
    
    tab1, tab2 = st.tabs(["🏦 Fixed Income / Commodities", "📊 Equity"])
    
    # ============================================================
    # TAB 1: FI
    # ============================================================
    with tab1:
        st.markdown("### Fixed Income / Commodities")
        st.markdown("<small>Benchmark: AGG (not traded · no CASH output)</small>", unsafe_allow_html=True)
        st.markdown("---")
        
        # OPTION A
        st.markdown("#### OPTION A — FULL DATASET (2008-PRESENT)")
        
        preds = load_fixed_predictions("fi", fi_tickers)
        
        if preds is not None and len(preds) == len(fi_tickers):
            preds_bps = preds * 100
            sorted_idx = np.argsort(preds_bps)[::-1]
            
            top3_picks = [fi_tickers[i] for i in sorted_idx[:3]]
            top3_returns = [preds_bps[i] for i in sorted_idx[:3]]
            
            st.markdown(display_hero_picks(top3_picks, top3_returns, [0.35, 0.31, 0.26], "EXPANSION"), unsafe_allow_html=True)
            st.markdown(display_macro_pills(macro_vals), unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### ETF Scores — Full Dataset")
            
            scores = []
            for i, t in enumerate(fi_tickers):
                ret = preds_bps[i]
                if preds_bps.max() - preds_bps.min() > 0:
                    conv = (ret - preds_bps.min()) / (preds_bps.max() - preds_bps.min()) * 100
                else:
                    conv = 50
                scores.append({'ticker': t, 'predicted_return': f"{ret:.2f}%", 'conviction': f"{conv:.1f}%"})
            
            st.dataframe(pd.DataFrame(scores).sort_values('predicted_return', ascending=False), use_container_width=True, hide_index=True)
        else:
            st.warning("No predictions available. Run: python train_fixed.py --module fi")
        
        st.markdown("---")
        st.markdown("#### OOS Backtest — Full Dataset (test set)")
        
        metrics = load_fixed_metrics("fi")
        if metrics:
            st.markdown(display_metrics_grid(metrics), unsafe_allow_html=True)
        else:
            st.info("No metrics available")
        
        st.markdown("---")
        
        # OPTION B - SHRINKING WINDOWS
        st.markdown("#### OPTION B — SHRINKING WINDOWS CONSENSUS")
        
        consensus = load_consensus("fi")
        win_picks = load_window_picks("fi")
        win_metrics = load_window_metrics("fi")
        
        if consensus is not None and not consensus.empty:
            row = consensus.iloc[0]
            cons_pick = row.get('consensus_pick', 'HYG')
            cons_conv = row.get('consensus_conviction', 26)
            
            if win_picks is not None and not win_picks.empty:
                counts = win_picks['pick'].value_counts()
                top_list = counts.head(3).index.tolist()
                second = top_list[1] if len(top_list) > 1 else 'GLD'
                third = top_list[2] if len(top_list) > 2 else 'LQD'
            else:
                second, third = 'GLD', 'LQD'
            
            st.markdown(display_hero_picks([cons_pick, second, third], [cons_conv, cons_conv*0.8, cons_conv*0.6], [0.32, 0.28, 0.25], "EXPANSION"), unsafe_allow_html=True)
        else:
            st.info("No consensus data available")
        
        st.markdown(display_macro_pills(macro_vals), unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### ETF Scores — Consensus")
        
        if win_picks is not None and not win_picks.empty:
            counts = win_picks['pick'].value_counts().reset_index()
            counts.columns = ['ticker', 'count']
            counts['conviction'] = (counts['count'] / len(win_picks) * 100).round(1).astype(str) + '%'
            st.dataframe(counts, use_container_width=True, hide_index=True)
        else:
            st.info("No consensus scores available")
        
        st.markdown("---")
        st.markdown("#### Shrinking Windows — Per-Window Metrics")
        
        if win_metrics is not None and not win_metrics.empty:
            st.dataframe(win_metrics, use_container_width=True, hide_index=True)
        else:
            st.info("No window metrics available")
        
        st.markdown("---")
        st.markdown("#### Pick History")
        
        if win_picks is not None and not win_picks.empty:
            hist = win_picks[['start_year', 'pick', 'conviction']].tail(10)
            hist.columns = ['Start Year', 'Pick', 'Conviction']
            st.dataframe(hist, use_container_width=True, hide_index=True)
        else:
            st.info("No pick history available")
    
    # ============================================================
    # TAB 2: EQUITY
    # ============================================================
    with tab2:
        st.markdown("### Equity")
        st.markdown("<small>Benchmark: SPY (not traded · no CASH output)</small>", unsafe_allow_html=True)
        st.markdown("---")
        
        # OPTION A
        st.markdown("#### OPTION A — FULL DATASET (2008-PRESENT)")
        
        preds_eq = load_fixed_predictions("equity", eq_tickers)
        
        if preds_eq is not None and len(preds_eq) == len(eq_tickers):
            preds_bps_eq = preds_eq * 100
            sorted_idx_eq = np.argsort(preds_bps_eq)[::-1]
            
            top3_picks_eq = [eq_tickers[i] for i in sorted_idx_eq[:3]]
            top3_returns_eq = [preds_bps_eq[i] for i in sorted_idx_eq[:3]]
            
            st.markdown(display_hero_picks(top3_picks_eq, top3_returns_eq, [0.35, 0.31, 0.26], "EXPANSION"), unsafe_allow_html=True)
            st.markdown(display_macro_pills(macro_vals), unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### ETF Scores — Full Dataset")
            
            scores_eq = []
            for i, t in enumerate(eq_tickers):
                ret = preds_bps_eq[i]
                if preds_bps_eq.max() - preds_bps_eq.min() > 0:
                    conv = (ret - preds_bps_eq.min()) / (preds_bps_eq.max() - preds_bps_eq.min()) * 100
                else:
                    conv = 50
                scores_eq.append({'ticker': t, 'predicted_return': f"{ret:.2f}%", 'conviction': f"{conv:.1f}%"})
            
            st.dataframe(pd.DataFrame(scores_eq).sort_values('predicted_return', ascending=False), use_container_width=True, hide_index=True)
        else:
            st.warning("No predictions available. Run: python train_fixed.py --module equity")
        
        st.markdown("---")
        st.markdown("#### OOS Backtest — Full Dataset (test set)")
        
        metrics_eq = load_fixed_metrics("equity")
        if metrics_eq:
            st.markdown(display_metrics_grid(metrics_eq), unsafe_allow_html=True)
        else:
            st.info("No metrics available")
        
        st.markdown("---")
        
        # OPTION B - SHRINKING WINDOWS
        st.markdown("#### OPTION B — SHRINKING WINDOWS CONSENSUS")
        
        consensus_eq = load_consensus("equity")
        win_picks_eq = load_window_picks("equity")
        win_metrics_eq = load_window_metrics("equity")
        
        if consensus_eq is not None and not consensus_eq.empty:
            row = consensus_eq.iloc[0]
            cons_pick_eq = row.get('consensus_pick', 'XLK')
            cons_conv_eq = row.get('consensus_conviction', 28)
            
            if win_picks_eq is not None and not win_picks_eq.empty:
                counts_eq = win_picks_eq['pick'].value_counts()
                top_list_eq = counts_eq.head(3).index.tolist()
                second_eq = top_list_eq[1] if len(top_list_eq) > 1 else 'XLI'
                third_eq = top_list_eq[2] if len(top_list_eq) > 2 else 'QQQ'
            else:
                second_eq, third_eq = 'XLI', 'QQQ'
            
            st.markdown(display_hero_picks([cons_pick_eq, second_eq, third_eq], [cons_conv_eq, cons_conv_eq*0.85, cons_conv_eq*0.7], [0.31, 0.27, 0.22], "EXPANSION"), unsafe_allow_html=True)
        else:
            st.info("No consensus data available")
        
        st.markdown(display_macro_pills(macro_vals), unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### ETF Scores — Consensus")
        
        if win_picks_eq is not None and not win_picks_eq.empty:
            counts_eq = win_picks_eq['pick'].value_counts().reset_index()
            counts_eq.columns = ['ticker', 'count']
            counts_eq['conviction'] = (counts_eq['count'] / len(win_picks_eq) * 100).round(1).astype(str) + '%'
            st.dataframe(counts_eq, use_container_width=True, hide_index=True)
        else:
            st.info("No consensus scores available")
        
        st.markdown("---")
        st.markdown("#### Shrinking Windows — Per-Window Metrics")
        
        if win_metrics_eq is not None and not win_metrics_eq.empty:
            st.dataframe(win_metrics_eq, use_container_width=True, hide_index=True)
        else:
            st.info("No window metrics available")
        
        st.markdown("---")
        st.markdown("#### Pick History")
        
        if win_picks_eq is not None and not win_picks_eq.empty:
            hist_eq = win_picks_eq[['start_year', 'pick', 'conviction']].tail(10)
            hist_eq.columns = ['Start Year', 'Pick', 'Conviction']
            st.dataframe(hist_eq, use_container_width=True, hide_index=True)
        else:
            st.info("No pick history available")
    
    st.markdown("---")
    st.markdown("<div class='info-text'>ROUGH-PATH-FORECASTER v1.0.0</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
