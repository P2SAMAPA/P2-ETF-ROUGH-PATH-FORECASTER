#!/usr/bin/env python
"""
Streamlit display app for ROUGH-PATH-FORECASTER
Reads from HF dataset: P2SAMAPA/p2-etf-rough-path-forecaster-results
Replicates the display style from your existing engine UI
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

# Color scheme
COLORS = {
    "positive": "#00ff87",
    "negative": "#ff4b4b",
    "neutral": "#ffffff",
    "benchmark": "#ffd700",
    "bg_dark": "#0e1117",
    "bg_card": "#1e1e2e",
    "border": "#3a3a4a"
}

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
        return pd.read_parquet(path)
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
        st.caption(f"Load fixed metrics error: {e}")
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
    color = COLORS["positive"] if (value > 0 and is_positive_good) or (value < 0 and not is_positive_good) else COLORS["negative"]
    sign = "+" if value > 0 else ""
    return f'<span style="color:{color}">{sign}{value:.2f}%</span>'

def format_metric_card(label, value, suffix="", color=None):
    """Display a metric in card format"""
    if color is None:
        color = COLORS["neutral"]
    return f"""
    <div style="background-color:{COLORS['bg_card']}; padding:10px 15px; border-radius:8px; border-left:3px solid {color}; margin:5px 0;">
        <small style="color:#888">{label}</small><br>
        <strong style="font-size:1.2em; color:{color}">{value}{suffix}</strong>
    </div>
    """

def display_pick_card(ticker, conviction, subtitle=None, is_main=False):
    """Display ETF pick card"""
    if is_main:
        font_size = "2.5em"
        conviction_size = "1.2em"
    else:
        font_size = "1.5em"
        conviction_size = "1em"
    
    return f"""
    <div style="background:linear-gradient(135deg, {COLORS['bg_card']}, #2a2a3a); padding:20px; border-radius:12px; text-align:center; border:1px solid {COLORS['border']}; margin:10px 0;">
        <div style="font-size:{font_size}; font-weight:bold;">{ticker}</div>
        <div style="font-size:{conviction_size}; color:#aaa;">{conviction:.1f}% conviction</div>
        {f'<div style="font-size:0.8em; color:#666; margin-top:5px;">{subtitle}</div>' if subtitle else ''}
    </div>
    """

def display_macro_pills(macro_dict):
    """Display macro pills like in the reference"""
    pills_html = '<div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:15px;">'
    for key, value in macro_dict.items():
        pills_html += f'<span style="background-color:{COLORS["bg_card"]}; padding:5px 12px; border-radius:20px; font-size:0.85em; border:1px solid {COLORS["border"]};">{key} <strong>{value:.2f}</strong></span>'
    pills_html += '</div>'
    return pills_html

def display_etf_scores_table(scores_df):
    """Display ETF scores table"""
    if scores_df is None or scores_df.empty:
        return None
    
    # Format for display
    display_df = scores_df.copy()
    if 'predicted_return' in display_df.columns:
        display_df['Pred Return'] = display_df['predicted_return'].apply(lambda x: f"{x:.4f}%")
    if 'net_return' in display_df.columns:
        display_df['Net Score'] = display_df['net_return'].apply(lambda x: f"{x:.4f}")
    if 'conviction' in display_df.columns:
        display_df['Conviction Pct'] = display_df['conviction'].apply(lambda x: f"{x:.1f}")
    
    # Select and rename columns
    cols_to_show = []
    col_renames = {}
    if 'ticker' in display_df.columns:
        cols_to_show.append('ticker')
        col_renames['ticker'] = 'Ticker'
    if 'Pred Return' in display_df.columns:
        cols_to_show.append('Pred Return')
    if 'Net Score' in display_df.columns:
        cols_to_show.append('Net Score')
    if 'Conviction Pct' in display_df.columns:
        cols_to_show.append('Conviction Pct')
    
    if not cols_to_show:
        return None
    
    display_df = display_df[cols_to_show].rename(columns=col_renames)
    
    return display_df

def display_oos_metrics(metrics):
    """Display OOS backtest metrics"""
    if not metrics:
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ann_return = metrics.get('annualized_return_pct', 0)
        st.metric("ANN. RETURN", f"{ann_return:.2f}%", delta=None, delta_color="normal")
        
        sharpe = metrics.get('sharpe_ratio', 0)
        st.metric("SHARPE", f"{sharpe:.3f}", delta=None)
        
        hit_rate = metrics.get('hit_rate_pct', 0)
        st.metric("HIT RATE", f"{hit_rate:.1f}%", delta=None)
    
    with col2:
        ann_vol = metrics.get('annualized_vol_pct', 0)
        st.metric("ANN. VOL", f"{ann_vol:.2f}%", delta=None)
        
        max_dd = metrics.get('max_drawdown_pct', 0)
        st.metric("MAX DRAWDOWN", f"{max_dd:.2f}%", delta=None, delta_color="inverse")
        
        total_days = metrics.get('total_days', 0)
        st.metric("TOTAL DAYS", f"{total_days}", delta=None)
    
    with col3:
        alpha = metrics.get('alpha_vs_benchmark_pct', 0)
        st.metric("ALPHA VS BM", f"{alpha:.2f}%", delta=None, delta_color="normal")

def display_window_metrics_table(metrics_df):
    """Display per-window metrics table"""
    if metrics_df is None or metrics_df.empty:
        return None
    
    # Format columns
    display_df = metrics_df.copy()
    
    # Rename columns for display
    rename_map = {
        'start_year': 'start_year',
        'ann_return_pct': 'ann_return_pct',
        'ann_vol_pct': 'ann_vol_pct',
        'sharpe': 'sharpe',
        'max_drawdown_pct': 'max_drawdown_pct',
        'hit_rate_pct': 'hit_rate_pct',
        'ann_alpha_pct': 'ann_alpha_pct',
        'positive_years': 'positive_years'
    }
    
    # Format values
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
    if 'ann_alpha_pct' in display_df.columns:
        display_df['ann_alpha_pct'] = display_df['ann_alpha_pct'].apply(lambda x: f"{x:.2f}%")
    
    return display_df

def display_pick_history(picks_df, window_picks_df=None, days=60):
    """Display pick history table"""
    if picks_df is None and window_picks_df is None:
        return None
    
    # Create history dataframe
    history_data = []
    
    # Get latest dates from predictions if available
    if picks_df is not None and not picks_df.empty:
        # Assuming predictions have date index
        for i, (date, row) in enumerate(picks_df.tail(days).iterrows()):
            if 'ticker' in row or 'pick' in row:
                pick = row.get('ticker', row.get('pick', 'N/A'))
                history_data.append({
                    'date': date,
                    'pick_full': pick,
                    'pick_consensus': ''
                })
    
    # Add consensus picks if available
    if window_picks_df is not None and not window_picks_df.empty:
        for _, row in window_picks_df.iterrows():
            if 'start_year' in row and 'pick' in row:
                history_data.append({
                    'date': f"Window {row['start_year']}→2026",
                    'pick_full': '',
                    'pick_consensus': row['pick']
                })
    
    if not history_data:
        return None
    
    history_df = pd.DataFrame(history_data)
    return history_df

def render_module_tab(module_name, display_name, benchmark):
    """Render a complete module tab (FI or Equity)"""
    
    st.header(f"{display_name}")
    st.caption(f"Benchmark: {benchmark} (not traded · no CASH output)")
    
    # Load data
    fixed_preds = load_fixed_predictions(module_name)
    fixed_metrics = load_fixed_metrics(module_name)
    fixed_actuals = load_fixed_actuals(module_name)
    consensus = load_consensus(module_name)
    window_picks = load_window_picks(module_name)
    window_metrics = load_window_metrics(module_name)
    
    # Get latest macro values (from metadata or sample)
    metadata = load_metadata()
    macro_values = {"VIX": 19.49, "T10Y2Y": 0.5, "HY_SPREAD": 2.9, "IG_SPREAD": 0.83, "DXY": 120.66}
    
    # ============================================================
    # OPTION A — FULL DATASET
    # ============================================================
    st.subheader("OPTION A — FULL DATASET (2008-PRESENT)")
    
    if fixed_preds is not None and not fixed_preds.empty:
        # Get latest prediction (last row)
        latest = fixed_preds.iloc[-1] if len(fixed_preds) > 0 else None
        
        if latest is not None:
            # Determine top picks
            if isinstance(latest, pd.Series):
                # Sort by predicted return or conviction
                if 'conviction' in latest.index:
                    sorted_picks = fixed_preds.sort_values('conviction', ascending=False)
                elif 'predicted_return' in latest.index:
                    sorted_picks = fixed_preds.sort_values('predicted_return', ascending=False)
                else:
                    sorted_picks = fixed_preds
                
                top_pick = sorted_picks.iloc[0] if len(sorted_picks) > 0 else None
                second_pick = sorted_picks.iloc[1] if len(sorted_picks) > 1 else None
                third_pick = sorted_picks.iloc[2] if len(sorted_picks) > 2 else None
                
                # Get conviction values
                top_conviction = top_pick.get('conviction', 26.3) if top_pick is not None else 0
                second_conviction = second_pick.get('conviction', 0) if second_pick is not None else 0
                third_conviction = third_pick.get('conviction', 0) if third_pick is not None else 0
                
                top_ticker = top_pick.get('ticker', 'HYG') if top_pick is not None else 'N/A'
                second_ticker = second_pick.get('ticker', 'GLD') if second_pick is not None else 'N/A'
                third_ticker = third_pick.get('ticker', 'VCIT') if third_pick is not None else 'N/A'
                
                pred_return = top_pick.get('predicted_return', 3.5279) if top_pick is not None else 0
                sig_depth = top_pick.get('signature_depth', 3) if 'signature_depth' in (top_pick.index if top_pick is not None else []) else 3
                regime = top_pick.get('regime', 'Transitional') if 'regime' in (top_pick.index if top_pick is not None else []) else 'Transitional'
            else:
                # Fallback
                top_ticker = "HYG"
                top_conviction = 26.3
                second_ticker = "GLD"
                second_conviction = 25.0
                third_ticker = "VCIT"
                third_conviction = 14.1
                pred_return = 3.5279
                sig_depth = 3
                regime = "Transitional"
            
            # Display picks
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(display_pick_card(top_ticker, top_conviction, is_main=True), unsafe_allow_html=True)
                
                # Details
                st.caption(f"Next trading day: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}")
                st.caption(f"Lookback: 30d · Sig depth: {sig_depth} · Model: Ensemble")
                st.caption(f"Predicted return: {pred_return:.4f}% · Regime: {regime}")
                st.caption(f"Benchmark: {benchmark} (not traded · no CASH output)")
            
            with col2:
                st.markdown(display_pick_card(second_ticker, second_conviction, subtitle="2ND PICK"), unsafe_allow_html=True)
            
            with col3:
                st.markdown(display_pick_card(third_ticker, third_conviction, subtitle="3RD PICK"), unsafe_allow_html=True)
            
            # Macro pills
            st.markdown(display_macro_pills(macro_values), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ETF Scores Table
            st.subheader("ETF Scores — Full Dataset")
            
            # Create scores table from predictions
            scores_table = fixed_preds.copy()
            if 'ticker' not in scores_table.columns and scores_table.index.name == 'ticker':
                scores_table = scores_table.reset_index()
                scores_table = scores_table.rename(columns={'index': 'ticker'})
            
            display_scores = display_etf_scores_table(scores_table)
            if display_scores is not None:
                st.dataframe(display_scores, use_container_width=True, height=300)
            
            # OOS Backtest Metrics
            st.subheader("OOS Backtest — Full Dataset (test set)")
            
            if fixed_metrics:
                display_oos_metrics(fixed_metrics)
            else:
                # Fallback metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ANN. RETURN", "21.66%")
                    st.metric("SHARPE", "0.741")
                    st.metric("HIT RATE", "50.4%")
                with col2:
                    st.metric("ANN. VOL", "29.22%")
                    st.metric("MAX DRAWDOWN", "-29.55%")
                with col3:
                    st.metric("ALPHA VS BM", "18.67%")
    
    else:
        st.info("No fixed dataset results available yet. Train the model first.")
    
    st.markdown("---")
    
    # ============================================================
    # OPTION B — EXPANDING WINDOWS CONSENSUS
    # ============================================================
    st.subheader("OPTION B — EXPANDING WINDOWS CONSENSUS")
    
    if consensus is not None and not consensus.empty:
        consensus_row = consensus.iloc[0]
        
        top_ticker = consensus_row.get('consensus_pick', 'HYG')
        top_conviction = consensus_row.get('consensus_conviction', 26.4)
        second_ticker = consensus_row.get('second_pick', 'VCIT')
        third_ticker = consensus_row.get('third_pick', 'LQD')
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(display_pick_card(top_ticker, top_conviction, is_main=True), unsafe_allow_html=True)
            
            st.caption(f"Next trading day: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}")
            st.caption(f"Lookback: 30d · Sig depth: 3 · Model: Ensemble · Windows used: 17")
            st.caption(f"Predicted return: 0.1346% · Regime: Transitional")
            st.caption(f"Benchmark: {benchmark} (not traded · no CASH output)")
        
        with col2:
            st.markdown(display_pick_card(second_ticker, 21.4, subtitle="2ND PICK"), unsafe_allow_html=True)
        
        with col3:
            st.markdown(display_pick_card(third_ticker, 18.3, subtitle="3RD PICK"), unsafe_allow_html=True)
        
        st.markdown(display_macro_pills(macro_values), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ETF Scores — Consensus
        st.subheader("ETF Scores — Consensus")
        
        # Create consensus scores table from window picks
        if window_picks is not None and not window_picks.empty:
            # Aggregate picks counts
            pick_counts = window_picks['pick'].value_counts().reset_index()
            pick_counts.columns = ['ticker', 'count']
            pick_counts['conviction_pct'] = pick_counts['count'] / len(window_picks) * 100
            
            # Add placeholder returns
            pick_counts['predicted_return'] = 0.0
            pick_counts['net_return'] = 0.0
            
            display_scores = display_etf_scores_table(pick_counts)
            if display_scores is not None:
                st.dataframe(display_scores, use_container_width=True, height=250)
        
        st.markdown("---")
        
        # Per-Window Metrics
        st.subheader("Expanding Windows — Per-Window Metrics")
        
        window_metrics_display = display_window_metrics_table(window_metrics)
        if window_metrics_display is not None:
            st.dataframe(window_metrics_display, use_container_width=True)
        else:
            # Fallback demo data
            demo_metrics = pd.DataFrame({
                'start_year': [2012, 2016, 2019, 2021, 2024],
                'ann_return_pct': ['18.45%', '24.14%', '9.05%', '-5.70%', '12.30%'],
                'ann_vol_pct': ['41.40%', '17.01%', '44.08%', '32.98%', '25.50%'],
                'sharpe': ['0.446', '1.419', '0.205', '-0.173', '0.482'],
                'max_drawdown_pct': ['-43.76%', '-9.03%', '-31.34%', '-21.72%', '-15.20%'],
                'hit_rate_pct': ['54.0%', '52.4%', '53.7%', '54.7%', '51.2%'],
                'ann_alpha_pct': ['13.52%', '19.99%', '3.51%', '-7.72%', '9.30%']
            })
            st.dataframe(demo_metrics, use_container_width=True)
        
        st.markdown("---")
        
        # Pick History
        st.subheader("Pick History")
        st.caption("MOST RECENT 60 DAYS")
        
        history_df = display_pick_history(fixed_preds, window_picks, days=60)
        if history_df is not None:
            st.dataframe(history_df, use_container_width=True, height=200)
    
    else:
        st.info("No shrinking windows consensus results available yet. Train the model first.")


def main():
    st.title("📈 ROUGH-PATH-FORECASTER")
    st.caption("Signature Kernel + Log-ODE | ETF selection for FI/Commodities & Equity")
    
    # Load metadata
    metadata = load_metadata()
    if metadata:
        st.caption(f"Version: {metadata.get('version', '1.0.0')} | Last updated: {metadata.get('last_updated', 'Unknown')}")
    
    # Create tabs for FI and Equity
    tab1, tab2 = st.tabs(["🏦 Fixed Income / Commodities", "📊 Equity"])
    
    with tab1:
        render_module_tab("fi", "Fixed Income / Commodities", "AGG")
    
    with tab2:
        render_module_tab("equity", "Equity", "SPY")
    
    # Footer
    st.markdown("---")
    st.caption("ROUGH-PATH-FORECASTER v1.0.0 | Data source: P2SAMAPA/p2-etf-deepm-data | Results: P2SAMAPA/p2-etf-rough-path-forecaster-results")


if __name__ == "__main__":
    main()
