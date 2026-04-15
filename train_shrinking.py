#!/usr/bin/env python
"""
Train ROUGH-PATH-FORECASTER on shrinking windows (2008→2026 through 2024→2026)
Usage: python train_shrinking.py --module [fi|equity]
"""

import argparse
import pickle
import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

from constants import SHRINKING_START_YEARS, SHRINKING_END_YEAR
from module_fi import FIModule
from module_equity import EquityModule
from outputs import ParquetWriter
from selection import ETFSelector
from utils import Logger, Timer, GitHubActionsHelpers


def compute_max_drawdown(returns_series):
    """
    Compute maximum drawdown for a series of returns
    Returns a positive percentage value (e.g., 25.0 for 25% drawdown)
    """
    if len(returns_series) == 0:
        return 0.0
    
    # Calculate cumulative returns
    cumulative = (1 + returns_series).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Return the maximum drawdown as a positive percentage
    max_dd = abs(drawdown.min()) if not pd.isna(drawdown.min()) else 0.0
    
    return max_dd


def compute_per_window_metrics(actuals_series, start_year):
    """
    Compute all performance metrics for a single window
    actuals_series: pandas Series of returns with datetime index
    """
    if len(actuals_series) < 5:
        return {
            'start_year': start_year,
            'n_days': len(actuals_series),
            'ann_return_pct': 0.0,
            'ann_vol_pct': 0.0,
            'sharpe': 0.0,
            'max_drawdown_pct': 0.0,
            'hit_rate_pct': 0.0,
            'ann_alpha_pct': 0.0,
            'positive_years': 0
        }
    
    # Annualized return (assuming 252 trading days)
    ann_return = actuals_series.mean() * 252
    
    # Annualized volatility
    ann_vol = actuals_series.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    
    # Maximum drawdown - this is the key fix
    max_dd = compute_max_drawdown(actuals_series)
    
    # Hit rate
    hit_rate = (actuals_series > 0).mean()
    
    # Alpha (excess return over 3% risk-free rate)
    ann_alpha = ann_return - 0.03
    
    # Count positive years
    if len(actuals_series) > 0 and hasattr(actuals_series.index, 'year'):
        yearly_returns = actuals_series.groupby(actuals_series.index.year).mean()
        positive_years = (yearly_returns > 0).sum()
    else:
        positive_years = 0
    
    return {
        'start_year': start_year,
        'n_days': len(actuals_series),
        'ann_return_pct': ann_return * 100,
        'ann_vol_pct': ann_vol * 100,
        'sharpe': sharpe,
        'max_drawdown_pct': max_dd * 100,
        'hit_rate_pct': hit_rate * 100,
        'ann_alpha_pct': ann_alpha * 100,
        'positive_years': positive_years
    }


def compute_consensus_picks(window_results, module, start_years):
    """
    Compute consensus picks across all windows using weighted scoring
    """
    tickers = module.tickers
    benchmark = module.benchmark
    
    window_picks = []
    window_convictions = []
    window_metrics = []
    
    for i, result in enumerate(window_results):
        start_year = start_years[i]
        
        predictions = result['predictions']
        actuals = result['actuals']
        dates = result.get('dates', None)
        
        if len(predictions) == 0 or len(actuals) == 0:
            continue
        
        # Get most recent prediction for this window
        if len(predictions.shape) > 1:
            last_pred = predictions[-1] if len(predictions) > 0 else predictions.mean(axis=0)
        else:
            last_pred = predictions[-1] if len(predictions) > 0 else predictions
        
        # Ensure 1D array
        if isinstance(last_pred, (int, float)):
            last_pred = np.ones(len(tickers)) * last_pred
        elif len(last_pred.shape) > 1:
            last_pred = last_pred.flatten()
        
        # Trim or pad
        if len(last_pred) > len(tickers):
            last_pred = last_pred[:len(tickers)]
        elif len(last_pred) < len(tickers):
            last_pred = np.pad(last_pred, (0, len(tickers) - len(last_pred)))
        
        # Get top pick for this window
        selector = ETFSelector(tickers, benchmark)
        scores = selector.compute_net_scores(last_pred[:len(tickers)])
        top_pick = scores.iloc[0]['ticker']
        top_conviction = scores.iloc[0]['conviction']
        
        window_picks.append(top_pick)
        window_convictions.append(top_conviction)
        
        # Compute metrics using actual returns for this window's test set
        if len(actuals) > 0:
            # Convert actuals to pandas Series
            if isinstance(actuals, np.ndarray):
                if len(actuals.shape) > 1:
                    # Use mean return across ETFs for strategy performance
                    actuals_series = pd.Series(actuals.mean(axis=1))
                else:
                    actuals_series = pd.Series(actuals)
            else:
                actuals_series = pd.Series(actuals)
            
            # Add datetime index if dates available
            if dates is not None and len(dates) == len(actuals_series):
                actuals_series.index = pd.DatetimeIndex(dates)
            
            # Calculate metrics for this specific window
            metrics = compute_per_window_metrics(actuals_series, start_year)
            window_metrics.append(metrics)
    
    # Build consensus
    weighted_picks = {}
    for pick, conv in zip(window_picks, window_convictions):
        weighted_picks[pick] = weighted_picks.get(pick, 0) + conv
    
    sorted_picks = sorted(weighted_picks.items(), key=lambda x: x[1], reverse=True)
    
    window_metrics_df = pd.DataFrame(window_metrics) if window_metrics else pd.DataFrame()
    
    consensus = {
        'consensus_pick': sorted_picks[0][0] if sorted_picks else None,
        'consensus_conviction': sorted_picks[0][1] / len(window_picks) if sorted_picks and len(window_picks) > 0 else 0,
        'second_pick': sorted_picks[1][0] if len(sorted_picks) > 1 else None,
        'third_pick': sorted_picks[2][0] if len(sorted_picks) > 2 else None,
        'window_picks': window_picks,
        'window_convictions': window_convictions,
        'window_metrics': window_metrics_df,
        'pick_counts': dict(Counter(window_picks))
    }
    
    return consensus


def main():
    parser = argparse.ArgumentParser(description="Train shrinking windows model")
    parser.add_argument("--module", type=str, required=True, choices=['fi', 'equity'])
    args = parser.parse_args()
    
    logger = Logger(f"Train-Shrinking-{args.module.upper()}")
    logger.info(f"Training {len(SHRINKING_START_YEARS)} shrinking windows")
    
    is_ci = GitHubActionsHelpers.is_github_actions()
    timer = Timer()
    timer.__enter__()
    
    try:
        if args.module == 'fi':
            module = FIModule()
            result = module.train_shrinking(SHRINKING_START_YEARS, SHRINKING_END_YEAR)
            save_dir = "models_saved/fi/shrinking"
        else:
            module = EquityModule()
            result = module.train_shrinking(SHRINKING_START_YEARS, SHRINKING_END_YEAR)
            save_dir = "models_saved/equity/shrinking"
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Compute consensus with proper per-window metrics
        consensus = compute_consensus_picks(result['windows'], module, SHRINKING_START_YEARS)
        
        # Save models
        for start_year, model in result['models'].items():
            model_path = os.path.join(save_dir, f"model_window_{start_year}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save window results
        window_results_list = []
        for w in result['windows']:
            window_results_list.append({
                'start_year': w['start_year'],
                'end_year': w['end_year'],
                'n_days': w['n_days']
            })
        window_results_df = pd.DataFrame(window_results_list)
        ParquetWriter.write_window_results(window_results_df, os.path.join(save_dir, "window_results.parquet"))
        
        # Save consensus
        consensus_df = pd.DataFrame([{
            'consensus_pick': consensus['consensus_pick'],
            'consensus_conviction': consensus['consensus_conviction'],
            'second_pick': consensus['second_pick'],
            'third_pick': consensus['third_pick']
        }])
        ParquetWriter.write_predictions(consensus_df, os.path.join(save_dir, "consensus.parquet"))
        
        # Save window picks
        if len(consensus['window_picks']) > 0:
            window_picks_df = pd.DataFrame({
                'start_year': SHRINKING_START_YEARS[:len(consensus['window_picks'])],
                'pick': consensus['window_picks'],
                'conviction': consensus['window_convictions']
            })
            ParquetWriter.write_predictions(window_picks_df, os.path.join(save_dir, "window_picks.parquet"))
        
        # Save window metrics (now each window has its own unique values)
        if not consensus['window_metrics'].empty:
            ParquetWriter.write_predictions(consensus['window_metrics'], os.path.join(save_dir, "window_metrics.parquet"))
        
        timer.__exit__(None, None, None)
        logger.info(f"Shrinking windows training completed in {timer.minutes:.2f} minutes")
        
        if is_ci:
            GitHubActionsHelpers.set_output("training_status", "success")
            GitHubActionsHelpers.set_output("consensus_pick", consensus['consensus_pick'])
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        timer.__exit__(None, None, None)
        if is_ci:
            GitHubActionsHelpers.set_failed(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
