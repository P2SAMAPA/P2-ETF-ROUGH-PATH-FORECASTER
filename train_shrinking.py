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

from constants import SHRINKING_START_YEARS, SHRINKING_END_YEAR
from module_fi import FIModule
from module_equity import EquityModule
from outputs import ParquetWriter
from selection import ETFSelector
from utils import Logger, Timer, GitHubActionsHelpers


def compute_consensus_picks(window_results, module, start_years):
    """
    Compute consensus picks across all windows using weighted scoring
    """
    tickers = module.tickers
    benchmark = module.benchmark
    
    # Collect picks from each window
    window_picks = []
    window_convictions = []
    window_metrics = []
    
    for i, result in enumerate(window_results):
        start_year = start_years[i]
        
        # Get last prediction from this window
        if len(result['predictions']) > 0:
            last_pred = result['predictions'][-1]
            if len(last_pred.shape) > 1:
                last_pred = last_pred.mean(axis=0)
            
            # Get top pick
            selector = ETFSelector(tickers, benchmark)
            scores = selector.compute_net_scores(last_pred)
            top_pick = scores.iloc[0]['ticker']
            top_conviction = scores.iloc[0]['conviction']
            
            window_picks.append(top_pick)
            window_convictions.append(top_conviction)
            
            # Compute window metrics
            actuals = result['actuals']
            if len(actuals) > 0:
                ann_return = actuals.mean() * 252
                ann_vol = actuals.std() * np.sqrt(252)
                sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                
                cumulative = (1 + actuals).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_dd = drawdown.min()
                
                window_metrics.append({
                    'start_year': start_year,
                    'ann_return_pct': ann_return * 100,
                    'ann_vol_pct': ann_vol * 100,
                    'sharpe': sharpe,
                    'max_drawdown_pct': max_dd * 100
                })
    
    # Count picks for consensus
    from collections import Counter
    pick_counts = Counter(window_picks)
    
    # Weighted consensus by conviction
    weighted_picks = {}
    for pick, conv in zip(window_picks, window_convictions):
        weighted_picks[pick] = weighted_picks.get(pick, 0) + conv
    
    # Top picks
    sorted_picks = sorted(weighted_picks.items(), key=lambda x: x[1], reverse=True)
    
    consensus = {
        'consensus_pick': sorted_picks[0][0] if sorted_picks else None,
        'consensus_conviction': sorted_picks[0][1] / len(window_picks) if sorted_picks else 0,
        'second_pick': sorted_picks[1][0] if len(sorted_picks) > 1 else None,
        'third_pick': sorted_picks[2][0] if len(sorted_picks) > 2 else None,
        'window_picks': window_picks,
        'window_convictions': window_convictions,
        'window_metrics': pd.DataFrame(window_metrics) if window_metrics else pd.DataFrame(),
        'pick_counts': dict(pick_counts)
    }
    
    return consensus


def main():
    parser = argparse.ArgumentParser(description="Train shrinking windows model")
    parser.add_argument("--module", type=str, required=True, choices=['fi', 'equity'],
                        help="Module to train: fi or equity")
    args = parser.parse_args()
    
    logger = Logger(f"Train-Shrinking-{args.module.upper()}")
    logger.info(f"Training {len(SHRINKING_START_YEARS)} shrinking windows")
    
    is_ci = GitHubActionsHelpers.is_github_actions()
    
    with Timer() as t:
        if args.module == 'fi':
            module = FIModule()
            result = module.train_shrinking(SHRINKING_START_YEARS, SHRINKING_END_YEAR)
            save_dir = "models_saved/fi/shrinking"
        else:
            module = EquityModule()
            result = module.train_shrinking(SHRINKING_START_YEARS, SHRINKING_END_YEAR)
            save_dir = "models_saved/equity/shrinking"
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Compute consensus picks
        consensus = compute_consensus_picks(result['windows'], module, SHRINKING_START_YEARS)
        
        # Save each window model
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
        window_results_path = os.path.join(save_dir, "window_results.parquet")
        ParquetWriter.write_window_results(window_results_df, window_results_path)
        
        # Save consensus
        consensus_df = pd.DataFrame([{
            'consensus_pick': consensus['consensus_pick'],
            'consensus_conviction': consensus['consensus_conviction'],
            'second_pick': consensus['second_pick'],
            'third_pick': consensus['third_pick']
        }])
        consensus_path = os.path.join(save_dir, "consensus.parquet")
        ParquetWriter.write_predictions(consensus_df, consensus_path)
        
        # Save window picks
        window_picks_df = pd.DataFrame({
            'start_year': SHRINKING_START_YEARS,
            'pick': consensus['window_picks'],
            'conviction': consensus['window_convictions']
        })
        window_picks_path = os.path.join(save_dir, "window_picks.parquet")
        ParquetWriter.write_predictions(window_picks_df, window_picks_path)
        
        # Save metrics
        if not consensus['window_metrics'].empty:
            metrics_path = os.path.join(save_dir, "window_metrics.parquet")
            ParquetWriter.write_predictions(consensus['window_metrics'], metrics_path)
    
    logger.info(f"Shrinking windows training completed in {t.minutes:.2f} minutes")
    
    if is_ci:
        GitHubActionsHelpers.set_output("training_status", "success")
        GitHubActionsHelpers.set_output("consensus_pick", consensus['consensus_pick'])


if __name__ == "__main__":
    main()
