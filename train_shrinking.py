#!/usr/bin/env python
"""
Train ROUGH-PATH-FORECASTER on shrinking windows
"""

import argparse
import pickle
import os
import pandas as pd
import numpy as np
from collections import Counter

from constants import SHRINKING_START_YEARS, SHRINKING_END_YEAR
from module_fi import FIModule
from module_equity import EquityModule
from outputs import ParquetWriter
from selection import ETFSelector
from utils import Logger, Timer, GitHubActionsHelpers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, required=True, choices=['fi', 'equity'])
    args = parser.parse_args()
    
    logger = Logger(f"Train-Shrinking-{args.module.upper()}")
    logger.info(f"Training {len(SHRINKING_START_YEARS)} shrinking windows")
    
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
        
        # Build metrics and picks from results
        window_metrics_list = []
        window_picks_list = []
        window_convictions_list = []
        
        for window in result['windows']:
            start_year = window['start_year']
            
            # Get top pick from predictions
            predictions = window.get('predictions')
            if predictions is not None and len(predictions) > 0:
                if len(predictions.shape) > 1:
                    last_pred = predictions[-1]
                else:
                    last_pred = predictions[-1]
                
                if isinstance(last_pred, (int, float)):
                    last_pred = np.ones(len(module.tickers)) * last_pred
                elif len(last_pred) > len(module.tickers):
                    last_pred = last_pred[:len(module.tickers)]
                elif len(last_pred) < len(module.tickers):
                    last_pred = np.pad(last_pred, (0, len(module.tickers) - len(last_pred)))
                
                selector = ETFSelector(module.tickers, module.benchmark)
                scores = selector.compute_net_scores(last_pred)
                top_pick = scores.iloc[0]['ticker']
                top_conviction = scores.iloc[0]['conviction']
            else:
                top_pick = "N/A"
                top_conviction = 0
            
            window_picks_list.append(top_pick)
            window_convictions_list.append(top_conviction)
            
            # Use pre-calculated metrics
            window_metrics_list.append({
                'start_year': start_year,
                'n_days': window['n_days'],
                'ann_return_pct': window['ann_return_pct'],
                'ann_vol_pct': window.get('ann_vol_pct', 0),
                'sharpe': window.get('sharpe', 0),
                'max_drawdown_pct': window['max_drawdown_pct'],
                'hit_rate_pct': window.get('hit_rate_pct', 0),
                'ann_alpha_pct': window['ann_return_pct'] - 3.0,
                'positive_years': 0
            })
            
            logger.info(f"Window {start_year}: Days={window['n_days']}, Max DD={window['max_drawdown_pct']:.2f}%")
        
        # Build consensus
        weighted_picks = {}
        for pick, conv in zip(window_picks_list, window_convictions_list):
            if pick != "N/A":
                weighted_picks[pick] = weighted_picks.get(pick, 0) + conv
        
        sorted_picks = sorted(weighted_picks.items(), key=lambda x: x[1], reverse=True)
        
        metrics_df = pd.DataFrame(window_metrics_list)
        picks_df = pd.DataFrame({
            'start_year': [w['start_year'] for w in result['windows']],
            'pick': window_picks_list,
            'conviction': window_convictions_list
        })
        
        consensus_df = pd.DataFrame([{
            'consensus_pick': sorted_picks[0][0] if sorted_picks else None,
            'consensus_conviction': sorted_picks[0][1] / len(window_picks_list) if sorted_picks and window_picks_list else 0,
            'second_pick': sorted_picks[1][0] if len(sorted_picks) > 1 else None,
            'third_pick': sorted_picks[2][0] if len(sorted_picks) > 2 else None
        }])
        
        # Save models
        for start_year, model in result['models'].items():
            model_path = os.path.join(save_dir, f"model_window_{start_year}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save results
        window_results_list = [{'start_year': w['start_year'], 'end_year': w['end_year'], 'n_days': w['n_days']} for w in result['windows']]
        ParquetWriter.write_window_results(pd.DataFrame(window_results_list), os.path.join(save_dir, "window_results.parquet"))
        ParquetWriter.write_predictions(metrics_df, os.path.join(save_dir, "window_metrics.parquet"))
        ParquetWriter.write_predictions(picks_df, os.path.join(save_dir, "window_picks.parquet"))
        ParquetWriter.write_predictions(consensus_df, os.path.join(save_dir, "consensus.parquet"))
        
        timer.__exit__(None, None, None)
        logger.info(f"Training completed in {timer.minutes:.2f} minutes")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        timer.__exit__(None, None, None)
        raise


if __name__ == "__main__":
    main()
