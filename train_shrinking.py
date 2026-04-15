#!/usr/bin/env python
"""
Train ROUGH-PATH-FORECASTER on shrinking windows (2008→2026 through 2024→2026)
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


def calculate_max_drawdown(returns_array):
    """Calculate max drawdown from returns array"""
    if returns_array is None or len(returns_array) == 0:
        return 0.0
    
    # Flatten if needed
    if len(returns_array.shape) > 1:
        returns_array = returns_array.mean(axis=1)
    
    # Remove NaN values
    returns_array = returns_array[~np.isnan(returns_array)]
    
    if len(returns_array) < 3:
        return 0.0
    
    # Calculate cumulative product
    cumulative = np.cumprod(1 + returns_array)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Return max drawdown as positive percentage
    max_dd = abs(np.min(drawdown)) * 100
    
    return max_dd


def calculate_window_metrics(y_test, start_year, dates):
    """Calculate all metrics for a single window - called immediately"""
    
    if y_test is None or len(y_test) == 0:
        return {
            'start_year': start_year,
            'n_days': 0,
            'ann_return_pct': 0.0,
            'ann_vol_pct': 0.0,
            'sharpe': 0.0,
            'max_drawdown_pct': 0.0,
            'hit_rate_pct': 0.0,
            'ann_alpha_pct': 0.0,
            'positive_years': 0
        }
    
    # Flatten y_test if needed
    if len(y_test.shape) > 1:
        # Use mean return across ETFs as strategy return
        returns = y_test.mean(axis=1)
    else:
        returns = y_test
    
    n_days = len(returns)
    
    # Annualized metrics
    ann_return = np.mean(returns) * 252
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    
    # Max drawdown - calculate directly
    max_dd = calculate_max_drawdown(returns)
    
    # Hit rate
    hit_rate = np.mean(returns > 0) * 100
    
    # Alpha
    ann_alpha = (ann_return - 0.03) * 100
    
    # Positive years
    positive_years = 0
    if dates is not None and len(dates) == len(returns):
        try:
            df = pd.DataFrame({'return': returns}, index=pd.DatetimeIndex(dates))
            yearly_returns = df.groupby(df.index.year)['return'].mean()
            positive_years = (yearly_returns > 0).sum()
        except:
            positive_years = 0
    
    return {
        'start_year': start_year,
        'n_days': n_days,
        'ann_return_pct': ann_return * 100,
        'ann_vol_pct': ann_vol * 100,
        'sharpe': sharpe,
        'max_drawdown_pct': max_dd,
        'hit_rate_pct': hit_rate,
        'ann_alpha_pct': ann_alpha,
        'positive_years': positive_years
    }


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
        
        # Process each window and calculate metrics immediately
        window_metrics_list = []
        window_picks_list = []
        window_convictions_list = []
        
        for i, window in enumerate(result['windows']):
            start_year = window['start_year']
            y_test = window.get('actuals')
            dates = window.get('dates')
            predictions = window.get('predictions')
            
            # Calculate metrics NOW while data is fresh
            metrics = calculate_window_metrics(y_test, start_year, dates)
            window_metrics_list.append(metrics)
            
            # Get top pick from predictions
            if predictions is not None and len(predictions) > 0:
                if len(predictions.shape) > 1:
                    last_pred = predictions[-1]
                else:
                    last_pred = predictions[-1]
                
                # Ensure correct length
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
                
                window_picks_list.append(top_pick)
                window_convictions_list.append(top_conviction)
            else:
                window_picks_list.append("N/A")
                window_convictions_list.append(0)
            
            logger.info(f"Window {start_year}: Days={metrics['n_days']}, Max DD={metrics['max_drawdown_pct']:.2f}%, Pick={window_picks_list[-1]}")
        
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
        
        # Save all files
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
        
        # Save metrics (NOW with correct values)
        ParquetWriter.write_predictions(metrics_df, os.path.join(save_dir, "window_metrics.parquet"))
        
        # Save picks
        ParquetWriter.write_predictions(picks_df, os.path.join(save_dir, "window_picks.parquet"))
        
        # Save consensus
        ParquetWriter.write_predictions(consensus_df, os.path.join(save_dir, "consensus.parquet"))
        
        timer.__exit__(None, None, None)
        logger.info(f"Training completed in {timer.minutes:.2f} minutes")
        logger.info(f"Final metrics shape: {metrics_df.shape}")
        logger.info(metrics_df[['start_year', 'n_days', 'max_drawdown_pct']].to_string())
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        timer.__exit__(None, None, None)
        raise


if __name__ == "__main__":
    main()
