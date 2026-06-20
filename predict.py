#!/usr/bin/env python
"""
Daily prediction for ROUGH-PATH-FORECASTER
Usage: python predict.py --module [fi|equity]
"""

import argparse
import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime

from data_pipeline import DataPipeline
from utils import Logger, Timer, GitHubActionsHelpers

# MUST match the PATH_WINDOW used in module_fi.py and module_equity.py
PREDICT_PATH_WINDOW = 21 


def load_latest_model(module, mode='fixed'):
    """Load the most recent trained model"""
    if mode == 'fixed':
        model_path = f"models_saved/{module}/fixed/model.pkl"
    else:
        model_path = f"models_saved/{module}/shrinking/model_window_2024.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # CRITICAL FIX: Handle both direct model saves and dict-wrapped saves
        if isinstance(data, dict):
            return data.get('model')
        return data
    return None


def get_latest_path_data(module, lookback_days=PREDICT_PATH_WINDOW):
    """Get the most recent lookback_days of path data"""
    pipeline = DataPipeline(module=module)
    pipeline.load_data()
    
    # Get recent macro data
    macro_data = pipeline.extract_macro_data()
    # CRITICAL FIX: Fetch exactly PREDICT_PATH_WINDOW days to match training shape
    recent_macro = macro_data.tail(lookback_days)
    
    # Get recent ETF returns
    etf_returns = pipeline.extract_etf_returns()
    recent_returns = etf_returns.tail(lookback_days)
    
    # Align
    common_dates = recent_macro.index.intersection(recent_returns.index)
    macro_aligned = recent_macro.loc[common_dates]
    
    # Ensure we have exactly the required window length
    if len(macro_aligned) < lookback_days:
        raise ValueError(f"Insufficient data: need {lookback_days} days, got {len(macro_aligned)}")

    # Create path
    X_path = pipeline.create_path_augmentation(macro_aligned)
    
    # CRITICAL FIX: Do NOT blindly reshape. 
    # If the pipeline adds lead-lag, the time dimension is dynamically 2N-1.
    # Just ensure it's a 2D numpy array (Time, Features), then wrap in a list.
    X_path = np.array(X_path)
    
    return [X_path], macro_aligned.iloc[-1].to_dict()


def main():
    parser = argparse.ArgumentParser(description="Run daily predictions")
    parser.add_argument("--module", type=str, required=True, choices=['fi', 'equity'],
                        help="Module to predict: fi or equity")
    args = parser.parse_args()
    
    logger = Logger(f"Predict-{args.module.upper()}")
    is_ci = GitHubActionsHelpers.is_github_actions()
    
    with Timer() as t:
        # Load model
        model = load_latest_model(args.module, mode='fixed')
        if model is None:
            logger.error(f"No model found for {args.module}")
            if is_ci:
                GitHubActionsHelpers.set_failed(f"No model found for {args.module}")
            return
        
        # Get latest data (strictly uses PREDICT_PATH_WINDOW now)
        X_paths, macro_values = get_latest_path_data(args.module)
        
        # Predict
        predictions = model.predict(X_paths)
        
        # Get per-ETF predictions
        if len(predictions.shape) > 1:
            per_etf_preds = predictions[0]
        else:
            per_etf_preds = predictions
        
        # Get macro regime
        from selection import MacroRegimeContext
        regime_detector = MacroRegimeContext()
        regime = regime_detector.get_regime(macro_values)
        
        # Select ETF
        from selection import ETFSelector
        if args.module == 'fi':
            from constants import FI_TICKERS, FI_BENCHMARK
            tickers = FI_TICKERS
            benchmark = FI_BENCHMARK
        else:
            from constants import EQUITY_TICKERS, EQUITY_BENCHMARK
            tickers = EQUITY_TICKERS
            benchmark = EQUITY_BENCHMARK
        
        selector = ETFSelector(tickers, benchmark)
        picks = selector.select_picks(per_etf_preds)
        
        # Generate signal
        from outputs import SignalGenerator
        signal_gen = SignalGenerator(args.module, benchmark, tickers)
        signal = signal_gen.generate_signal(
            picks=picks,
            macro_regime=regime,
            roughness_info={},
            signature_depth=3,
            lookback_days=PREDICT_PATH_WINDOW,
            model_type="Ensemble"
        )
        
        # Save signal
        os.makedirs("outputs", exist_ok=True)
        signal_path = f"outputs/{args.module}_signal_{datetime.now().strftime('%Y%m%d')}.json"
        import json
        with open(signal_path, 'w') as f:
            json.dump(signal, f, indent=2)
        
        logger.info(f"Prediction for {args.module}: {signal['etf_pick']} with {signal['conviction_percentage']}% conviction")
        logger.info(f"Signal saved to {signal_path}")
    
    logger.info(f"Prediction completed in {t.seconds:.2f} seconds")
    
    if is_ci:
        GitHubActionsHelpers.set_output("prediction_pick", signal['etf_pick'])
        GitHubActionsHelpers.set_output("prediction_conviction", signal['conviction_percentage'])


if __name__ == "__main__":
    main()
