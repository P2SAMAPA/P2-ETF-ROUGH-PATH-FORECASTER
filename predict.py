#!/usr/bin/env python
"""
Daily prediction for ROUGH-PATH-FORECASTER
Usage: python predict.py --module [fi|equity]
"""

import argparse
import pickle
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from data_pipeline import DataPipeline
from utils import Logger, Timer, GitHubActionsHelpers

# CRITICAL: Must strictly match the PATH_WINDOW used in module_fi.py and module_equity.py
# Signature geometry breaks if inference paths are a different length than training paths.
PREDICT_PATH_WINDOW = 21 


def load_latest_model(module, mode='fixed'):
    """Load the most recent trained model"""
    if mode == 'fixed':
        model_path = f"models_saved/{module}/fixed/model.pkl"
    else:
        # For shrinking, load the consensus model (use most recent window)
        model_path = f"models_saved/{module}/shrinking/model_window_2024.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # CRITICAL FIX: Handle both direct model saves and dict-wrapped saves
        # (train_shrinking.py saves {'model': model, 'scaler': None})
        if isinstance(data, dict):
            return data.get('model')
        return data
    return None


def get_latest_path_data(module):
    """
    Get the most recent path data for inference.
    Returns a list containing a single 2D path array, and a dict of current macro values.
    """
    pipeline = DataPipeline(module=module)
    
    # Fetches exactly PREDICT_PATH_WINDOW days of raw macro features
    X_path, dates, macro_values = pipeline.get_recent_features(window=PREDICT_PATH_WINDOW)
    
    # CRITICAL FIX: Do NOT reshape or add a batch dimension.
    # The new models.py expects a LIST of 2D paths: [(Time, Features), ...]
    # Wrapping it in a list natively handles the batch dimension without 
    # destroying the temporal shape if data_pipeline applies Lead-Lag.
    return [X_path], macro_values


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
            logger.warning(f"No model found for {args.module}. Skipping prediction.")
            # Do NOT call set_failed. Missing model is a valid state before training runs.
            return
        
        # Get latest data (strictly uses PREDICT_PATH_WINDOW)
        try:
            X_paths, macro_values = get_latest_path_data(args.module)
        except ValueError as e:
            logger.error(f"Data error: {e}")
            if is_ci:
                GitHubActionsHelpers.set_failed(str(e))
            return
        
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
