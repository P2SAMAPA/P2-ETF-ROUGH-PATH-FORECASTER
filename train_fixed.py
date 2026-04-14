#!/usr/bin/env python
"""
Train ROUGH-PATH-FORECASTER on fixed dataset (2008-2026 YTD)
Usage: python train_fixed.py --module [fi|equity]
"""

import argparse
import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime

from module_fi import FIModule
from module_equity import EquityModule
from outputs import ParquetWriter
from utils import Logger, Timer, GitHubActionsHelpers


def main():
    parser = argparse.ArgumentParser(description="Train fixed dataset model")
    parser.add_argument("--module", type=str, required=True, choices=['fi', 'equity'],
                        help="Module to train: fi or equity")
    args = parser.parse_args()
    
    logger = Logger(f"Train-Fixed-{args.module.upper()}")
    
    # Check if in GitHub Actions
    is_ci = GitHubActionsHelpers.is_github_actions()
    
    with Timer() as t:
        if args.module == 'fi':
            logger.info("Training FI module on fixed dataset")
            module = FIModule()
            result = module.train_fixed()
            save_dir = "models_saved/fi/fixed"
        else:
            logger.info("Training Equity module on fixed dataset")
            module = EquityModule()
            result = module.train_fixed()
            save_dir = "models_saved/equity/fixed"
        
        if result is None:
            logger.error("Training failed - no result returned")
            if is_ci:
                GitHubActionsHelpers.set_failed("Training failed")
            return
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f)
        logger.info(f"Saved model to {model_path}")
        
        # Save predictions - handle both 1D and 2D arrays
        predictions = result['predictions']
        if len(predictions.shape) == 1:
            # 1D array - reshape to (n_samples, 1)
            predictions_df = pd.DataFrame(predictions, columns=['predicted_return'])
        else:
            # 2D array - use tickers as columns
            n_etfs = predictions.shape[1]
            if n_etfs == len(module.tickers):
                predictions_df = pd.DataFrame(predictions, columns=module.tickers)
            else:
                # Fallback: use generic column names
                predictions_df = pd.DataFrame(predictions, columns=[f"etf_{i}" for i in range(n_etfs)])
        
        predictions_path = os.path.join(save_dir, "predictions.parquet")
        ParquetWriter.write_predictions(predictions_df, predictions_path)
        logger.info(f"Saved predictions to {predictions_path}")
        
        # Save metrics
        metrics_path = os.path.join(save_dir, "metrics.json")
        ParquetWriter.write_metrics(result['metrics'], metrics_path)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save actuals if available
        if 'y_test' in result and result['y_test'] is not None:
            y_test = result['y_test']
            if len(y_test.shape) == 1:
                actuals_df = pd.DataFrame(y_test, columns=['actual_return'])
            else:
                n_etfs = y_test.shape[1]
                if n_etfs == len(module.tickers):
                    actuals_df = pd.DataFrame(y_test, columns=module.tickers)
                else:
                    actuals_df = pd.DataFrame(y_test, columns=[f"etf_{i}" for i in range(n_etfs)])
            
            actuals_path = os.path.join(save_dir, "actuals.parquet")
            ParquetWriter.write_predictions(actuals_df, actuals_path)
            logger.info(f"Saved actuals to {actuals_path}")
    
    logger.info(f"Fixed training completed in {t.minutes:.2f} minutes")
    
    if is_ci:
        GitHubActionsHelpers.set_output("training_status", "success")
        GitHubActionsHelpers.set_output("model_path", model_path)


if __name__ == "__main__":
    main()
