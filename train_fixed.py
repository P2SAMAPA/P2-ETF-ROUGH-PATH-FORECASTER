#!/usr/bin/env python
"""
Train ROUGH-PATH-FORECASTER on fixed dataset (2008-2026 YTD)
Usage: python train_fixed.py --module [fi|equity]
"""

import argparse
import pickle
import os
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
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f)
        logger.info(f"Saved model to {model_path}")
        
        # Save predictions
        predictions_path = os.path.join(save_dir, "predictions.parquet")
        ParquetWriter.write_predictions(
            pd.DataFrame(result['predictions'], columns=module.tickers),
            predictions_path
        )
        
        # Save metrics
        metrics_path = os.path.join(save_dir, "metrics.json")
        ParquetWriter.write_metrics(result['metrics'], metrics_path)
        
        # Save actuals if available
        if 'y_test' in result:
            actuals_path = os.path.join(save_dir, "actuals.parquet")
            ParquetWriter.write_predictions(
                pd.DataFrame(result['y_test'], columns=module.tickers),
                actuals_path
            )
    
    logger.info(f"Fixed training completed in {t.minutes:.2f} minutes")
    
    if is_ci:
        GitHubActionsHelpers.set_output("training_status", "success")
        GitHubActionsHelpers.set_output("model_path", model_path)


if __name__ == "__main__":
    import pandas as pd
    main()
