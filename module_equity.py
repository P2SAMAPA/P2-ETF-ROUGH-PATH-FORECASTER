"""
Equity Module for ROUGH-PATH-FORECASTER
"""

import numpy as np
import pandas as pd
from constants import EQUITY_TICKERS, EQUITY_BENCHMARK
from data_pipeline import DataPipeline
from models import EnsembleForecaster
from selection import ETFSelector, MacroRegimeContext
from outputs import SignalGenerator, BenchmarkComparator
from utils import Logger, Timer


class EquityModule:
    def __init__(self):
        self.tickers = EQUITY_TICKERS
        self.benchmark = EQUITY_BENCHMARK
        self.logger = Logger("Equity-Module")
        self.regime_detector = MacroRegimeContext()
    
    def train_shrinking(self, start_years, end_year=2026):
        self.logger.info(f"Training Equity module on {len(start_years)} shrinking windows")
        results = []
        models = {}
        
        for start_year in start_years:
            self.logger.info(f"Training window: {start_year} -> {end_year}")
            timer = Timer()
            timer.__enter__()
            
            try:
                pipeline = DataPipeline(module='equity')
                X, y, dates, _ = pipeline.get_window_data(start_year, end_year)
                
                if len(X) == 0:
                    self.logger.warning(f"No data for window {start_year}, skipping")
                    timer.__exit__(None, None, None)
                    continue
                
                # Use time-based split: last 20% of dates as test
                n = len(dates)
                test_size = int(n * 0.2)
                train_size = n - test_size
                
                # Split by position (last test_size days are test)
                X_train = X[:train_size]
                y_train = y[:train_size]
                X_test = X[train_size:]
                y_test = y[train_size:].copy()
                test_dates = dates[train_size:].copy()
                
                self.logger.info(f"Window {start_year}: train={len(X_train)}, test={len(X_test)}")
                
                model = EnsembleForecaster(depths=[2, 3, 4])
                model.fit(X_train, y_train)
                preds = model.predict(X_test).copy()
                
                # Calculate metrics immediately
                returns = y_test.mean(axis=1) if len(y_test.shape) > 1 else y_test
                ann_return = np.mean(returns) * 252
                ann_vol = np.std(returns) * np.sqrt(252)
                sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                
                # Calculate max drawdown
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_dd = abs(np.min(drawdown)) * 100
                
                hit_rate = np.mean(returns > 0) * 100
                
                self.logger.info(f"Window {start_year}: Days={len(X_test)}, Max DD={max_dd:.2f}%, Return={ann_return*100:.2f}%")
                
                results.append({
                    'start_year': start_year,
                    'end_year': end_year,
                    'n_days': len(X_test),
                    'model': model,
                    'predictions': preds,
                    'actuals': y_test,
                    'dates': test_dates,
                    'ann_return_pct': ann_return * 100,
                    'max_drawdown_pct': max_dd,
                    'sharpe': sharpe,
                    'hit_rate_pct': hit_rate
                })
                models[start_year] = model
                
                timer.__exit__(None, None, None)
                self.logger.info(f"Window {start_year} complete in {timer.minutes:.2f} min")
                
            except Exception as e:
                self.logger.error(f"Window {start_year} failed: {e}")
                timer.__exit__(None, None, None)
                continue
        
        return {'windows': results, 'models': models}
    
    def train_fixed(self):
        self.logger.info("Training Equity module on fixed dataset")
        timer = Timer()
        timer.__enter__()
        try:
            pipeline = DataPipeline(module='equity')
            X, y, dates, _ = pipeline.get_window_data(2008, 2026)
            n = len(dates)
            train_size = int(n * 0.8)
            val_size = int(n * 0.1)
            
            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
            X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
            
            model = EnsembleForecaster(depths=[2, 3, 4])
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.vstack([y_train, y_val])
            model.fit(X_combined, y_combined)
            
            predictions = model.predict(X_test)
            returns = y_test.mean(axis=1)
            pred_returns = predictions.mean(axis=1) if len(predictions.shape) > 1 else predictions
            
            metrics = BenchmarkComparator.compute_performance_metrics(pd.Series(pred_returns), pd.Series(returns))
            
            timer.__exit__(None, None, None)
            self.logger.info(f"Fixed training complete in {timer.minutes:.2f} minutes")
            
            return {'model': model, 'predictions': predictions, 'y_test': y_test, 'metrics': metrics}
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            timer.__exit__(None, None, None)
            return None


def get_equity_module():
    return EquityModule()
