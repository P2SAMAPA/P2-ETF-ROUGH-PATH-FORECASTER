"""
Fixed Income / Commodities Module for ROUGH-PATH-FORECASTER - FIXED VERSION
"""

import numpy as np
import pandas as pd
from constants import FI_TICKERS, FI_BENCHMARK
from data_pipeline import DataPipeline
from models import EnsembleForecaster
from selection import ETFSelector, MacroRegimeContext
from outputs import SignalGenerator, BenchmarkComparator
from utils import Logger, Timer


class FIModule:
    def __init__(self):
        self.tickers = FI_TICKERS
        self.benchmark = FI_BENCHMARK
        self.logger = Logger("FI-Module")
        self.regime_detector = MacroRegimeContext()
    
    def train_shrinking(self, start_years, end_year=2026):
        self.logger.info(f"Training FI module on {len(start_years)} shrinking windows")
        results = []
        models = {}
        
        for start_year in start_years:
            self.logger.info(f"Training window: {start_year} -> {end_year}")
            timer = Timer()
            timer.__enter__()
            
            try:
                pipeline = DataPipeline(module='fi')
                X, y, dates, _ = pipeline.get_window_data(start_year, end_year)
                
                if len(X) == 0:
                    self.logger.warning(f"No data for window {start_year}, skipping")
                    timer.__exit__(None, None, None)
                    continue
                
                # Use a FIXED test size (e.g., 252 trading days = 1 year)
                test_days = 252
                
                if len(dates) <= test_days:
                    self.logger.warning(f"Window {start_year} has only {len(dates)} days, skipping")
                    timer.__exit__(None, None, None)
                    continue
                
                train_size = len(dates) - test_days
                
                X_train = X[:train_size]
                y_train = y[:train_size]
                X_test = X[train_size:]
                y_test = y[train_size:].copy()
                test_dates = dates[train_size:].copy()
                
                self.logger.info(f"Window {start_year}: train={len(X_train)}, test={len(X_test)}")
                
                model = EnsembleForecaster(depths=[2, 3, 4])
                model.fit(X_train, y_train)
                preds = model.predict(X_test).copy()
                
                returns = y_test.mean(axis=1) if len(y_test.shape) > 1 else y_test
                
                ann_return = np.mean(returns) * 252
                ann_vol = np.std(returns) * np.sqrt(252)
                sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_dd = abs(np.min(drawdown)) * 100
                
                hit_rate = np.mean(returns > 0) * 100
                
                self.logger.info(f"Window {start_year}: Days={len(X_test)}, Max DD={max_dd:.2f}%, Vol={ann_vol*100:.2f}%")
                
                results.append({
                    'start_year': start_year,
                    'end_year': end_year,
                    'n_days': len(X_test),
                    'model': model,
                    'predictions': preds,
                    'actuals': y_test,
                    'dates': test_dates,
                    'ann_return_pct': ann_return * 100,
                    'ann_vol_pct': ann_vol * 100,
                    'max_drawdown_pct': max_dd,
                    'sharpe': sharpe,
                    'hit_rate_pct': hit_rate
                })
                models[start_year] = model
                
                timer.__exit__(None, None, None)
                self.logger.info(f"Window {start_year} complete in {timer.minutes:.2f} min")
                
            except Exception as e:
                self.logger.error(f"Window {start_year} failed: {e}")
                import traceback
                traceback.print_exc()
                timer.__exit__(None, None, None)
                continue
        
        return {'windows': results, 'models': models}
    
    def train_fixed(self):
        self.logger.info("Training FI module on fixed dataset")
        timer = Timer()
        timer.__enter__()
        try:
            pipeline = DataPipeline(module='fi')
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


def get_fi_module():
    return FIModule()
