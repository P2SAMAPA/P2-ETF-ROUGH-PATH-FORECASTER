"""
Equity Module for ROUGH-PATH-FORECASTER
Benchmark: SPY
Tickers: QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLRE, XLB, GDX, XME, IWM
"""

import numpy as np
import pandas as pd
from constants import EQUITY_TICKERS, EQUITY_BENCHMARK
from data_pipeline import DataPipeline
from models import EnsembleForecaster
from selection import ETFSelector, MacroRegimeContext
from outputs import SignalGenerator, BenchmarkComparator
from utils import Logger, Timer, CacheManager


class EquityModule:
    """Equity module"""
    
    def __init__(self):
        self.tickers = EQUITY_TICKERS
        self.benchmark = EQUITY_BENCHMARK
        self.logger = Logger("Equity-Module")
        self.cache = CacheManager()
        self.regime_detector = MacroRegimeContext()
    
    def get_data(self, start_year=None, end_year=None):
        pipeline = DataPipeline(module='equity')
        data = pipeline.get_data()
        if data is None:
            return None
        if start_year:
            mask = (data['macro_dates'].year >= start_year) & (data['macro_dates'].year <= end_year)
            if len(data['train'][0]) > 0:
                data['train'] = (data['train'][0][mask[:len(data['train'][0])]], 
                                data['train'][1][mask[:len(data['train'][1])]])
            if len(data['val'][0]) > 0:
                data['val'] = (data['val'][0][mask[:len(data['val'][0])]], 
                              data['val'][1][mask[:len(data['val'][1])]])
            if len(data['test'][0]) > 0:
                data['test'] = (data['test'][0][mask[:len(data['test'][0])]], 
                               data['test'][1][mask[:len(data['test'][1])]])
        return data
    
    def train_fixed(self):
        self.logger.info("Training Equity module on fixed dataset (2008-2026)")
        timer = Timer()
        timer.__enter__()
        try:
            data = self.get_data()
            if data is None:
                return None
            X_train = np.vstack([data['train'][0], data['val'][0]])
            y_train = np.vstack([data['train'][1], data['val'][1]])
            model = EnsembleForecaster(depths=[2, 3, 4])
            model.fit(X_train, y_train)
            X_test = data['test'][0]
            y_test = data['test'][1]
            predictions = model.predict(X_test)
            if len(predictions.shape) == 1:
                pred_series = pd.Series(predictions)
                y_test_for_metrics = y_test.mean(axis=1) if len(y_test.shape) > 1 else y_test
            else:
                pred_series = pd.Series(predictions.mean(axis=1))
                y_test_for_metrics = y_test.mean(axis=1) if len(y_test.shape) > 1 else y_test
            metrics = BenchmarkComparator.compute_performance_metrics(pred_series, pd.Series(y_test_for_metrics))
            result = {'model': model, 'predictions': predictions, 'y_test': y_test, 'metrics': metrics}
            timer.__exit__(None, None, None)
            self.logger.info(f"Equity fixed training complete in {timer.minutes:.2f} minutes")
            return result
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            timer.__exit__(None, None, None)
            return None
    
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
                X, y, dates, macro_dates = pipeline.get_window_data(start_year, end_year)
                if len(X) == 0:
                    self.logger.warning(f"No data for window {start_year}-{end_year}, skipping")
                    timer.__exit__(None, None, None)
                    continue
                
                n = len(X)
                train_end = int(n * 0.8)
                val_end = int(n * 0.9)
                
                X_train = X[:train_end]
                y_train = y[:train_end]
                X_val = X[train_end:val_end]
                y_val = y[train_end:val_end]
                X_test = X[val_end:]
                y_test = y[val_end:].copy()  # CRITICAL: Deep copy
                test_dates = dates[val_end:].copy() if len(dates) > val_end else dates.copy()
                
                self.logger.info(f"Window {start_year}: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
                
                model = EnsembleForecaster(depths=[2, 3, 4])
                X_combined = np.vstack([X_train, X_val])
                y_combined = np.vstack([y_train, y_val])
                model.fit(X_combined, y_combined)
                preds = model.predict(X_test).copy()
                
                results.append({
                    'start_year': start_year,
                    'end_year': end_year,
                    'n_days': len(X_test),
                    'model': model,
                    'predictions': preds,
                    'actuals': y_test,
                    'dates': test_dates
                })
                models[start_year] = model
                
                timer.__exit__(None, None, None)
                self.logger.info(f"Window {start_year} complete in {timer.minutes:.2f} min")
            except Exception as e:
                self.logger.error(f"Window {start_year} failed: {e}")
                timer.__exit__(None, None, None)
                continue
        
        return {'windows': results, 'models': models}


def get_equity_module():
    return EquityModule()
