"""
Equity Module for ROUGH-PATH-FORECASTER
Benchmark: SPY
Tickers: SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLRE, XLB, GDX, XME, IWM
"""

import numpy as np
import pandas as pd
from constants import EQUITY_TICKERS, EQUITY_BENCHMARK
from data_pipeline import DataPipeline
from models import EnsembleForecaster
from forecasting import ExpandingWindowConsensus
from selection import ETFSelector, MacroRegimeContext, RoughnessAnalyzer
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
        self.roughness_analyzer = RoughnessAnalyzer()
    
    def get_data(self, start_year=None, end_year=None):
        """Get processed data for Equity universe"""
        pipeline = DataPipeline(module='equity')
        data = pipeline.get_data()
        
        if data is None:
            return None
        
        if start_year:
            # Filter to specific window
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
        """Train on fixed dataset 2008-2026"""
        self.logger.info("Training Equity module on fixed dataset (2008-2026)")
        
        timer = Timer()
        timer.__enter__()
        
        try:
            data = self.get_data()
            
            if data is None:
                self.logger.error("Failed to load data")
                timer.__exit__(None, None, None)
                return None
            
            # Check if data has enough samples
            if len(data['train'][0]) == 0 or len(data['val'][0]) == 0 or len(data['test'][0]) == 0:
                self.logger.error("Not enough data for training")
                self.logger.error(f"Train samples: {len(data['train'][0])}, Val: {len(data['val'][0])}, Test: {len(data['test'][0])}")
                timer.__exit__(None, None, None)
                return None
            
            X_train = np.vstack([data['train'][0], data['val'][0]])
            y_train = np.vstack([data['train'][1], data['val'][1]])
            
            self.logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
            
            # Train ensemble
            model = EnsembleForecaster(depths=[2, 3, 4])
            model.fit(X_train, y_train)
            
            # Test predictions
            X_test = data['test'][0]
            y_test = data['test'][1]
            predictions = model.predict(X_test)
            
            self.logger.info(f"Predictions shape: {predictions.shape}")
            
            # Compute metrics - handle both 1D and 2D
            if len(predictions.shape) == 1:
                pred_series = pd.Series(predictions)
                y_test_for_metrics = y_test.mean(axis=1) if len(y_test.shape) > 1 else y_test
            else:
                pred_series = pd.Series(predictions.mean(axis=1))
                y_test_for_metrics = y_test.mean(axis=1) if len(y_test.shape) > 1 else y_test
            
            metrics = BenchmarkComparator.compute_performance_metrics(pred_series, pd.Series(y_test_for_metrics))
            
            result = {
                'model': model,
                'predictions': predictions,
                'y_test': y_test,
                'metrics': metrics,
                'dates': data['test'][2] if len(data['test']) > 2 else None
            }
            
            timer.__exit__(None, None, None)
            self.logger.info(f"Equity fixed training complete in {timer.minutes:.2f} minutes")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            timer.__exit__(None, None, None)
            return None
    
    def train_shrinking(self, start_years, end_year=2026):
        """Train on expanding windows"""
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
                y_test = y[val_end:]
                
                # Train model
                model = EnsembleForecaster(depths=[2, 3, 4])
                X_combined = np.vstack([X_train, X_val])
                y_combined = np.vstack([y_train, y_val])
                model.fit(X_combined, y_combined)
                
                # Predict on test
                preds = model.predict(X_test)
                
                # Store results
                window_result = {
                    'start_year': start_year,
                    'end_year': end_year,
                    'n_days': len(X_test),
                    'model': model,
                    'predictions': preds,
                    'actuals': y_test,
                    'dates': dates[val_end:] if len(dates) > val_end else dates
                }
                
                results.append(window_result)
                models[start_year] = model
                
                timer.__exit__(None, None, None)
                self.logger.info(f"Window {start_year} complete in {timer.minutes:.2f} min")
                
            except Exception as e:
                self.logger.error(f"Window {start_year} failed: {e}")
                timer.__exit__(None, None, None)
                continue
        
        # Compute consensus
        consensus_weights = {'annualized_return': 0.60, 'sharpe_ratio': 0.20, 'max_drawdown': 0.20}
        consensus = ExpandingWindowConsensus(start_years, end_year, consensus_weights)
        
        return {
            'windows': results,
            'models': models,
            'consensus': consensus
        }
    
    def predict(self, model, X_paths, macro_values):
        """Generate prediction for next day"""
        predictions = model.predict(X_paths)
        
        # Handle 1D vs 2D predictions
        if len(predictions.shape) == 1:
            mean_pred = predictions[0] if len(predictions) > 0 else 0
            per_etf_preds = np.ones(len(self.tickers)) * mean_pred
        else:
            mean_pred = predictions.mean(axis=1)[0] if len(predictions) > 0 else 0
            if len(predictions[0]) == len(self.tickers):
                per_etf_preds = predictions[0]
            else:
                per_etf_preds = np.ones(len(self.tickers)) * mean_pred
        
        regime = self.regime_detector.get_regime(macro_values)
        
        selector = ETFSelector(self.tickers, self.benchmark)
        picks = selector.select_picks(per_etf_preds)
        
        signal_generator = SignalGenerator('equity', self.benchmark, self.tickers)
        signal = signal_generator.generate_signal(
            picks=picks,
            macro_regime=regime,
            roughness_info={},
            signature_depth=3,
            lookback_days=30,
            model_type="Ensemble"
        )
        
        return signal


def get_equity_module():
    """Factory function to get Equity module instance"""
    return EquityModule()
