"""
Fixed Income / Commodities Module for ROUGH-PATH-FORECASTER
Benchmark: AGG
Tickers: TLT, LQD, HYG, VNQ, GLD, SLV, VCIT
"""

from constants import FI_TICKERS, FI_BENCHMARK
from data_pipeline import DataPipeline
from models import EnsembleForecaster
from forecasting import ExpandingWindowConsensus
from selection import ETFSelector, MacroRegimeContext, RoughnessAnalyzer
from outputs import SignalGenerator, BenchmarkComparator
from utils import Logger, Timer, CacheManager

import numpy as np
import pandas as pd


class FIModule:
    """Fixed Income/Commodities module"""
    
    def __init__(self):
        self.tickers = FI_TICKERS
        self.benchmark = FI_BENCHMARK
        self.logger = Logger("FI-Module")
        self.cache = CacheManager()
        self.regime_detector = MacroRegimeContext()
        self.roughness_analyzer = RoughnessAnalyzer()
    
    def get_data(self, start_year=None, end_year=None):
        """Get processed data for FI universe"""
        pipeline = DataPipeline(module='fi')
        data = pipeline.get_data()
        
        if start_year:
            # Filter to specific window
            mask = (data['macro_dates'].year >= start_year) & (data['macro_dates'].year <= end_year)
            data['train'] = (data['train'][0][mask[:len(data['train'][0])]], data['train'][1][mask[:len(data['train'][1])]])
            data['val'] = (data['val'][0][mask[:len(data['val'][0])]], data['val'][1][mask[:len(data['val'][1])]])
            data['test'] = (data['test'][0][mask[:len(data['test'][0])]], data['test'][1][mask[:len(data['test'][1])]])
        
        return data
    
    def train_fixed(self):
        """Train on fixed dataset 2008-2026"""
        self.logger.info("Training FI module on fixed dataset (2008-2026)")
        
        with Timer() as t:
            data = self.get_data()
            
            # Combine train+val for final model
            X_train = np.vstack([data['train'][0], data['val'][0]])
            y_train = np.vstack([data['train'][1], data['val'][1]])
            
            # Train ensemble
            model = EnsembleForecaster(depths=[2, 3, 4])
            model.fit(X_train, y_train)
            
            # Test predictions
            X_test = data['test'][0]
            y_test = data['test'][1]
            predictions = model.predict(X_test)
            
            # Compute metrics
            metrics = BenchmarkComparator.compute_performance_metrics(
                pd.Series(predictions.mean(axis=1)), 
                pd.Series(y_test.mean(axis=1))
            )
            
            self.logger.info(f"Fixed training complete in {t.minutes:.2f} min")
            
            return {
                'model': model,
                'predictions': predictions,
                'y_test': y_test,
                'metrics': metrics,
                'dates': data['test'][2] if len(data['test']) > 2 else None
            }
    
    def train_shrinking(self, start_years, end_year=2026):
        """Train on expanding windows"""
        self.logger.info(f"Training FI module on {len(start_years)} shrinking windows")
        
        results = []
        models = {}
        
        for start_year in start_years:
            self.logger.info(f"Training window: {start_year} -> {end_year}")
            
            with Timer() as t:
                pipeline = DataPipeline(module='fi')
                X, y, dates, _ = pipeline.get_window_data(start_year, end_year)
                
                # Split into train/val/test (80/10/10)
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
                    'dates': dates[val_end:]
                }
                
                results.append(window_result)
                models[start_year] = model
                
                self.logger.info(f"Window {start_year} complete in {t.minutes:.2f} min")
        
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
        # Predict returns
        predictions = model.predict(X_paths)
        mean_pred = predictions.mean(axis=1)[0] if len(predictions.shape) > 1 else predictions[0]
        
        # Get regime
        regime = self.regime_detector.get_regime(macro_values)
        
        # Select ETF
        selector = ETFSelector(self.tickers, self.benchmark)
        
        # For simplicity, use mean prediction across ETFs
        # In production, use per-ETF predictions
        per_etf_preds = predictions[0] if len(predictions.shape) > 1 else predictions
        picks = selector.select_picks(per_etf_preds, date=pd.Timestamp.now())
        
        # Generate signal
        signal_generator = SignalGenerator('fi', self.benchmark, self.tickers)
        signal = signal_generator.generate_signal(
            picks=picks,
            macro_regime=regime,
            roughness_info={},
            signature_depth=3,
            lookback_days=30,
            model_type="Ensemble"
        )
        
        return signal


def get_fi_module():
    """Factory function to get FI module instance"""
    return FIModule()
