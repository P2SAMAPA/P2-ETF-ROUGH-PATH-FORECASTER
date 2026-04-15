#!/usr/bin/env python
"""
Integration tests for ROUGH-PATH-FORECASTER
"""

import unittest
import numpy as np
import os
import tempfile


class TestDataPipelineIntegration(unittest.TestCase):
    def test_data_loading(self):
        """Test data loading from HF (requires internet)"""
        import pytest
        pytest.skip("Skipping HF data test - requires internet and token")
        
        from data_pipeline import DataPipeline
        
        pipeline = DataPipeline(module='fi')
        pipeline.load_data()
        
        self.assertIsNotNone(pipeline.raw_data)
        self.assertGreater(len(pipeline.raw_data), 0)


class TestTrainPredictIntegration(unittest.TestCase):
    def test_train_and_predict_small(self):
        """Test training and prediction on synthetic data"""
        from models import EnsembleForecaster
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 50
        path_length = 20
        n_features = 5
        n_etfs = 3
        
        X_paths = [np.random.randn(path_length, n_features) for _ in range(n_samples)]
        y_returns = np.random.randn(n_samples, n_etfs)  # 3 ETFs
        
        # Train
        model = EnsembleForecaster(depths=[2, 3])
        model.fit(X_paths, y_returns)
        
        # Predict
        test_paths = [np.random.randn(path_length, n_features) for _ in range(5)]
        predictions = model.predict(test_paths)
        
        # Check predictions shape
        self.assertEqual(len(predictions), 5)
        self.assertEqual(predictions.shape[1], n_etfs)
        self.assertIsInstance(predictions, np.ndarray)
        
        # Test single prediction - returns array of predictions for each ETF
        single_pred = model.predict_single(test_paths[0])
        self.assertIsInstance(single_pred, np.ndarray)
        self.assertEqual(len(single_pred), n_etfs)


class TestOutputGeneration(unittest.TestCase):
    def test_signal_generation(self):
        from selection import ETFSelector, MacroRegimeContext
        from outputs import SignalGenerator
        
        tickers = ['TLT', 'LQD', 'HYG']
        selector = ETFSelector(tickers, 'AGG')
        
        predictions = np.array([0.01, 0.005, 0.02])
        picks = selector.select_picks(predictions)
        
        regime_detector = MacroRegimeContext()
        regime = regime_detector.get_regime({'VIX': 15, 'HY_SPREAD': 0.03, 'T10Y2Y': 0.5})
        
        signal_gen = SignalGenerator('fi', 'AGG', tickers)
        signal = signal_gen.generate_signal(
            picks=picks,
            macro_regime=regime,
            roughness_info={'roughness': 0.5, 'hurst': 0.6},
            signature_depth=3,
            lookback_days=30
        )
        
        self.assertEqual(signal['etf_pick'], 'HYG')
        self.assertIn('conviction_percentage', signal)
        self.assertIn('macro_pills', signal)
        self.assertEqual(len(signal['macro_pills']), 5)


class TestCacheSystem(unittest.TestCase):
    def test_cache_manager(self):
        from utils import CacheManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)
            
            # Set and get
            cache.set("test_key", "test_value")
            value = cache.get("test_key")
            
            self.assertEqual(value, "test_value")
            
            # Clear
            cache.clear()
            value_after = cache.get("test_key")
            self.assertIsNone(value_after)


if __name__ == '__main__':
    unittest.main()
