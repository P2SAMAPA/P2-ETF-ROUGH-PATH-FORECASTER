#!/usr/bin/env python
"""
Core unit tests for ROUGH-PATH-FORECASTER
"""

import unittest
import numpy as np
import pandas as pd


class TestSignatureComputer(unittest.TestCase):
    def test_signature_computation(self):
        from signature_core import SignatureComputer
        
        computer = SignatureComputer(depth=2)
        path = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        
        sig = computer.compute_signature(path)
        
        self.assertIsNotNone(sig)
        self.assertIn(tuple(), sig)  # depth 0
        self.assertGreater(len(sig), 0)
    
    def test_adaptive_depth_selection(self):
        from signature_core import AdaptiveDepthSelector
        
        selector = AdaptiveDepthSelector(depths=[2, 3])
        X_paths = [np.random.randn(10, 2) for _ in range(20)]
        y_vals = np.random.randn(20)
        
        best_depth = selector.select_depth(X_paths, y_vals)
        
        self.assertIn(best_depth, [2, 3])


class TestKernelEngine(unittest.TestCase):
    def test_neumann_kernel(self):
        from signature_core import NeumannSignatureKernel
        
        kernel = NeumannSignatureKernel(depth=2, tile_size=10)
        path1 = np.random.randn(20, 3)
        path2 = np.random.randn(20, 3)
        
        # Should compute without errors
        result = kernel._neumann_expansion(path1, path2)
        
        self.assertIsInstance(result, float)


class TestETFSelector(unittest.TestCase):
    def test_select_picks(self):
        from selection import ETFSelector
        
        tickers = ['AAPL', 'GOOG', 'MSFT']
        selector = ETFSelector(tickers, 'SPY')
        
        predictions = np.array([0.01, 0.02, -0.005])
        picks = selector.select_picks(predictions)
        
        self.assertEqual(len(picks), 3)
        self.assertEqual(picks[0]['ticker'], 'GOOG')  # Highest return


class TestMacroRegimeContext(unittest.TestCase):
    def test_regime_detection(self):
        from selection import MacroRegimeContext
        
        detector = MacroRegimeContext()
        
        # Risk-on regime
        macro_risk_on = {'VIX': 15, 'HY_SPREAD': 0.02, 'T10Y2Y': 0.5}
        regime_on = detector.get_regime(macro_risk_on)
        self.assertEqual(regime_on['regime_label'], 'Risk-On')
        
        # Risk-off regime
        macro_risk_off = {'VIX': 22, 'HY_SPREAD': 0.045, 'T10Y2Y': 0.2}
        regime_off = detector.get_regime(macro_risk_off)
        self.assertEqual(regime_off['regime_label'], 'Risk-Off')


class TestPerformanceMetrics(unittest.TestCase):
    def test_sharpe_calculation(self):
        from outputs import BenchmarkComparator
        
        returns = pd.Series([0.001, -0.0005, 0.002, 0.0005, 0.0015])
        metrics = BenchmarkComparator.compute_performance_metrics(returns)
        
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('annualized_return_pct', metrics)


if __name__ == '__main__':
    unittest.main()
