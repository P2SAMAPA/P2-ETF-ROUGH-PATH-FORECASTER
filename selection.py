"""
ETF selection and conviction scoring for ROUGH-PATH-FORECASTER
"""

import numpy as np
import pandas as pd
from constants import TRADE_COST_BPS, MIN_CONVICTION_FOR_TRADE, TOP_N_PICKS


class ETFSelector:
    """Select top ETFs based on predicted returns and conviction"""
    
    def __init__(self, tickers, benchmark):
        self.tickers = tickers
        self.benchmark = benchmark
        self.trade_cost_bps = TRADE_COST_BPS
        self.min_conviction = MIN_CONVICTION_FOR_TRADE
        self.top_n = TOP_N_PICKS
    
    def compute_net_scores(self, predicted_returns, predicted_uncertainties=None):
        """
        Compute net scores after trade cost penalty
        """
        # Apply trade cost penalty
        trade_cost = self.trade_cost_bps / 10000  # Convert bps to decimal
        net_returns = predicted_returns - trade_cost
        
        # Calculate conviction (higher for higher predicted return, lower for higher uncertainty)
        if predicted_uncertainties is not None:
            # Conviction = (pred_return - trade_cost) / (uncertainty + epsilon)
            epsilon = 1e-6
            conviction = net_returns / (predicted_uncertainties + epsilon)
            # Normalize to 0-100 range
            conviction = (conviction - conviction.min()) / (conviction.max() - conviction.min() + epsilon)
            conviction = conviction * 100
        else:
            # Simple normalized conviction
            conviction = (net_returns - net_returns.min()) / (net_returns.max() - net_returns.min() + 1e-6) * 100
        
        # Create results DataFrame
        results = pd.DataFrame({
            'ticker': self.tickers,
            'predicted_return': predicted_returns,
            'net_return': net_returns,
            'conviction': conviction
        })
        
        # Sort by net return
        results = results.sort_values('net_return', ascending=False)
        
        return results
    
    def select_picks(self, predicted_returns, predicted_uncertainties=None, date=None):
        """
        Select top N ETFs as picks
        """
        scores = self.compute_net_scores(predicted_returns, predicted_uncertainties)
        
        picks = []
        for i in range(min(self.top_n, len(scores))):
            row = scores.iloc[i]
            if row['conviction'] >= self.min_conviction or i == 0:  # Always include top pick
                picks.append({
                    'rank': i + 1,
                    'ticker': row['ticker'],
                    'predicted_return': row['predicted_return'],
                    'net_return': row['net_return'],
                    'conviction': row['conviction']
                })
        
        return picks
    
    def get_benchmark_info(self):
        """Get benchmark information"""
        return {
            'ticker': self.benchmark,
            'note': 'not traded · no CASH output'
        }


class ConvictionScorer:
    """Compute conviction scores for picks"""
    
    @staticmethod
    def compute_confidence(predicted_return, uncertainty, historical_accuracy=0.5):
        """
        Compute confidence score (0-100)
        """
        # Signal strength component (higher return = higher confidence)
        return_magnitude = min(abs(predicted_return) * 100, 1.0)
        
        # Uncertainty component (lower uncertainty = higher confidence)
        if uncertainty > 0:
            uncertainty_component = 1.0 / (1.0 + uncertainty * 10)
        else:
            uncertainty_component = 1.0
        
        # Historical accuracy component
        accuracy_component = historical_accuracy
        
        # Combine
        confidence = (return_magnitude * 0.4 + uncertainty_component * 0.3 + accuracy_component * 0.3) * 100
        
        return min(confidence, 100.0)
    
    @staticmethod
    def normalize_conviction(scores):
        """Normalize conviction scores to sum to 100%"""
        total = scores.sum()
        if total > 0:
            return scores / total * 100
        return scores


class MacroRegimeContext:
    """Determine market regime from macro data"""
    
    REGIME_LABELS = {
        0: "Risk-On",
        1: "Risk-Off", 
        2: "Transitional",
        3: "Crisis"
    }
    
    def __init__(self, macro_cols=["VIX", "T10Y2Y", "HY_SPREAD"]):
        self.macro_cols = macro_cols
        self.regime_model = None
    
    def get_regime(self, macro_values):
        """
        Determine current regime based on macro values
        Simple rule-based for now (can be upgraded to KMeans)
        """
        vix = macro_values.get('VIX', 15)
        hy_spread = macro_values.get('HY_SPREAD', 0.03)
        t10y2y = macro_values.get('T10Y2Y', 0.5)
        
        # Rule-based regime detection
        if vix > 25 and hy_spread > 0.05:
            regime = 3  # Crisis
        elif vix > 20 or hy_spread > 0.04:
            regime = 1  # Risk-Off
        elif t10y2y < 0:
            regime = 2  # Transitional (inverted curve)
        else:
            regime = 0  # Risk-On
        
        return {
            'regime_id': regime,
            'regime_label': self.REGIME_LABELS.get(regime, "Unknown"),
            'vix': vix,
            'hy_spread': hy_spread,
            't10y2y': t10y2y
        }


class RoughnessAnalyzer:
    """Analyze path roughness for signal confidence"""
    
    @staticmethod
    def roughness_to_confidence(roughness):
        """
        Convert roughness to confidence adjustment
        Lower roughness = more predictable = higher confidence
        """
        # Roughness typically in [0, 1]
        confidence_factor = 1.0 / (1.0 + roughness * 2)
        return confidence_factor
    
    @staticmethod
    def hurst_to_confidence(hurst):
        """
        Convert Hurst exponent to confidence
        H near 0.5 (random walk) = lower confidence
        H near 1 (trending) = higher confidence
        """
        # Deviation from 0.5
        deviation = abs(hurst - 0.5) * 2
        confidence_factor = 0.5 + deviation * 0.5
        return min(confidence_factor, 1.0)
