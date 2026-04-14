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
        trade_cost = self.trade_cost_bps / 10000
        
        # Ensure predicted_returns is 1D array of length len(tickers)
        if len(predicted_returns.shape) > 1:
            predicted_returns = predicted_returns.flatten()
        
        # Trim to match tickers length
        if len(predicted_returns) > len(self.tickers):
            predicted_returns = predicted_returns[:len(self.tickers)]
        elif len(predicted_returns) < len(self.tickers):
            # Pad with zeros
            predicted_returns = np.pad(predicted_returns, (0, len(self.tickers) - len(predicted_returns)))
        
        net_returns = predicted_returns - trade_cost
        
        # Calculate conviction: higher for higher predicted return
        # Normalize to 0-100 range
        min_ret = net_returns.min()
        max_ret = net_returns.max()
        
        if max_ret - min_ret > 1e-6:
            conviction = (net_returns - min_ret) / (max_ret - min_ret) * 100
        else:
            conviction = np.ones(len(net_returns)) * 50  # Default 50% if all equal
        
        # If uncertainties provided, adjust conviction (higher uncertainty = lower conviction)
        if predicted_uncertainties is not None:
            if len(predicted_uncertainties) > len(self.tickers):
                predicted_uncertainties = predicted_uncertainties[:len(self.tickers)]
            elif len(predicted_uncertainties) < len(self.tickers):
                predicted_uncertainties = np.pad(predicted_uncertainties, (0, len(self.tickers) - len(predicted_uncertainties)))
            
            # Normalize uncertainties
            std_norm = (predicted_uncertainties - predicted_uncertainties.min()) / (predicted_uncertainties.max() - predicted_uncertainties.min() + 1e-6)
            # Lower uncertainty = higher conviction adjustment
            uncertainty_factor = 1 - std_norm * 0.3
            conviction = conviction * uncertainty_factor
        
        # Create results DataFrame
        results = pd.DataFrame({
            'ticker': self.tickers,
            'predicted_return': predicted_returns * 100,  # Convert to percentage
            'net_return': net_returns * 100,
            'conviction': conviction
        })
        
        # Sort by net return (descending)
        results = results.sort_values('net_return', ascending=False)
        
        return results
    
    def select_picks(self, predicted_returns, predicted_uncertainties=None, date=None):
        """
        Select top N ETFs as picks
        Returns all top N picks regardless of conviction threshold
        """
        scores = self.compute_net_scores(predicted_returns, predicted_uncertainties)
        
        picks = []
        for i in range(min(self.top_n, len(scores))):
            row = scores.iloc[i]
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
        # Signal strength component (higher absolute return = higher confidence)
        return_magnitude = min(abs(predicted_return) * 10, 1.0)
        
        # Uncertainty component (lower uncertainty = higher confidence)
        if uncertainty > 0:
            uncertainty_component = 1.0 / (1.0 + uncertainty * 5)
        else:
            uncertainty_component = 1.0
        
        # Combine
        confidence = (return_magnitude * 0.5 + uncertainty_component * 0.5) * 100
        
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
    
    def get_regime(self, macro_values):
        """
        Determine current regime based on macro values
        
        Rules (checked in order - most severe first):
        - Crisis: VIX > 30 AND HY spread > 0.06 (most severe)
        - Risk-Off: VIX > 20 OR HY spread > 0.04
        - Transitional: T10Y2Y < 0 (inverted yield curve)
        - Risk-On: everything else
        """
        vix = macro_values.get('VIX', 15)
        hy_spread = macro_values.get('HY_SPREAD', 0.03)
        t10y2y = macro_values.get('T10Y2Y', 0.5)
        
        # Crisis: most severe - check first
        if vix > 30 and hy_spread > 0.06:
            regime = 3  # Crisis
        # Risk-Off: VIX > 20 OR HY spread > 0.04
        elif vix > 20 or hy_spread > 0.04:
            regime = 1  # Risk-Off
        # Transitional: inverted yield curve (T10Y2Y < 0)
        elif t10y2y < 0:
            regime = 2  # Transitional
        else:
            regime = 0  # Risk-On
        
        return {
            'regime_id': regime,
            'regime_label': self.REGIME_LABELS.get(regime, "Unknown"),
            'vix': vix,
            'hy_spread': hy_spread,
            't10y2y': t10y2y,
            'dxy': macro_values.get('DXY', 100),
            'ig_spread': macro_values.get('IG_SPREAD', 0.8)
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
