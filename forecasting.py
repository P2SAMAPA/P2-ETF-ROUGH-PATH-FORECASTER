"""
Forecasting module for ROUGH-PATH-FORECASTER
Handles rolling windows, expanding windows, and uncertainty quantification
"""

import numpy as np
import pandas as pd
from datetime import datetime
from signature_core import SignatureComputer, AdaptiveDepthSelector
from models import EnsembleForecaster, SignatureGPModel


class RollingWindowForecaster:
    """Rolling window predictions"""
    
    def __init__(self, window_days=30, step_days=1):
        self.window_days = window_days
        self.step_days = step_days
    
    def generate_windows(self, X_paths, y_returns, dates):
        """Generate rolling window indices"""
        windows = []
        n = len(X_paths)
        
        for start in range(0, n - self.window_days, self.step_days):
            end = start + self.window_days
            if end <= n:
                windows.append({
                    'train_start': start,
                    'train_end': end,
                    'test_idx': end,
                    'train_dates': dates[start:end],
                    'test_date': dates[end] if end < n else None
                })
        
        return windows
    
    def predict_rolling(self, model_class, X_paths, y_returns, dates, **model_kwargs):
        """Generate rolling predictions"""
        windows = self.generate_windows(X_paths, y_returns, dates)
        predictions = []
        
        for window in windows:
            X_train = X_paths[window['train_start']:window['train_end']]
            y_train = y_returns[window['train_start']:window['train_end']]
            X_test = X_paths[window['test_idx']:window['test_idx']+1]
            
            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]
            
            predictions.append({
                'date': window['test_date'],
                'prediction': pred,
                'window_start': window['train_dates'][0],
                'window_end': window['train_dates'][-1]
            })
        
        return pd.DataFrame(predictions)


class ExpandingWindowConsensus:
    """Expanding windows with consensus scoring"""
    
    def __init__(self, start_years, end_year, consensus_weights):
        """
        Args:
            start_years: List of start years for each window
            end_year: End year for all windows
            consensus_weights: Dict with 'annualized_return', 'sharpe_ratio', 'max_drawdown'
        """
        self.start_years = start_years
        self.end_year = end_year
        self.consensus_weights = consensus_weights
    
    def compute_window_metrics(self, predictions_df, returns_df):
        """
        Compute per-window performance metrics
        """
        metrics = {}
        
        for start_year in self.start_years:
            window_key = f"window_{start_year}"
            
            # Filter to window period
            mask = (predictions_df.index.year >= start_year) & (predictions_df.index.year <= self.end_year)
            window_preds = predictions_df[mask]
            window_returns = returns_df.loc[window_preds.index]
            
            if len(window_preds) == 0:
                metrics[window_key] = self._empty_metrics()
                continue
            
            # Compute metrics
            ann_return = window_returns.mean() * 252
            ann_vol = window_returns.std() * np.sqrt(252)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            max_dd = self._max_drawdown(window_returns)
            hit_rate = (window_returns > 0).mean()
            
            metrics[window_key] = {
                'start_year': start_year,
                'n_days': len(window_returns),
                'ann_return_pct': ann_return * 100,
                'ann_vol_pct': ann_vol * 100,
                'sharpe': sharpe,
                'max_drawdown_pct': max_dd * 100,
                'hit_rate_pct': hit_rate * 100,
                'ann_alpha_pct': (ann_return - 0.03) * 100,  # Assuming 3% risk-free
                'positive_years': self._positive_years_count(window_returns)
            }
        
        return pd.DataFrame(metrics).T
    
    def _max_drawdown(self, returns):
        """Compute maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _positive_years_count(self, returns):
        """Count number of positive years"""
        if len(returns) == 0:
            return 0
        yearly_returns = returns.groupby(returns.index.year).mean()
        return (yearly_returns > 0).sum()
    
    def _empty_metrics(self):
        """Empty metrics for missing windows"""
        return {
            'start_year': 0, 'n_days': 0, 'ann_return_pct': 0,
            'ann_vol_pct': 0, 'sharpe': 0, 'max_drawdown_pct': 0,
            'hit_rate_pct': 0, 'ann_alpha_pct': 0, 'positive_years': 0
        }
    
    def compute_consensus_scores(self, predictions_df, returns_df):
        """
        Compute consensus scores across all windows using weighted metrics
        """
        metrics_df = self.compute_window_metrics(predictions_df, returns_df)
        
        # Normalize metrics for scoring
        ann_return_norm = (metrics_df['ann_return_pct'] - metrics_df['ann_return_pct'].min()) / \
                          (metrics_df['ann_return_pct'].max() - metrics_df['ann_return_pct'].min() + 1e-8)
        
        sharpe_norm = (metrics_df['sharpe'] - metrics_df['sharpe'].min()) / \
                      (metrics_df['sharpe'].max() - metrics_df['sharpe'].min() + 1e-8)
        
        # For max drawdown, lower is better
        dd_norm = (metrics_df['max_drawdown_pct'].max() - metrics_df['max_drawdown_pct']) / \
                  (metrics_df['max_drawdown_pct'].max() - metrics_df['max_drawdown_pct'].min() + 1e-8)
        
        # Weighted score
        scores = (self.consensus_weights['annualized_return'] * ann_return_norm +
                  self.consensus_weights['sharpe_ratio'] * sharpe_norm +
                  self.consensus_weights['max_drawdown'] * dd_norm)
        
        metrics_df['consensus_score'] = scores
        metrics_df = metrics_df.sort_values('consensus_score', ascending=False)
        
        return metrics_df
    
    def get_consensus_pick(self, predictions_df, returns_df, top_n=3):
        """
        Get top N picks based on consensus scoring
        """
        scores = self.compute_consensus_scores(predictions_df, returns_df)
        
        if len(scores) == 0:
            return []
        
        # Aggregate picks across top windows
        pick_counts = {}
        for idx, row in scores.head(top_n * 2).iterrows():
            # Get the pick for this window from predictions
            window_preds = predictions_df[predictions_df.index.year >= row['start_year']]
            if len(window_preds) > 0:
                # Most recent prediction for this window
                pick = window_preds.iloc[-1]['pick']
                pick_counts[pick] = pick_counts.get(pick, 0) + 1
        
        if not pick_counts:
            return []
        
        # Sort by frequency
        sorted_picks = sorted(pick_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [p[0] for p in sorted_picks[:top_n]]


class UncertaintyQuantifier:
    """Uncertainty quantification for predictions"""
    
    @staticmethod
    def gaussian_uncertainty(predictions, historical_errors, confidence_level=0.95):
        """
        Estimate uncertainty assuming Gaussian errors
        """
        from scipy import stats
        
        error_std = np.std(historical_errors)
        z_score = stats.norm.ppf(confidence_level)
        
        margin = z_score * error_std
        lower = predictions - margin
        upper = predictions + margin
        
        return lower, upper, error_std
    
    @staticmethod
    def conformal_prediction(model, X_calib, y_calib, X_test, alpha=0.1):
        """
        Conformal prediction for finite-sample coverage guarantees
        """
        # Compute non-conformity scores on calibration set
        calib_preds = model.predict(X_calib)
        scores = np.abs(y_calib - calib_preds)
        
        # Get quantile
        q = np.percentile(scores, (1 - alpha) * 100)
        
        # Apply to test set
        test_preds = model.predict(X_test)
        lower = test_preds - q
        upper = test_preds + q
        
        return lower, upper, test_preds
