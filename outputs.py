"""
Output generation for ROUGH-PATH-FORECASTER
Signal generation, JSON serialization, HF upload
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from huggingface_hub import HfApi, Repository
import os


class SignalGenerator:
    """Generate final trading signals"""
    
    def __init__(self, module, benchmark, tickers):
        self.module = module
        self.benchmark = benchmark
        self.tickers = tickers
    
    def generate_signal(self, picks, macro_regime, roughness_info, signature_depth, 
                        predicted_returns=None, lookback_days=30, model_type="Ridge"):
        """
        Generate complete signal output
        """
        if not picks:
            return None
        
        top_pick = picks[0]
        second_pick = picks[1] if len(picks) > 1 else None
        third_pick = picks[2] if len(picks) > 2 else None
        
        signal = {
            "timestamp": datetime.utcnow().isoformat(),
            "engine": "ROUGH-PATH-FORECASTER",
            "version": "1.0.0",
            "module": self.module,
            "benchmark": self.benchmark,
            "etf_pick": top_pick['ticker'],
            "conviction_percentage": round(top_pick['conviction'], 1),
            "next_trading_day": (datetime.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "lookback_days": lookback_days,
            "signature_depth": signature_depth,
            "model_type": model_type,
            "predicted_return": round(top_pick['predicted_return'] * 100, 4),
            "regime": macro_regime['regime_label'],
            "benchmark_note": f"{self.benchmark} (not traded · no CASH output)",
            "macro_pills": {
                "VIX": round(macro_regime.get('vix', 0), 2),
                "T10Y2Y": round(macro_regime.get('t10y2y', 0), 2),
                "HY_SPREAD": round(macro_regime.get('hy_spread', 0), 2),
                "IG_SPREAD": round(macro_regime.get('ig_spread', 0), 2),
                "DXY": round(macro_regime.get('dxy', 0), 2)
            }
        }
        
        # Add second and third picks if available
        if second_pick:
            signal["second_pick"] = second_pick['ticker']
            signal["second_conviction"] = round(second_pick['conviction'], 1)
        
        if third_pick:
            signal["third_pick"] = third_pick['ticker']
            signal["third_conviction"] = round(third_pick['conviction'], 1)
        
        # Add roughness info if available
        if roughness_info:
            signal["path_roughness"] = round(roughness_info.get('roughness', 0), 4)
            signal["hurst_exponent"] = round(roughness_info.get('hurst', 0), 4)
        
        return signal


class JSONSerializer:
    """Serialize signals to JSON"""
    
    @staticmethod
    def serialize(signal, filepath):
        """Write signal to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(signal, f, indent=2)
        return filepath
    
    @staticmethod
    def deserialize(filepath):
        """Read signal from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)


class ParquetWriter:
    """Write results to Parquet format"""
    
    @staticmethod
    def write_predictions(predictions_df, filepath):
        """Write predictions DataFrame to Parquet"""
        predictions_df.to_parquet(filepath, index=True)
        return filepath
    
    @staticmethod
    def write_metrics(metrics_dict, filepath):
        """Write metrics to JSON"""
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        return filepath
    
    @staticmethod
    def write_window_results(results_df, filepath):
        """Write window results to Parquet"""
        results_df.to_parquet(filepath, index=True)
        return filepath


class HuggingFaceUploader:
    """Upload results to Hugging Face dataset"""
    
    def __init__(self, repo_id):
        self.repo_id = repo_id
        self.api = HfApi()
    
    def upload_file(self, local_path, remote_path, commit_message=None):
        """Upload a single file to HF dataset"""
        if commit_message is None:
            commit_message = f"Update {remote_path}"
        
        try:
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=commit_message
            )
            print(f"Uploaded {local_path} -> {remote_path}")
            return True
        except Exception as e:
            print(f"Failed to upload {local_path}: {e}")
            return False
    
    def upload_directory(self, local_dir, remote_dir, commit_message=None):
        """Upload entire directory"""
        if commit_message is None:
            commit_message = f"Update {remote_dir}"
        
        uploaded = []
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_dir)
                remote_path = f"{remote_dir}/{rel_path}"
                
                if self.upload_file(local_path, remote_path, commit_message):
                    uploaded.append(remote_path)
        
        return uploaded


class BenchmarkComparator:
    """Compare engine performance against benchmark"""
    
    @staticmethod
    def compute_alpha(engine_returns, benchmark_returns, risk_free_rate=0.03):
        """
        Compute alpha vs benchmark
        """
        # Annualized returns
        engine_ann = engine_returns.mean() * 252
        benchmark_ann = benchmark_returns.mean() * 252
        
        # Beta (simplified)
        covariance = np.cov(engine_returns, benchmark_returns)[0, 1]
        variance = np.var(benchmark_returns)
        beta = covariance / variance if variance > 0 else 1
        
        # Alpha
        alpha = engine_ann - (risk_free_rate + beta * (benchmark_ann - risk_free_rate))
        
        return alpha
    
    @staticmethod
    def compute_performance_metrics(returns, benchmark_returns=None):
        """
        Compute comprehensive performance metrics
        """
        if len(returns) == 0:
            return {}
        
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Hit rate
        hit_rate = (returns > 0).mean()
        
        metrics = {
            'annualized_return_pct': ann_return * 100,
            'annualized_vol_pct': ann_vol * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd * 100,
            'hit_rate_pct': hit_rate * 100,
            'total_days': len(returns)
        }
        
        # Alpha if benchmark provided
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            alpha = BenchmarkComparator.compute_alpha(returns, benchmark_returns)
            metrics['alpha_vs_benchmark_pct'] = alpha * 100
        
        return metrics
