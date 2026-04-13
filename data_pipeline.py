"""
Data pipeline for ROUGH-PATH-FORECASTER
Loads master.parquet from HF, processes ETF and macro data
"""

import numpy as np
import pandas as pd
from datetime import datetime
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
from constants import (
    HF_SOURCE_REPO, HF_SOURCE_FILE, MACRO_COLS,
    FI_TICKERS, EQUITY_TICKERS, FI_BENCHMARK, EQUITY_BENCHMARK,
    START_YEAR, END_YEAR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)


class DataPipeline:
    """Load and preprocess data for rough path forecasting"""
    
    def __init__(self, module='fi'):
        """
        Args:
            module: 'fi' for Fixed Income/Commodities, 'equity' for Equity
        """
        self.module = module
        
        if module == 'fi':
            self.tickers = FI_TICKERS
            self.benchmark = FI_BENCHMARK
        else:
            self.tickers = EQUITY_TICKERS
            self.benchmark = EQUITY_BENCHMARK
        
        self.macro_cols = MACRO_COLS
        self.raw_data = None
        self.processed_data = None
        self.scaler = StandardScaler()
    
    def load_data(self):
        """Load master.parquet from Hugging Face"""
        print(f"Loading data from {HF_SOURCE_REPO}/{HF_SOURCE_FILE}")
        
        local_path = hf_hub_download(
            repo_id=HF_SOURCE_REPO,
            filename=HF_SOURCE_FILE,
            repo_type="dataset"
        )
        
        self.raw_data = pd.read_parquet(local_path)
        
        # Ensure datetime index
        if 'datetime' in self.raw_data.columns:
            self.raw_data['datetime'] = pd.to_datetime(self.raw_data['datetime'])
            self.raw_data.set_index('datetime', inplace=True)
        
        # Filter to date range
        self.raw_data = self.raw_data.loc[f"{START_YEAR}-01-01":f"{END_YEAR}-12-31"]
        
        print(f"Loaded {len(self.raw_data)} rows from {self.raw_data.index[0]} to {self.raw_data.index[-1]}")
        return self
    
    def extract_etf_returns(self):
        """Extract ETF returns for all tickers in universe"""
        returns = []
        
        for ticker in self.tickers:
            # Try different column naming conventions
            col_candidates = [ticker, f"{ticker}_return", f"{ticker}_adj_close", ticker.lower()]
            
            for col in col_candidates:
                if col in self.raw_data.columns:
                    returns_col = self.raw_data[col]
                    break
            else:
                # If not found, compute from price if available
                price_col = f"{ticker}_adj_close" if f"{ticker}_adj_close" in self.raw_data.columns else ticker
                if price_col in self.raw_data.columns:
                    returns_col = self.raw_data[price_col].pct_change()
                else:
                    print(f"Warning: {ticker} not found in data")
                    returns_col = pd.Series(0, index=self.raw_data.index)
            
            returns.append(returns_col)
        
        etf_returns = pd.DataFrame(returns, index=self.tickers).T
        etf_returns = etf_returns.dropna()
        
        return etf_returns
    
    def extract_macro_data(self):
        """Extract macro columns"""
        available_macro = [col for col in self.macro_cols if col in self.raw_data.columns]
        
        if not available_macro:
            raise ValueError(f"None of macro columns {self.macro_cols} found in data")
        
        macro_data = self.raw_data[available_macro].copy()
        macro_data = macro_data.dropna()
        
        return macro_data
    
    def align_data(self, etf_returns, macro_data):
        """Align ETF returns and macro data on common dates"""
        common_dates = etf_returns.index.intersection(macro_data.index)
        
        etf_aligned = etf_returns.loc[common_dates]
        macro_aligned = macro_data.loc[common_dates]
        
        return etf_aligned, macro_aligned
    
    def create_path_augmentation(self, macro_data):
        """Create augmented path for signature computation"""
        # Standardize macro data
        macro_scaled = self.scaler.fit_transform(macro_data)
        macro_scaled = pd.DataFrame(macro_scaled, index=macro_data.index, columns=macro_data.columns)
        
        # Add time channel (normalized)
        t = np.linspace(0, 1, len(macro_scaled))
        macro_scaled['time'] = t
        
        return macro_scaled.values
    
    def train_val_test_split(self, X, y):
        """Split data into train/val/test sets"""
        n = len(X)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_data(self):
        """Main method to get processed data for training"""
        self.load_data()
        
        etf_returns = self.extract_etf_returns()
        macro_data = self.extract_macro_data()
        
        etf_aligned, macro_aligned = self.align_data(etf_returns, macro_data)
        
        # Create path augmentations
        X = self.create_path_augmentation(macro_aligned)
        y = etf_aligned.values
        
        # Get train/val/test splits
        splits = self.train_val_test_split(X, y)
        
        return {
            'train': splits[0],
            'val': splits[1],
            'test': splits[2],
            'macro_dates': macro_aligned.index,
            'etf_dates': etf_aligned.index,
            'tickers': self.tickers,
            'benchmark': self.benchmark,
            'macro_cols': macro_aligned.columns.tolist()
        }
    
    def get_window_data(self, start_year, end_year):
        """Get data for a specific expanding window"""
        # Filter to window period
        window_data = self.raw_data.loc[f"{start_year}-01-01":f"{end_year}-12-31"]
        
        # Extract features as above
        etf_returns = []
        for ticker in self.tickers:
            col_candidates = [ticker, f"{ticker}_return", f"{ticker}_adj_close"]
            for col in col_candidates:
                if col in window_data.columns:
                    returns_col = window_data[col]
                    break
            else:
                price_col = f"{ticker}_adj_close" if f"{ticker}_adj_close" in window_data.columns else ticker
                if price_col in window_data.columns:
                    returns_col = window_data[price_col].pct_change()
                else:
                    returns_col = pd.Series(0, index=window_data.index)
            etf_returns.append(returns_col)
        
        etf_returns_df = pd.DataFrame(etf_returns, index=self.tickers).T.dropna()
        
        available_macro = [col for col in self.macro_cols if col in window_data.columns]
        macro_df = window_data[available_macro].dropna()
        
        common_dates = etf_returns_df.index.intersection(macro_df.index)
        etf_aligned = etf_returns_df.loc[common_dates]
        macro_aligned = macro_df.loc[common_dates]
        
        X = self.create_path_augmentation(macro_aligned)
        y = etf_aligned.values
        
        return X, y, etf_aligned.index, macro_aligned.index


def get_latest_macro_pipeline():
    """Get latest macro values for display"""
    pipeline = DataPipeline('fi')
    pipeline.load_data()
    
    latest = {}
    for col in MACRO_COLS:
        if col in pipeline.raw_data.columns:
            latest[col] = pipeline.raw_data[col].iloc[-1]
    
    return latest
