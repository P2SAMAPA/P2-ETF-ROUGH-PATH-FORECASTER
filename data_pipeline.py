"""
Data pipeline for ROUGH-PATH-FORECASTER
Loads master.parquet from HF, processes ETF and macro data
Data format: {TICKER}_Open, {TICKER}_High, {TICKER}_Low, {TICKER}_Close, {TICKER}_Volume
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
        
        # Print first few columns for debugging
        print(f"Columns in data: {list(self.raw_data.columns)[:10]}...")
        
        # Ensure datetime index
        if 'datetime' in self.raw_data.columns:
            self.raw_data['datetime'] = pd.to_datetime(self.raw_data['datetime'])
            self.raw_data.set_index('datetime', inplace=True)
        
        # Filter to date range
        self.raw_data = self.raw_data.loc[f"{START_YEAR}-01-01":f"{END_YEAR}-12-31"]
        
        print(f"Loaded {len(self.raw_data)} rows from {self.raw_data.index[0]} to {self.raw_data.index[-1]}")
        return self
    
    def extract_etf_returns(self):
        """Extract ETF returns from OHLCV data using Close prices"""
        returns_dict = {}
        
        for ticker in self.tickers:
            # Look for Close price column
            close_col = f"{ticker}_Close"
            
            if close_col in self.raw_data.columns:
                prices = self.raw_data[close_col]
                # Calculate returns
                returns = prices.pct_change()
                returns_dict[ticker] = returns
                print(f"Found {ticker} -> {close_col}")
            else:
                # Try alternative naming
                alt_cols = [f"{ticker}_close", f"{ticker}_adj_close", f"{ticker}_price"]
                found = False
                for alt in alt_cols:
                    if alt in self.raw_data.columns:
                        prices = self.raw_data[alt]
                        returns = prices.pct_change()
                        returns_dict[ticker] = returns
                        print(f"Found {ticker} -> {alt}")
                        found = True
                        break
                
                if not found:
                    print(f"Warning: {ticker}_Close not found in data. Using zeros.")
                    returns_dict[ticker] = pd.Series(0, index=self.raw_data.index)
        
        # Create DataFrame
        etf_returns = pd.DataFrame(returns_dict)
        etf_returns = etf_returns.dropna()
        
        return etf_returns
    
    def extract_macro_data(self):
        """Extract macro columns (already in correct format)"""
        available_macro = [col for col in self.macro_cols if col in self.raw_data.columns]
        
        if not available_macro:
            print(f"Warning: None of macro columns {self.macro_cols} found in data")
            print(f"Available columns sample: {list(self.raw_data.columns)[:20]}")
            return pd.DataFrame(index=self.raw_data.index)
        
        macro_data = self.raw_data[available_macro].copy()
        macro_data = macro_data.dropna()
        
        print(f"Macro columns found: {available_macro}")
        
        return macro_data
    
    def align_data(self, etf_returns, macro_data):
        """Align ETF returns and macro data on common dates"""
        common_dates = etf_returns.index.intersection(macro_data.index)
        
        etf_aligned = etf_returns.loc[common_dates]
        macro_aligned = macro_data.loc[common_dates]
        
        print(f"Aligned data: {len(common_dates)} common dates")
        
        return etf_aligned, macro_aligned
    
    def create_path_augmentation(self, macro_data):
        """Create augmented path for signature computation"""
        if macro_data.empty or len(macro_data) < 2:
            # Return dummy path
            return np.zeros((10, len(self.macro_cols) + 1))
        
        # Handle NaN values
        macro_data = macro_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Standardize macro data
        macro_scaled = self.scaler.fit_transform(macro_data)
        macro_scaled = pd.DataFrame(macro_scaled, index=macro_data.index, columns=macro_data.columns)
        
        # Add time channel (normalized)
        t = np.linspace(0, 1, len(macro_scaled)).reshape(-1, 1)
        macro_with_time = np.hstack([macro_scaled.values, t])
        
        return macro_with_time
    
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
        
        print(f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_data(self):
        """Main method to get processed data for training"""
        self.load_data()
        
        etf_returns = self.extract_etf_returns()
        macro_data = self.extract_macro_data()
        
        if macro_data.empty:
            print("Error: No macro data available")
            return None
        
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
            'macro_cols': macro_aligned.columns.tolist() if not macro_aligned.empty else []
        }
    
    def get_window_data(self, start_year, end_year):
        """Get data for a specific expanding window"""
        self.load_data()
        
        # Filter to window period
        window_data = self.raw_data.loc[f"{start_year}-01-01":f"{end_year}-12-31"]
        
        # Extract ETF returns from Close prices
        etf_returns = {}
        for ticker in self.tickers:
            close_col = f"{ticker}_Close"
            if close_col in window_data.columns:
                prices = window_data[close_col]
                returns = prices.pct_change()
                etf_returns[ticker] = returns
            else:
                # Try alternative
                alt_cols = [f"{ticker}_close", f"{ticker}_adj_close"]
                found = False
                for alt in alt_cols:
                    if alt in window_data.columns:
                        prices = window_data[alt]
                        returns = prices.pct_change()
                        etf_returns[ticker] = returns
                        found = True
                        break
                if not found:
                    etf_returns[ticker] = pd.Series(0, index=window_data.index)
        
        etf_returns_df = pd.DataFrame(etf_returns)
        etf_returns_df = etf_returns_df.dropna()
        
        # Extract macro
        available_macro = [col for col in self.macro_cols if col in window_data.columns]
        macro_df = window_data[available_macro].copy()
        macro_df = macro_df.dropna()
        
        if macro_df.empty:
            print(f"Warning: No macro data for window {start_year}-{end_year}")
            return np.array([]), np.array([]), pd.DatetimeIndex([]), pd.DatetimeIndex([])
        
        # Align
        common_dates = etf_returns_df.index.intersection(macro_df.index)
        etf_aligned = etf_returns_df.loc[common_dates]
        macro_aligned = macro_df.loc[common_dates]
        
        if len(etf_aligned) < 10:
            print(f"Warning: Only {len(etf_aligned)} samples for window {start_year}-{end_year}")
        
        # Handle NaN in macro
        macro_aligned = macro_aligned.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
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
            val = pipeline.raw_data[col].iloc[-1]
            latest[col] = float(val) if not pd.isna(val) else 0.0
        else:
            latest[col] = 0.0
    
    return latest
