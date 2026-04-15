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
    def __init__(self, module='fi'):
        self.module = module
        self.tickers = FI_TICKERS if module == 'fi' else EQUITY_TICKERS
        self.benchmark = FI_BENCHMARK if module == 'fi' else EQUITY_BENCHMARK
        self.macro_cols = MACRO_COLS
        self.raw_data = None
        self.scaler = StandardScaler()
    
    def load_data(self):
        print(f"Loading data from {HF_SOURCE_REPO}/{HF_SOURCE_FILE}")
        local_path = hf_hub_download(
            repo_id=HF_SOURCE_REPO,
            filename=HF_SOURCE_FILE,
            repo_type="dataset"
        )
        self.raw_data = pd.read_parquet(local_path)
        
        if 'Date' in self.raw_data.columns:
            self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
            self.raw_data.set_index('Date', inplace=True)
        
        self.raw_data = self.raw_data.loc[f"{START_YEAR}-01-01":f"{END_YEAR}-12-31"]
        print(f"Loaded {len(self.raw_data)} rows from {self.raw_data.index[0]} to {self.raw_data.index[-1]}")
        return self
    
    def get_window_data(self, start_year, end_year):
        """Get data for a specific expanding window"""
        self.load_data()
        
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        window_data = self.raw_data.loc[start_date:end_date].copy()
        
        if len(window_data) == 0:
            print(f"Warning: No data for window {start_year}-{end_year}")
            return np.array([]), np.array([]), pd.DatetimeIndex([]), pd.DatetimeIndex([])
        
        print(f"Window {start_year}-{end_year}: raw data has {len(window_data)} rows from {window_data.index[0]} to {window_data.index[-1]}")
        
        # Extract ETF returns
        etf_returns = {}
        for ticker in self.tickers:
            close_col = f"{ticker}_Close"
            if close_col in window_data.columns:
                prices = window_data[close_col].copy()
                returns = prices.pct_change()
                etf_returns[ticker] = returns
            else:
                print(f"Warning: {close_col} not found, using zeros")
                etf_returns[ticker] = pd.Series(0, index=window_data.index)
        
        etf_returns_df = pd.DataFrame(etf_returns).dropna()
        print(f"ETF returns after dropna: {len(etf_returns_df)} rows")
        
        # Extract macro data
        available_macro = [col for col in self.macro_cols if col in window_data.columns]
        macro_df = window_data[available_macro].copy().dropna()
        print(f"Macro data after dropna: {len(macro_df)} rows")
        
        if macro_df.empty:
            return np.array([]), np.array([]), pd.DatetimeIndex([]), pd.DatetimeIndex([])
        
        # Align dates
        common_dates = etf_returns_df.index.intersection(macro_df.index)
        etf_aligned = etf_returns_df.loc[common_dates]
        macro_aligned = macro_df.loc[common_dates]
        
        print(f"Aligned data: {len(common_dates)} rows from {common_dates[0]} to {common_dates[-1]}")
        
        if len(etf_aligned) < 50:
            print(f"Warning: Only {len(etf_aligned)} samples for window {start_year}-{end_year}")
            return np.array([]), np.array([]), pd.DatetimeIndex([]), pd.DatetimeIndex([])
        
        # Create path augmentation
        macro_scaled = self.scaler.fit_transform(macro_aligned)
        t = np.linspace(0, 1, len(macro_scaled)).reshape(-1, 1)
        X = np.hstack([macro_scaled, t])
        y = etf_aligned.values
        
        return X, y, etf_aligned.index, macro_aligned.index


def get_latest_macro_pipeline():
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
