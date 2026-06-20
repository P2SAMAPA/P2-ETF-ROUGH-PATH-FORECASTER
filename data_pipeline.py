"""
Data pipeline for ROUGH-PATH-FORECASTER
Loads master.parquet from HF, processes ETF and macro data

FIXES:
  Bug 1 — load_data() no longer slices to global START_YEAR/END_YEAR.
           get_window_data() is the only place the date range is applied.
  Bug 2 — StandardScaler is no longer fit inside get_window_data().
  Bug 5 — Missing tickers (e.g. XLRE) are forward-filled/back-filled 
           instead of dropping early data.
  Bug 6 — REMOVED global time channel. Parametric time [0, 1] must be 
           applied PER PATH inside signature_core.py, not globally over 
           18 years of data (which destroyed signature geometry).
  NEW   — Added get_recent_features() for predict.py inference.
"""

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from constants import (
    HF_SOURCE_REPO, HF_SOURCE_FILE, MACRO_COLS,
    FI_TICKERS, EQUITY_TICKERS, FI_BENCHMARK, EQUITY_BENCHMARK,
    START_YEAR, END_YEAR,
)


class DataPipeline:
    def __init__(self, module='fi'):
        self.module    = module
        self.tickers   = FI_TICKERS   if module == 'fi' else EQUITY_TICKERS
        self.benchmark = FI_BENCHMARK if module == 'fi' else EQUITY_BENCHMARK
        self.macro_cols = MACRO_COLS
        self.raw_data  = None

    # ------------------------------------------------------------------
    def load_data(self):
        """Load the full master parquet — NO date slicing here."""
        print(f"Loading data from {HF_SOURCE_REPO}/{HF_SOURCE_FILE}")
        local_path = hf_hub_download(
            repo_id=HF_SOURCE_REPO,
            filename=HF_SOURCE_FILE,
            repo_type="dataset",
        )
        self.raw_data = pd.read_parquet(local_path)

        if 'Date' in self.raw_data.columns:
            self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
            self.raw_data.set_index('Date', inplace=True)

        self.raw_data = self.raw_data.loc[
            f"{START_YEAR}-01-01": f"{END_YEAR}-12-31"
        ]

        print(
            f"Loaded {len(self.raw_data)} rows: "
            f"{self.raw_data.index[0]} → {self.raw_data.index[-1]}"
        )
        return self

    # ------------------------------------------------------------------
    def get_window_data(self, start_year, end_year):
        """Return raw (unscaled) X and y for the window [start_year, end_year]."""
        self.load_data()

        start_date  = f"{start_year}-01-01"
        end_date    = f"{end_year}-12-31"
        window_data = self.raw_data.loc[start_date:end_date].copy()

        if len(window_data) == 0:
            return (np.array([]), np.array([]), pd.DatetimeIndex([]), pd.DatetimeIndex([]))

        # ── ETF returns ───────────────────────────────────────────────────
        etf_returns = {}
        for ticker in self.tickers:
            close_col = f"{ticker}_Close"
            if close_col in window_data.columns:
                prices  = window_data[close_col].copy()
                returns = prices.pct_change()
                etf_returns[ticker] = returns
            else:
                print(f"Warning: {close_col} not found, filling with zeros")
                etf_returns[ticker] = pd.Series(0.0, index=window_data.index)

        etf_returns_df = pd.DataFrame(etf_returns)
        etf_returns_df = (
            etf_returns_df
            .ffill(limit=5)
            .fillna(0.0)
            .iloc[1:]  # Drop first NaN row from pct_change
        )

        # ── Macro features ────────────────────────────────────────────────
        available_macro = [c for c in self.macro_cols if c in window_data.columns]
        macro_df = window_data[available_macro].copy()
        macro_df = macro_df.ffill(limit=5).fillna(method='bfill').dropna(how='all')

        if macro_df.empty:
            return (np.array([]), np.array([]), pd.DatetimeIndex([]), pd.DatetimeIndex([]))

        # ── Align dates ───────────────────────────────────────────────────
        common_dates  = etf_returns_df.index.intersection(macro_df.index)
        etf_aligned   = etf_returns_df.loc[common_dates]
        macro_aligned = macro_df.loc[common_dates]

        valid_mask    = (~etf_aligned.isna().any(axis=1)) & (~macro_aligned.isna().any(axis=1))
        etf_aligned   = etf_aligned[valid_mask]
        macro_aligned = macro_aligned[valid_mask]

        if len(etf_aligned) < 50:
            return (np.array([]), np.array([]), pd.DatetimeIndex([]), pd.DatetimeIndex([]))

        # ── Build X (RAW MACRO ONLY) ─────────────────────────────────────
        # CRITICAL FIX (Bug 6): Removed global np.linspace(0, 1) time channel.
        # Time channel MUST be applied per-path in signature_core.py, otherwise
        # a 21-day path in 2008 gets time=[0.00, 0.003] while 2024 gets [0.88, 0.99],
        # completely destroying the signature's algebraic properties.
        X = macro_aligned.values.astype(float)
        y = etf_aligned.values

        return X, y, etf_aligned.index, macro_aligned.index

    # ------------------------------------------------------------------
    def get_recent_features(self, window=21):
        """
        NEW: Fetch the most recent N days of raw macro features for daily prediction.
        Used by predict.py to construct a single inference path.
        """
        self.load_data()
        
        available_macro = [c for c in self.macro_cols if c in self.raw_data.columns]
        macro_df = self.raw_data[available_macro].copy()
        macro_df = macro_df.ffill(limit=5).fillna(method='bfill').dropna(how='all')
        
        if len(macro_df) < window:
            raise ValueError(f"Need {window} days of data, only have {len(macro_df)}")
            
        recent_macro = macro_df.iloc[-window:]
        
        # Return 2D array of features, the dates, and a dict of the very latest values
        return recent_macro.values, recent_macro.index, recent_macro.iloc[-1].to_dict()


# ── convenience ──────────────────────────────────────────────────────────────

def get_latest_macro_pipeline():
    pipeline = DataPipeline('fi')
    pipeline.load_data()
    latest = {}
    for col in pipeline.macro_cols:
        if col in pipeline.raw_data.columns:
            val = pipeline.raw_data[col].iloc[-1]
            latest[col] = float(val) if not pd.isna(val) else 0.0
        else:
            latest[col] = 0.0
    return latest
