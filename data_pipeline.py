"""
Data pipeline for ROUGH-PATH-FORECASTER
Loads master.parquet from HF, processes ETF and macro data

FIXES:
  Bug 1 — load_data() no longer slices to global START_YEAR/END_YEAR.
           get_window_data() is the only place the date range is applied,
           so each shrinking window actually sees a different slice of data.

  Bug 2 — StandardScaler is no longer fit inside get_window_data().
           Raw (unscaled) X is returned; callers must fit the scaler on
           X_train only and transform X_test, preventing future leakage.

  Bug 5 — (Equity-specific) XLRE did not exist until 2015-10-07.
           Using DataFrame.dropna() on the ETF return matrix dropped ALL
           rows before XLRE's inception, causing every window starting
           2008-2015 to silently collapse to the same post-2015 data.

           Fix: missing tickers are forward-filled from their first valid
           date and then back-filled with zeros before that date.
           This preserves the full date range for every window while
           ensuring no NaN values propagate into training.
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
        """Load the full master parquet — NO date slicing here.
        Slicing is deferred to get_window_data() so each window is independent.
        """
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

        # Apply only a broad outer boundary — do NOT narrow to a specific
        # start_year here; that is done per-window in get_window_data().
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
        """Return raw (unscaled) X and y for the window [start_year, end_year].

        Returns
        -------
        X          : np.ndarray  shape (n, n_macro+1)  — unscaled macro + time
        y          : np.ndarray  shape (n, n_tickers)  — ETF daily returns
        etf_index  : DatetimeIndex aligned with X/y rows
        macro_index: DatetimeIndex (same as etf_index after alignment)

        NOTE: callers are responsible for scaling X.  Fit StandardScaler on
        X_train only, then transform both X_train and X_test.
        """
        self.load_data()

        # ── Bug 1 fix: slice to THIS window's dates ───────────────────────
        start_date  = f"{start_year}-01-01"
        end_date    = f"{end_year}-12-31"
        window_data = self.raw_data.loc[start_date:end_date].copy()

        if len(window_data) == 0:
            print(f"Warning: No data for window {start_year}-{end_year}")
            return (
                np.array([]), np.array([]),
                pd.DatetimeIndex([]), pd.DatetimeIndex([]),
            )

        print(
            f"Window {start_year}-{end_year}: {len(window_data)} rows "
            f"{window_data.index[0]} → {window_data.index[-1]}"
        )

        # ── ETF returns ───────────────────────────────────────────────────
        # Bug 5 fix: build returns column-by-column, then handle missing
        # tickers (e.g. XLRE pre-2015) via forward-fill + zero-fill instead
        # of a blanket dropna() that silently nukes entire early windows.
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

        # Forward-fill up to 5 days for tickers with sporadic NaNs
        # (handles late-inception tickers whose first few rows are NaN
        # after pct_change).  Then back-fill remaining NaNs (pre-inception)
        # with 0 so no rows are dropped.
        etf_returns_df = (
            etf_returns_df
            .ffill(limit=5)
            .fillna(0.0)
        )

        # Drop only the very first row which is always NaN from pct_change
        etf_returns_df = etf_returns_df.iloc[1:]

        # ── Macro features ────────────────────────────────────────────────
        available_macro = [c for c in self.macro_cols if c in window_data.columns]
        macro_df = window_data[available_macro].copy()

        # Forward-fill macro too (macro series sometimes have weekend gaps)
        macro_df = macro_df.ffill(limit=5).fillna(method='bfill')

        # Drop rows where ALL macro values are still NaN (genuinely empty)
        macro_df = macro_df.dropna(how='all')

        if macro_df.empty:
            print(f"Warning: No macro data for window {start_year}-{end_year}")
            return (
                np.array([]), np.array([]),
                pd.DatetimeIndex([]), pd.DatetimeIndex([]),
            )

        # ── Align dates ───────────────────────────────────────────────────
        common_dates  = etf_returns_df.index.intersection(macro_df.index)
        etf_aligned   = etf_returns_df.loc[common_dates]
        macro_aligned = macro_df.loc[common_dates]

        # Final safety: drop any remaining NaN rows (should be none now)
        valid_mask    = (~etf_aligned.isna().any(axis=1)) & (~macro_aligned.isna().any(axis=1))
        etf_aligned   = etf_aligned[valid_mask]
        macro_aligned = macro_aligned[valid_mask]
        common_dates  = etf_aligned.index

        print(
            f"Aligned: {len(common_dates)} rows "
            f"{common_dates[0]} → {common_dates[-1]}"
        )

        if len(etf_aligned) < 50:
            print(
                f"Warning: Only {len(etf_aligned)} samples for "
                f"window {start_year}-{end_year} — skipping"
            )
            return (
                np.array([]), np.array([]),
                pd.DatetimeIndex([]), pd.DatetimeIndex([]),
            )

        # ── Build X (raw, unscaled) — macro + time index ──────────────────
        # Bug 2 fix: do NOT call scaler.fit_transform here.
        # Callers must fit on X_train and transform X_test separately.
        macro_raw = macro_aligned.values.astype(float)
        t         = np.linspace(0, 1, len(macro_raw)).reshape(-1, 1)
        X         = np.hstack([macro_raw, t])
        y         = etf_aligned.values

        return X, y, etf_aligned.index, macro_aligned.index


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
