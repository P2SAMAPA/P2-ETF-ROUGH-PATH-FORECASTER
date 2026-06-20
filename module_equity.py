"""
Equity Module for ROUGH-PATH-FORECASTER

v2.0 FIXES:
  - Removed pre-signature StandardScaler (was destroying polynomial path properties).
  - Converted flat 2D feature matrices into proper rolling 3D paths for the Signature computer.
  - Signature scaling is now correctly handled inside models.py post-computation.
  - Maintains previous fixes: no future leakage, correct test window sizing, strategy metrics.
"""

import numpy as np
import pandas as pd
import traceback

from constants import EQUITY_TICKERS, EQUITY_BENCHMARK
from data_pipeline import DataPipeline
from models import EnsembleForecaster
from selection import ETFSelector, MacroRegimeContext
from outputs import SignalGenerator, BenchmarkComparator
from utils import Logger, Timer


# Configuration for path construction
PATH_WINDOW = 21  # Number of days to look back to form a single path


def _reshape_to_paths(X_2d, y_2d, window=PATH_WINDOW):
    """
    Convert a flat 2D feature matrix (n_days, n_features) into a list of 
    3D paths (n_paths, window, n_features) for the Signature computer.
    Aligns targets to the end of each path.
    """
    paths = [X_2d[i - window:i] for i in range(window, len(X_2d))]
    y_aligned = y_2d[window:]
    return paths, y_aligned


class EquityModule:
    def __init__(self):
        self.tickers  = EQUITY_TICKERS
        self.benchmark = EQUITY_BENCHMARK
        self.logger   = Logger("Equity-Module")
        self.regime_detector = MacroRegimeContext()

    # ------------------------------------------------------------------
    # Shrinking-window training
    # ------------------------------------------------------------------
    def train_shrinking(self, start_years, end_year=2026):
        self.logger.info(
            f"Training Equity module on {len(start_years)} shrinking windows"
        )
        results = []
        models  = {}

        for start_year in start_years:
            self.logger.info(f"Training window: {start_year} -> {end_year}")
            timer = Timer()
            timer.__enter__()
            try:
                pipeline = DataPipeline(module='equity')
                X_raw, y, dates, _ = pipeline.get_window_data(start_year, end_year)

                if len(X_raw) == 0:
                    self.logger.warning(
                        f"No data for window {start_year}, skipping"
                    )
                    timer.__exit__(None, None, None)
                    continue

                n_total    = len(dates)
                train_size = int(n_total * 0.80)

                if (n_total - train_size) < 21:
                    self.logger.warning(
                        f"Window {start_year} too short ({n_total} days), skipping"
                    )
                    timer.__exit__(None, None, None)
                    continue

                X_train_raw = X_raw[:train_size]
                y_train     = y[:train_size]
                X_test_raw  = X_raw[train_size:]
                y_test      = y[train_size:].copy()
                test_dates  = dates[train_size:].copy()

                # CRITICAL FIX: Convert 2D features to 3D paths
                X_train_paths, y_train = _reshape_to_paths(X_train_raw, y_train)
                X_test_paths, y_test = _reshape_to_paths(X_test_raw, y_test)

                self.logger.info(
                    f"Window {start_year}: train_paths={len(X_train_paths)}, "
                    f"test_paths={len(X_test_paths)}"
                )

                # Train model (Scaling is handled internally in models.py)
                model = EnsembleForecaster(depths=[2, 3, 4])
                model.fit(X_train_paths, y_train)
                preds = model.predict(X_test_paths).copy()

                # Compute metrics on strategy returns
                strategy_returns = _compute_strategy_returns(preds, y_test)

                ann_return = float(np.mean(strategy_returns) * 252)
                ann_vol    = float(np.std(strategy_returns) * np.sqrt(252))
                sharpe     = ann_return / ann_vol if ann_vol > 0 else 0.0

                cum       = np.cumprod(1 + strategy_returns)
                run_max   = np.maximum.accumulate(cum)
                drawdown  = (cum - run_max) / run_max
                max_dd    = float(abs(np.min(drawdown)) * 100)
                hit_rate  = float(np.mean(strategy_returns > 0) * 100)

                self.logger.info(
                    f"Window {start_year}: Days={len(X_test_paths)}, "
                    f"AnnRet={ann_return*100:.2f}%, "
                    f"MaxDD={max_dd:.2f}%, Vol={ann_vol*100:.2f}%"
                )

                results.append({
                    'start_year':      start_year,
                    'end_year':        end_year,
                    'n_days':          len(X_test_paths),
                    'model':           model,
                    'predictions':     preds,
                    'actuals':         y_test,
                    'dates':           test_dates[PATH_WINDOW:], # Align dates
                    'ann_return_pct':  ann_return * 100,
                    'ann_vol_pct':     ann_vol    * 100,
                    'max_drawdown_pct': max_dd,
                    'sharpe':          sharpe,
                    'hit_rate_pct':    hit_rate,
                })
                models[start_year] = model
                timer.__exit__(None, None, None)
                self.logger.info(
                    f"Window {start_year} complete in {timer.minutes:.2f} min"
                )

            except Exception as e:
                self.logger.error(f"Window {start_year} failed: {e}")
                traceback.print_exc()
                timer.__exit__(None, None, None)
                continue

        return {'windows': results, 'models': models}

    # ------------------------------------------------------------------
    # Fixed-dataset training
    # ------------------------------------------------------------------
    def train_fixed(self):
        self.logger.info("Training Equity module on fixed dataset")
        timer = Timer()
        timer.__enter__()
        try:
            pipeline = DataPipeline(module='equity')
            X_raw, y, dates, _ = pipeline.get_window_data(2008, 2026)

            n          = len(dates)
            train_size = int(n * 0.8)
            val_size   = int(n * 0.1)

            X_train_raw = X_raw[:train_size]
            y_train     = y[:train_size]
            X_val_raw   = X_raw[train_size:train_size + val_size]
            y_val       = y[train_size:train_size + val_size]
            X_test_raw  = X_raw[train_size + val_size:]
            y_test      = y[train_size + val_size:]

            # CRITICAL FIX: Convert 2D features to 3D paths
            X_train_paths, y_train = _reshape_to_paths(X_train_raw, y_train)
            X_val_paths, y_val = _reshape_to_paths(X_val_raw, y_val)
            X_test_paths, y_test = _reshape_to_paths(X_test_raw, y_test)

            # Combine train and validation paths
            X_combined_paths = X_train_paths + X_val_paths
            y_combined = np.vstack([y_train, y_val])

            # Train model (Scaling is handled internally in models.py)
            model = EnsembleForecaster(depths=[2, 3, 4])
            model.fit(X_combined_paths, y_combined)

            predictions  = model.predict(X_test_paths)
            strat_ret    = _compute_strategy_returns(predictions, y_test)
            bench_ret    = y_test.mean(axis=1)

            metrics = BenchmarkComparator.compute_performance_metrics(
                pd.Series(strat_ret), pd.Series(bench_ret)
            )

            timer.__exit__(None, None, None)
            self.logger.info(
                f"Fixed training complete in {timer.minutes:.2f} minutes"
            )
            return {
                'model': model, 
                'predictions': predictions, 
                'y_test': y_test,
                'metrics': metrics,
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            traceback.print_exc()
            timer.__exit__(None, None, None)
            return None


# ── helpers ──────────────────────────────────────────────────────────────────

def _compute_strategy_returns(preds: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Return the daily P&L of a long-only strategy that holds the single
    ticker the model scores highest each day.

    Parameters
    ----------
    preds  : (n_days, n_tickers)  model predicted scores / returns
    y_test : (n_days, n_tickers)  actual next-day returns

    Returns
    -------
    strategy_returns : (n_days,)
    """
    if preds.ndim == 1 or preds.shape[1] == 1:
        # Scalar prediction — fall back to universe average
        return y_test.mean(axis=1)

    best_idx = np.argmax(preds, axis=1)            # shape (n_days,)
    return np.array(
        [y_test[i, best_idx[i]] for i in range(len(y_test))]
    )


def get_equity_module():
    return EquityModule()
