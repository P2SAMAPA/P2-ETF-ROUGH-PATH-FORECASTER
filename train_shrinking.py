#!/usr/bin/env python
"""
Train ROUGH-PATH-FORECASTER on shrinking windows

FIXES applied here:
  Bug 4 — window_metrics_list now reads ann_return_pct, ann_vol_pct, sharpe,
           max_drawdown_pct, and hit_rate_pct directly from the window dict,
           which is populated by the module using strategy returns
           (model's best-ticker P&L each day), not the universe average.
  Misc  — scaler is included alongside the model in each pickle so that
           inference can scale new macro inputs the same way as training.
"""

import argparse
import pickle
import os

import numpy as np
import pandas as pd

from constants import SHRINKING_START_YEARS, SHRINKING_END_YEAR
from module_fi import FIModule
from module_equity import EquityModule
from outputs import ParquetWriter
from selection import ETFSelector
from utils import Logger, Timer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module", type=str, required=True, choices=['fi', 'equity']
    )
    args = parser.parse_args()

    logger = Logger(f"Train-Shrinking-{args.module.upper()}")
    logger.info(f"Training {len(SHRINKING_START_YEARS)} shrinking windows")

    timer = Timer()
    timer.__enter__()

    try:
        if args.module == 'fi':
            module    = FIModule()
            save_dir  = "models_saved/fi/shrinking"
        else:
            module    = EquityModule()
            save_dir  = "models_saved/equity/shrinking"

        result = module.train_shrinking(SHRINKING_START_YEARS, SHRINKING_END_YEAR)
        os.makedirs(save_dir, exist_ok=True)

        window_metrics_list    = []
        window_picks_list      = []
        window_convictions_list = []

        for window in result['windows']:
            start_year  = window['start_year']
            predictions = window.get('predictions')

            # ── Best-ticker pick from model predictions ────────────────────
            if predictions is not None and len(predictions) > 0:
                last_pred = predictions[-1]

                if predictions.ndim > 1:
                    last_pred = predictions[-1]          # shape (n_tickers,)
                else:
                    last_pred = predictions[-1]          # scalar

                # Pad / truncate to match ticker list length
                if isinstance(last_pred, (int, float, np.floating)):
                    last_pred = np.ones(len(module.tickers)) * float(last_pred)
                elif len(last_pred) > len(module.tickers):
                    last_pred = last_pred[:len(module.tickers)]
                elif len(last_pred) < len(module.tickers):
                    last_pred = np.pad(
                        last_pred, (0, len(module.tickers) - len(last_pred))
                    )

                selector     = ETFSelector(module.tickers, module.benchmark)
                scores       = selector.compute_net_scores(last_pred)
                top_pick     = scores.iloc[0]['ticker']
                top_conviction = scores.iloc[0]['conviction']
            else:
                top_pick       = "N/A"
                top_conviction = 0

            window_picks_list.append(top_pick)
            window_convictions_list.append(top_conviction)

            # ── Bug 4 fix: read pre-computed strategy metrics from module ──
            # The module now fills these with _compute_strategy_returns(),
            # so they reflect the model's actual daily pick performance.
            window_metrics_list.append({
                'start_year':       start_year,
                'n_days':           window['n_days'],
                'ann_return_pct':   window['ann_return_pct'],
                'ann_vol_pct':      window.get('ann_vol_pct', 0.0),
                'sharpe':           window.get('sharpe', 0.0),
                'max_drawdown_pct': window['max_drawdown_pct'],
                'hit_rate_pct':     window.get('hit_rate_pct', 0.0),
                # alpha: strategy return minus a rough risk-free rate proxy
                'ann_alpha_pct':    window['ann_return_pct'] - 3.0,
                'positive_years':   int(window['ann_return_pct'] > 0),
            })

            logger.info(
                f"Window {start_year}: Days={window['n_days']}, "
                f"AnnRet={window['ann_return_pct']:.2f}%, "
                f"MaxDD={window['max_drawdown_pct']:.2f}%"
            )

        # ── Consensus ──────────────────────────────────────────────────────
        weighted_picks: dict = {}
        for pick, conv in zip(window_picks_list, window_convictions_list):
            if pick != "N/A":
                weighted_picks[pick] = weighted_picks.get(pick, 0) + conv

        sorted_picks = sorted(
            weighted_picks.items(), key=lambda x: x[1], reverse=True
        )

        # ── DataFrames ─────────────────────────────────────────────────────
        metrics_df = pd.DataFrame(window_metrics_list)

        picks_df = pd.DataFrame({
            'start_year': [w['start_year'] for w in result['windows']],
            'pick':       window_picks_list,
            'conviction': window_convictions_list,
        })

        n_windows = max(len(window_picks_list), 1)
        consensus_df = pd.DataFrame([{
            'consensus_pick':
                sorted_picks[0][0] if sorted_picks else None,
            'consensus_conviction':
                sorted_picks[0][1] / n_windows if sorted_picks else 0,
            'second_pick':
                sorted_picks[1][0] if len(sorted_picks) > 1 else None,
            'third_pick':
                sorted_picks[2][0] if len(sorted_picks) > 2 else None,
        }])

        # ── Persist models (model + scaler bundled together) ───────────────
        for start_year, model in result['models'].items():
            # Find matching window to get scaler
            matching = [w for w in result['windows'] if w['start_year'] == start_year]
            scaler   = matching[0].get('scaler') if matching else None

            model_path = os.path.join(
                save_dir, f"model_window_{start_year}.pkl"
            )
            with open(model_path, 'wb') as f:
                pickle.dump({'model': model, 'scaler': scaler}, f)

        # ── Persist result parquets ────────────────────────────────────────
        window_results_list = [
            {
                'start_year': w['start_year'],
                'end_year':   w['end_year'],
                'n_days':     w['n_days'],
            }
            for w in result['windows']
        ]

        ParquetWriter.write_window_results(
            pd.DataFrame(window_results_list),
            os.path.join(save_dir, "window_results.parquet"),
        )
        ParquetWriter.write_predictions(
            metrics_df,
            os.path.join(save_dir, "window_metrics.parquet"),
        )
        ParquetWriter.write_predictions(
            picks_df,
            os.path.join(save_dir, "window_picks.parquet"),
        )
        ParquetWriter.write_predictions(
            consensus_df,
            os.path.join(save_dir, "consensus.parquet"),
        )

        timer.__exit__(None, None, None)
        logger.info(f"Training completed in {timer.minutes:.2f} minutes")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        timer.__exit__(None, None, None)
        raise


if __name__ == "__main__":
    main()
