#!/usr/bin/env python
"""
Upload all training results to Hugging Face dataset
Destination: P2SAMAPA/p2-etf-rough-path-forecaster-results
"""

import os
import json
import shutil
from datetime import datetime
from huggingface_hub import HfApi, Repository
import tempfile

from constants import HF_RESULTS_REPO
from utils import Logger, Timer, GitHubActionsHelpers


def upload_results():
    """Upload all results to HF dataset"""
    logger = Logger("Upload-to-Hub")
    
    is_ci = GitHubActionsHelpers.is_github_actions()
    
    # Check if HF token is available
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        logger.warning("HF_TOKEN not found in environment. Skipping upload.")
        if is_ci:
            GitHubActionsHelpers.set_output("upload_status", "skipped_no_token")
        return
    
    with Timer() as t:
        # Create temporary directory for upload structure
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure matching HF dataset
            # Structure: fi/fixed/, fi/shrinking/, equity/fixed/, equity/shrinking/
            
            for module in ['fi', 'equity']:
                for mode in ['fixed', 'shrinking']:
                    source_dir = f"models_saved/{module}/{mode}"
                    if not os.path.exists(source_dir):
                        logger.warning(f"Source directory {source_dir} not found, skipping")
                        continue
                    
                    # Copy files to temp dir
                    dest_dir = os.path.join(tmpdir, module, mode)
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    for file in os.listdir(source_dir):
                        src_file = os.path.join(source_dir, file)
                        if os.path.isfile(src_file):
                            shutil.copy2(src_file, os.path.join(dest_dir, file))
            
            # Create metadata.json
            metadata = {
                "engine": "ROUGH-PATH-FORECASTER",
                "version": "1.0.0",
                "last_updated": datetime.utcnow().isoformat(),
                "description": "Signature kernel + Log-ODE ETF forecasting results",
                "universes": {
                    "fi": {
                        "benchmark": "AGG",
                        "tickers": ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "VCIT"]
                    },
                    "equity": {
                        "benchmark": "SPY",
                        "tickers": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB", "GDX", "XME", "IWM"]
                    }
                },
                "training_modes": ["fixed", "shrinking"],
                "shrinking_windows": {
                    "start_years": list(range(2008, 2025)),
                    "end_year": 2026,
                    "num_windows": 17
                },
                "consensus_weights": {
                    "annualized_return": 0.60,
                    "sharpe_ratio": 0.20,
                    "max_drawdown": 0.20
                }
            }
            
            metadata_path = os.path.join(tmpdir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create README.md for dataset card
            readme_content = f"""---
license: mit
task_categories:
- time-series-forecasting
- quantitative-finance
---

# P2 ETF Rough Path Forecaster Results

This dataset contains the output from the ROUGH-PATH-FORECASTER engine.

## Engine Description

Uses signature kernel methods and Log-ODE for ETF return forecasting.

## Universe

- **FI/Commodities**: AGG benchmark with 7 tickers
- **Equity**: SPY benchmark with 15 tickers

## Training Modes

- **Fixed**: 2008-2026 YTD (80/10/10 split)
- **Shrinking**: 17 expanding windows (2008→2026 through 2024→2026)

## Consensus Weights

- 60% Annualized Return
- 20% Sharpe Ratio
- 20% (-)Max Drawdown

## Output Structure
fi/
├── fixed/
│ ├── model.pkl
│ ├── predictions.parquet
│ ├── actuals.parquet
│ └── metrics.json
└── shrinking/
├── model_window_*.pkl
├── window_results.parquet
├── consensus.parquet
├── window_picks.parquet
└── window_metrics.parquet

equity/
└── (same structure as fi/)

## Last Updated

{datetime.utcnow().isoformat()}
"""
            
            readme_path = os.path.join(tmpdir, "README.md")
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            # Upload using HfApi
            api = HfApi()
            
            # Upload each file individually
            uploaded_count = 0
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, tmpdir)
                    
                    try:
                        api.upload_file(
                            path_or_fileobj=local_path,
                            path_in_repo=rel_path,
                            repo_id=HF_RESULTS_REPO,
                            repo_type="dataset",
                            commit_message=f"Update {rel_path}"
                        )
                        uploaded_count += 1
                        logger.info(f"Uploaded: {rel_path}")
                    except Exception as e:
                        logger.error(f"Failed to upload {rel_path}: {e}")
            
            logger.info(f"Successfully uploaded {uploaded_count} files to {HF_RESULTS_REPO}")
    
    logger.info(f"Upload completed in {t.minutes:.2f} minutes")
    
    if is_ci:
        GitHubActionsHelpers.set_output("upload_status", "success")
        GitHubActionsHelpers.set_output("files_uploaded", str(uploaded_count))


def main():
    upload_results()


if __name__ == "__main__":
    main()
