#!/usr/bin/env python
"""
Upload all training results to Hugging Face dataset
Destination: P2SAMAPA/p2-etf-rough-path-forecaster-results
Usage:
    python upload_to_hub.py                    # Upload both modules
    python upload_to_hub.py --module fi        # Upload only FI module
    python upload_to_hub.py --module equity    # Upload only Equity module
    python upload_to_hub.py --mode metadata    # Upload only metadata.json
"""

import os
import json
import shutil
import argparse
from datetime import datetime
from huggingface_hub import HfApi
import tempfile

from constants import HF_RESULTS_REPO, FI_TICKERS, EQUITY_TICKERS, FI_BENCHMARK, EQUITY_BENCHMARK
from utils import Logger, Timer, GitHubActionsHelpers


logger = Logger("Upload-to-Hub")


def upload_module(module):
    """Upload a single module (fi or equity) to HF dataset"""
    
    logger.info(f"Uploading module: {module}")
    
    # Check if source directory exists
    source_dir = f"models_saved/{module}"
    if not os.path.exists(source_dir):
        logger.warning(f"Source directory {source_dir} not found, skipping {module}")
        return False
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy module directory to temp
        dest_module_dir = os.path.join(tmpdir, module)
        shutil.copytree(source_dir, dest_module_dir)
        
        # Upload using HfApi
        api = HfApi()
        uploaded_count = 0
        
        for root, dirs, files in os.walk(dest_module_dir):
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
        
        logger.info(f"Uploaded {uploaded_count} files for module {module}")
        return True


def upload_metadata():
    """Upload only metadata.json to HF dataset"""
    
    logger.info("Uploading metadata.json")
    
    # Create metadata
    metadata = {
        "engine": "ROUGH-PATH-FORECASTER",
        "version": "1.0.0",
        "last_updated": datetime.utcnow().isoformat(),
        "description": "Signature kernel + Log-ODE ETF forecasting results",
        "universes": {
            "fi": {
                "benchmark": FI_BENCHMARK,
                "tickers": FI_TICKERS,
                "count": len(FI_TICKERS)
            },
            "equity": {
                "benchmark": EQUITY_BENCHMARK,
                "tickers": EQUITY_TICKERS,
                "count": len(EQUITY_TICKERS)
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
        },
        "signature_params": {
            "depths": [2, 3, 4],
            "lead_lag": True,
            "basepoint": True,
            "time_channel": True
        },
        "kernel_params": {
            "type": "neumann_signature_kernel",
            "tile_size": 500,
            "dynamic_truncation_epsilon": 1e-6
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = os.path.join(tmpdir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        api = HfApi()
        try:
            api.upload_file(
                path_or_fileobj=metadata_path,
                path_in_repo="metadata.json",
                repo_id=HF_RESULTS_REPO,
                repo_type="dataset",
                commit_message="Update metadata.json"
            )
            logger.info("Uploaded: metadata.json")
            return True
        except Exception as e:
            logger.error(f"Failed to upload metadata.json: {e}")
            return False


def upload_readme():
    """Upload README.md dataset card"""
    
    logger.info("Uploading README.md")
    
    readme_content = f"""---
license: mit
task_categories:
- time-series-forecasting
- quantitative-finance
tags:
- rough-path
- signature-kernel
- log-ode
- etf-forecasting
- quantitative-finance
---

# P2 ETF Rough Path Forecaster Results

This dataset contains the output from the **ROUGH-PATH-FORECASTER** engine.

## Engine Description

Uses signature kernel methods and Log-ODE for ETF return forecasting.

- **Signature Kernel**: Neumann series expansion with dynamic truncation
- **Log-ODE**: Neural controlled differential equations on log-signature space
- **Ensemble**: Weighted combination of depths 2, 3, and 4

## Universes

### Fixed Income / Commodities
- **Benchmark**: {FI_BENCHMARK}
- **Tickers** ({len(FI_TICKERS)}): {', '.join(FI_TICKERS)}

### Equity
- **Benchmark**: {EQUITY_BENCHMARK}
- **Tickers** ({len(EQUITY_TICKERS)}): {', '.join(EQUITY_TICKERS)}

## Training Modes

### Fixed Dataset
- Period: 2008 → 2026 YTD
- Split: 80% train, 10% validation, 10% test
- Single model trained on all available data

### Shrinking Windows (17 windows)
- Start years: 2008 through 2024
- End year: 2026 YTD (all windows)
- Each window: independent model
- Consensus scoring across windows

## Consensus Weights
- 60% Annualized Return
- 20% Sharpe Ratio
- 20% (-)Max Drawdown

## Output Structure
