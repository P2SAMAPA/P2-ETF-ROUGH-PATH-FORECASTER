# ROUGH-PATH-FORECASTER

Signature kernel + Log-ODE based ETF forecasting engine for P2 ecosystem.

## Overview

This engine uses rough path theory and signature kernel methods to forecast ETF returns across Fixed Income/Commodities and Equity universes.

### Key Features
- **Signature Kernel** with Neumann series expansion (CPU-optimized)
- **Log-ODE** for interpretable path dynamics
- **Gaussian Process** on signature space
- **17-window expanding consensus** (2008→2026 through 2024→2026)
- **Fixed dataset** (2008-2026 YTD) with 80/10/10 split

### Asset Universes

| Module | Benchmark | Tickers |
|--------|-----------|---------|
| FI | AGG | TLT, LQD, HYG, VNQ, GLD, SLV, VCIT |
| Equity | SPY | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLRE, XLB, GDX, XME, IWM |

### Data Sources
- **Input**: `P2SAMAPA/p2-etf-deepm-data/data/master.parquet`
- **Output**: `P2SAMAPA/p2-etf-rough-path-forecaster-results`

### Training Modes

#### Fixed Dataset
- Period: 2008 → 2026 YTD
- Split: 80% train, 10% val, 10% test

#### Shrinking Windows (17 windows)
- Start years: 2008 through 2024
- End year: 2026 YTD for all
- Consensus weights: 60% ann return + 20% Sharpe + 20% (-max drawdown)

## Installation

```bash
pip install -r requirements.txt
Engine Outputs
Output	Description
ETF pick	Selected ETF for next day
Conviction %	Confidence score (0-100)
2nd/3rd picks	Alternatives
Predicted return	μ from GP or Kernel Ridge
Signature depth	Depth used (2/3/4)
Path roughness	Roughness estimate from log-ODE
Kernel alignment	Signature kernel alignment score
