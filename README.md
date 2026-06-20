license: apache-2.0task_categories: - time-series-forecastingtags: - finance - etf - rough-paths - signature-methods - kernel-methods
ROUGH-PATH-FORECASTER
Signature kernel + Log-ODE based ETF forecasting engine for P2 ecosystem.

Overview
This engine uses rough path theory and signature methods to forecast ETF returns across Fixed Income/Commodities and Equity universes.

Key Features (v2.0)
Chen's Iteration for mathematically rigorous, high-speed truncated signature computation
Corrected Lead-Lag augmentation preserving true quadratic covariation
Signature Kernel Ridge Regression with proper post-signature standardization
Log-ODE (Neural CDE) for interpretable, path-conditioned dynamics
3D Rolling Path Construction preventing temporal data leakage
17-window expanding consensus (2008→2026 through 2024→2026)
Fixed dataset (2008-2026 YTD) with 80/10/10 split
Asset Universes
Module	Benchmark	Tickers
FI	AGG	TLT, LQD, HYG, VNQ, GLD, SLV, VCIT
Equity	SPY	SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLRE, XLB, GDX, XME, IWM
Data Sources
Input: P2SAMAPA/p2-etf-deepm-data/data/master.parquet
Output: P2SAMAPA/p2-etf-rough-path-forecaster-results
Training Modes
Fixed Dataset
Period: 2008 → 2026 YTD
Split: 80% train, 10% val, 10% test
Constructs 21-day rolling 3D paths from macro features
Shrinking Windows (17 windows)
Start years: 2008 through 2024
End year: 2026 YTD for all
Consensus weights: 60% ann return + 20% Sharpe + 20% (-max drawdown)
Installation
pip install -r requirements.txt
Engine Outputs
Output
Description
ETF pick	Selected ETF for next day
Conviction %	Confidence score (0-100)
2nd/3rd picks	Alternatives
Predicted return	μ from Kernel Ridge Regression
Signature depth	Depth used (2/3/4) via adaptive selection
Path roughness	Roughness estimate from log-signature variation
