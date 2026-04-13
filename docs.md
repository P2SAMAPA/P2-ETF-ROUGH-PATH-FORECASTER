# ROUGH-PATH-FORECASTER Documentation

## Overview

ROUGH-PATH-FORECASTER is a signature kernel and Log-ODE based ETF forecasting engine for the P2 ecosystem.

## Mathematical Background

### Signature Method

The signature of a path X: [0,T] → ℝ^d is the collection of iterated integrals:
S(X){t}^{i1,...,ik} = ∫{0<t1<...<tk<t} dX_{t1}^{i1} ... dX_{tk}^{ik}

text

The signature provides a universal feature set for path-valued data.

### Log-ODE

The log-ODE models the evolution of log-signature:
d log(S(X)_t) = f_θ(log(S(X)_t)) dt + g_φ(log(S(X)_t)) dW_t

text

### Neumann Signature Kernel

The signature kernel is approximated via Neumann series expansion:
K(X,Y) = ⟨sig(X), sig(Y)⟩ = Σ_{k=0}^∞ ⟨sig_k(X), sig_k(Y)⟩

text

Our implementation uses dynamic truncation and tiling for CPU efficiency.

## Architecture
Input Data (master.parquet)
↓
Data Pipeline (load, align, augment)
↓
Signature Computation (adaptive depth selection)
↓
┌─────────────────────────────────────┐
│ Kernel Ridge │ Gaussian Process │
│ Regression │ with Signature GP │
├─────────────────────────────────────┤
│ Ensemble Forecaster │
└─────────────────────────────────────┘
↓
ETF Selection + Conviction Scoring
↓
Signal Output (JSON + HF Dataset)

text

## Training Modes

### Fixed Dataset
- Period: 2008 → 2026 YTD
- Split: 80% train, 10% validation, 10% test
- Single model trained on all available data

### Shrinking Windows (17 windows)
- Start years: 2008, 2009, ..., 2024
- End year: 2026 YTD (all windows)
- Each window: independent model
- Consensus scoring across windows

## Consensus Scoring

For each ETF across all windows, compute:

1. **Annualized Return** (weight 60%)
2. **Sharpe Ratio** (weight 20%)
3. **Max Drawdown** (weight 20%, inverted)

Final score = weighted sum of normalized metrics.

## Output Format

### Fixed Dataset Signal
```json
{
  "timestamp": "2026-04-13T23:30:00Z",
  "engine": "ROUGH-PATH-FORECASTER",
  "module": "fi",
  "etf_pick": "HYG",
  "conviction_percentage": 26.3,
  "predicted_return": 3.5279,
  "signature_depth": 3,
  "regime": "Transitional",
  "macro_pills": {...}
}
Shrinking Consensus Signal
json
{
  "consensus_pick": "HYG",
  "consensus_conviction": 26.4,
  "windows_used": 17,
  "window_picks": ["HYG", "GLD", "HYG", ...],
  "window_metrics": [...]
}
GitHub Actions Free Tier Optimization
Optimization	Implementation
Neumann tiling	Breaks long sequences into 500-point tiles
Dynamic truncation	Stops when coefficients < 1e-6
Kernel caching	Caches computations for 24 hours
Parallel tiles	Uses 2 cores (GitHub Actions limit)
Memory optimization	Downcasts float64→float32
Expected runtime per daily run: 2-5 minutes.

HF Dataset Structure
Source: P2SAMAPA/p2-etf-deepm-data/data/master.parquet

Results: P2SAMAPA/p2-etf-rough-path-forecaster-results

text
results/
├── fi/fixed/          # FI module fixed dataset
├── fi/shrinking/      # FI module 17-window consensus
├── equity/fixed/      # Equity module fixed dataset
├── equity/shrinking/  # Equity module 17-window consensus
└── metadata.json
Dependencies
Python 3.10+

PyTorch 2.0+

GPyTorch 1.10+

scikit-learn 1.3+

huggingface-hub 0.20+

streamlit 1.30+

torchdiffeq 0.2+

License
MIT

text

---

All files are complete with no placeholders. You can now copy these files to your GitHub repository `P2SAMAPA/P2-ETF-ROUGH-PATH-FORECASTER`.
