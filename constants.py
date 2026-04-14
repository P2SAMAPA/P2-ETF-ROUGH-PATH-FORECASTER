"""
Constants for ROUGH-PATH-FORECASTER
"""

# HF Repositories
HF_SOURCE_REPO = "P2SAMAPA/p2-etf-deepm-data"
HF_SOURCE_FILE = "data/master.parquet"
HF_RESULTS_REPO = "P2SAMAPA/p2-etf-rough-path-forecaster-results"

# Macro columns from source data
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Universes - EXCLUDING benchmarks (AGG and SPY are benchmarks only, NOT tradable)
FI_TICKERS = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "VCIT"]
FI_BENCHMARK = "AGG"

EQUITY_TICKERS = [
    "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
    "XLP", "XLU", "XLRE", "XLB", "GDX", "XME", "IWM"
]
EQUITY_BENCHMARK = "SPY"

# Shrinking windows start years (2008 through 2024 = 17 windows)
SHRINKING_START_YEARS = list(range(2008, 2025))
SHRINKING_END_YEAR = 2026

# Consensus weights
CONSENSUS_WEIGHTS = {
    "annualized_return": 0.60,
    "sharpe_ratio": 0.20,
    "max_drawdown": 0.20
}

# Training splits
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Signature parameters
SIGNATURE_DEPTHS = [2, 3, 4]
SIGNATURE_LEAD_LAG = True
SIGNATURE_BASEPOINT = True
SIGNATURE_TIME_CHANNEL = True
SIGNATURE_TRUNCATION_TOLERANCE = 1e-6

# Kernel parameters
KERNEL_TILE_SIZE = 500
KERNEL_DYNAMIC_TRUNCATION_EPSILON = 1e-6
KERNEL_PARALLEL_TILES = 2

# Log-ODE parameters
LOG_ODE_SOLVER = "dop5"
LOG_ODE_VECTOR_FIELD_HIDDEN = [64, 64]
LOG_ODE_RTOL = 1e-5
LOG_ODE_ATOL = 1e-6

# Selection parameters
TRADE_COST_BPS = 12
MIN_CONVICTION_FOR_TRADE = 5.0  # Minimum 5% conviction to trade
TOP_N_PICKS = 3

# Date range
START_YEAR = 2008
END_YEAR = 2026

# Random seeds
RANDOM_SEED = 42
