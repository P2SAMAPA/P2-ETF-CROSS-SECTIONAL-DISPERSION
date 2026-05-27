import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-cross-sectional-dispersion-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

# ── Dispersion parameters ──────────────────────────────────────────────────
# Windows over which cross-sectional dispersion is computed
WINDOWS = [1, 5, 21, 63]          # daily, weekly, monthly, quarterly

# Dispersion measure: 'std' | 'mad' | 'iqr'
#   std = cross-sectional standard deviation of returns (Stivers & Sun 2002)
#   mad = mean absolute deviation (robust to outliers)
#   iqr = interquartile range (most robust)
DISPERSION_MEASURE = "std"

# Weighting: 'equal' | 'volume'
#   equal  = equal weight all ETFs in universe
#   volume = weight each ETF's return by its dollar volume share (DVOL_20D)
#            captures where money is actually moving
WEIGHTING = "volume"

# Regime lookback: z-score dispersion relative to this window
REGIME_LOOKBACK = 252

# Signal construction:
#   HIGH dispersion → momentum signal (trend-following)
#   LOW  dispersion → reversal signal (mean-reversion)
# Score blends both: high dispersion = favour recent winner ETFs
#                    low  dispersion = favour recent loser ETFs (contrarian)
MOMENTUM_WINDOW  = 21    # recent return used as momentum/reversal base signal
REVERSAL_WINDOW  = 5     # short-term reversal signal

# Blend weights
W_MOMENTUM = 0.60
W_REVERSAL = 0.40

TOP_N = 3
