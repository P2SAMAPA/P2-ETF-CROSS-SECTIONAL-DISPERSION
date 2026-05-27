"""
Cross-Sectional Dispersion Engine
==================================
Implements the Stivers & Sun (2010) cross-sectional return dispersion
framework as an ETF ranking signal.

Key insight (Stivers & Sun 2002, 2010):
----------------------------------------
Cross-sectional dispersion (CSD) measures how spread out returns are
across assets on any given day/week/month.

  HIGH CSD → returns are very heterogeneous across assets
             → idiosyncratic factors dominate
             → momentum strategies work (winners keep winning)
             → MOMENTUM signal: overweight recent winners

  LOW  CSD → returns are highly correlated across assets
             → systematic/macro factor dominates
             → momentum strategies fail, reversal more likely
             → REVERSAL signal: overweight recent losers

This gives CSD a dual role:
  1. A direct ranking signal (momentum vs reversal regime detector)
  2. A meta-signal that modulates other engines' signals

References:
-----------
Stivers, C., Sun, L. (2002). Stock Return Comovements and Cross-Sectional
    Dispersion. Working Paper.
Stivers, C., Sun, L. (2010). Cross-Sectional Return Dispersion and the
    Magnitude of Momentum. Journal of Financial and Quantitative Analysis.
Ang, A., Hodrick, R., Xing, Y., Zhang, X. (2006). The Cross-Section of
    Volatility and Expected Returns. Journal of Finance.
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore as sp_zscore


# ── Dispersion measures ────────────────────────────────────────────────────

def cross_sectional_std(returns_row: pd.Series) -> float:
    """Standard deviation of returns across assets (Stivers & Sun 2002)."""
    clean = returns_row.dropna()
    if len(clean) < 3:
        return np.nan
    return float(clean.std())


def cross_sectional_mad(returns_row: pd.Series) -> float:
    """Mean absolute deviation — robust to return outliers."""
    clean = returns_row.dropna()
    if len(clean) < 3:
        return np.nan
    return float((clean - clean.mean()).abs().mean())


def cross_sectional_iqr(returns_row: pd.Series) -> float:
    """Interquartile range — most robust dispersion measure."""
    clean = returns_row.dropna()
    if len(clean) < 4:
        return np.nan
    return float(clean.quantile(0.75) - clean.quantile(0.25))


DISPERSION_FUNCS = {
    "std": cross_sectional_std,
    "mad": cross_sectional_mad,
    "iqr": cross_sectional_iqr,
}


# ── Volume weighting ──────────────────────────────────────────────────────

def _volume_weighted_dispersion(returns_row: pd.Series,
                                 weights: pd.Series,
                                 measure: str = "std") -> float:
    """
    Volume-weighted cross-sectional dispersion.
    Weights = dollar volume share of each ETF in the universe.
    """
    common = returns_row.dropna().index.intersection(weights.dropna().index)
    if len(common) < 3:
        return cross_sectional_std(returns_row)   # fallback to equal weight

    r = returns_row[common]
    w = weights[common]
    w = w / w.sum()   # normalise to sum = 1

    w_mean = (r * w).sum()
    if measure == "std":
        return float(np.sqrt(((r - w_mean) ** 2 * w).sum()))
    elif measure == "mad":
        return float(((r - w_mean).abs() * w).sum())
    else:  # iqr fallback to equal weight
        return cross_sectional_iqr(returns_row)


# ── Main dispersion series ────────────────────────────────────────────────

def compute_dispersion_series(prices: pd.DataFrame,
                               tickers: list,
                               window: int,
                               measure: str = "std",
                               weighting: str = "equal",
                               dvol_df: pd.DataFrame = None) -> pd.Series:
    """
    Compute time series of cross-sectional return dispersion.

    Parameters
    ----------
    prices    : DataFrame of ETF closing prices, indexed by date.
    tickers   : list of tickers to include in dispersion calculation.
    window    : return horizon in days (1=daily, 5=weekly, 21=monthly).
    measure   : 'std' | 'mad' | 'iqr'
    weighting : 'equal' | 'volume'
    dvol_df   : DataFrame of dollar volume columns (<TICKER>_DVOL_20D)
                required if weighting='volume'

    Returns
    -------
    pd.Series of dispersion values, indexed by date.
    """
    avail = [t for t in tickers if t in prices.columns]
    if len(avail) < 3:
        return pd.Series(dtype=float)

    # Window returns
    log_ret = np.log(prices[avail] / prices[avail].shift(window))

    disp_func = DISPERSION_FUNCS.get(measure, cross_sectional_std)
    results   = {}

    for date in log_ret.index:
        row = log_ret.loc[date]
        if weighting == "volume" and dvol_df is not None:
            # Build weight series: dollar volume for this date
            w = pd.Series({
                t: dvol_df.loc[date, f"{t}_DVOL_20D"]
                for t in avail
                if f"{t}_DVOL_20D" in dvol_df.columns and date in dvol_df.index
            })
            results[date] = _volume_weighted_dispersion(row, w, measure)
        else:
            results[date] = disp_func(row)

    return pd.Series(results, name=f"CSD_{window}d_{measure}").sort_index()


# ── Regime detection ──────────────────────────────────────────────────────

def dispersion_regime(disp_series: pd.Series,
                       regime_lookback: int = 252) -> pd.Series:
    """
    Z-score the dispersion series relative to a rolling lookback window.
    Returns z-score: >0 = high dispersion (momentum regime),
                     <0 = low dispersion (reversal regime).
    """
    roll_mean = disp_series.rolling(regime_lookback, min_periods=63).mean()
    roll_std  = disp_series.rolling(regime_lookback, min_periods=63).std()
    z = (disp_series - roll_mean) / (roll_std + 1e-10)
    return z.rename("CSD_zscore")


# ── ETF-level scoring ──────────────────────────────────────────────────────

def compute_etf_scores(prices: pd.DataFrame,
                        tickers: list,
                        disp_z: pd.Series,
                        momentum_window: int = 21,
                        reversal_window: int = 5,
                        w_momentum: float = 0.60,
                        w_reversal: float = 0.40) -> pd.Series:
    """
    Compute ETF-level scores by blending momentum and reversal signals,
    with the blend dynamically modulated by the dispersion regime.

    Logic:
        - In HIGH dispersion regime (z > 0):
            favour momentum = recent winners score higher
        - In LOW dispersion regime (z < 0):
            favour reversal = recent losers score higher

    The dispersion z-score acts as a continuous switch between the two signals.

    Parameters
    ----------
    prices           : ETF price DataFrame.
    tickers          : tickers to score.
    disp_z           : dispersion z-score series (from dispersion_regime).
    momentum_window  : days for momentum return signal.
    reversal_window  : days for reversal return signal.
    w_momentum       : weight on momentum component.
    w_reversal       : weight on reversal component.

    Returns
    -------
    pd.Series {ticker: score}, cross-sectionally z-scored.
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail or len(prices) < max(momentum_window, reversal_window) + 5:
        return pd.Series(dtype=float)

    # Latest dispersion regime: positive = momentum, negative = reversal
    if disp_z.empty or disp_z.dropna().empty:
        current_z = 0.0
    else:
        current_z = float(disp_z.dropna().iloc[-1])

    # Sigmoid-style regime weight: maps z-score to [0, 1]
    # regime_weight=1 → pure momentum, regime_weight=0 → pure reversal
    regime_weight = 1.0 / (1.0 + np.exp(-current_z))  # logistic function

    # Momentum signal: recent return (higher = better in momentum regime)
    mom_ret = {}
    for t in avail:
        r = np.log(prices[t].iloc[-1] / prices[t].iloc[-1 - momentum_window]) \
            if len(prices) > momentum_window else np.nan
        mom_ret[t] = r
    mom_series = pd.Series(mom_ret).dropna()

    # Reversal signal: negate short-term return (loser = better in reversal regime)
    rev_ret = {}
    for t in avail:
        r = np.log(prices[t].iloc[-1] / prices[t].iloc[-1 - reversal_window]) \
            if len(prices) > reversal_window else np.nan
        rev_ret[t] = -r   # negate: recent loser scores higher
    rev_series = pd.Series(rev_ret).dropna()

    # Cross-sectional z-score each component
    def _xsz(s):
        if len(s) < 2:
            return s
        return (s - s.mean()) / (s.std() + 1e-10)

    mom_z = _xsz(mom_series)
    rev_z = _xsz(rev_series)

    # Blend using regime weight
    scores = {}
    for t in avail:
        m = mom_z.get(t, 0.0)
        r = rev_z.get(t, 0.0)
        # regime_weight close to 1 → mostly momentum
        # regime_weight close to 0 → mostly reversal
        scores[t] = (
            regime_weight * w_momentum * m +
            (1 - regime_weight) * w_reversal * r
        )

    result = pd.Series(scores).dropna()
    if len(result) < 2:
        return result
    return _xsz(result)
