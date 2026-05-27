# P2-ETF-CROSS-SECTIONAL-DISPERSION

**Cross-Sectional Dispersion Engine** — part of the P2Quant Engine Suite (v15).

Implements the Stivers & Sun (2010) cross-sectional return dispersion framework as a dual momentum/reversal ETF ranking signal. Uses volume-weighted dispersion computed from closing prices and ADV data in the master dataset.

---

## Mathematical Foundation

### Cross-Sectional Dispersion (CSD)

CSD measures how spread out returns are across assets on any given period:

```
CSD_t = std({ r_i,t : i = 1..N })          equal-weighted
CSD_t = sqrt( sum_i w_i (r_i,t - r̄_t)^2 ) volume-weighted
```

where w_i = DVOL_i / sum_j DVOL_j (dollar volume share).

### Regime Detection (Stivers & Sun 2010)

```
z_t = (CSD_t - rolling_mean(CSD, 252d)) / rolling_std(CSD, 252d)
```

| z-score | Regime | Signal |
|---------|--------|--------|
| z > 0 | HIGH dispersion — idiosyncratic factors dominate | **Momentum** |
| z < 0 | LOW dispersion — systematic factor dominates | **Reversal** |

### Composite Score

```
regime_weight = sigmoid(z_t) ∈ (0, 1)

score_i = regime_weight × w_mom × z(momentum_i)
        + (1 − regime_weight) × w_rev × z(reversal_i)
```

where momentum_i = 21d return, reversal_i = −5d return (negate for contrarian).

---

## Repository Structure

```
P2-ETF-CROSS-SECTIONAL-DISPERSION/
├── config.py               # Universe, windows, measure, weights
├── dispersion_engine.py    # CSD computation, regime detection, scoring
├── data_manager.py         # HF data loader, prices + DVOL extraction
├── trainer.py              # Main runner
├── streamlit_app.py        # Two-tab Streamlit dashboard
├── push_results.py         # HuggingFace upload
├── us_calendar.py          # Next trading day
├── requirements.txt
└── .github/workflows/daily.yml
```

---

## Data Requirements

Uses `master_data.parquet` from `P2SAMAPA/fi-etf-macro-signal-master-data`.

| Column type | Example | Used for |
|-------------|---------|---------|
| ETF closing prices | `SPY`, `TLT` | Return computation |
| Dollar volume | `SPY_DVOL_20D` | Volume weighting of CSD |

Both column types are now present after the June 2026 seeding update.

---

## Windows

| Window | Duration | Signal character |
|--------|----------|-----------------|
| 1d | Daily | Very noisy; regime detection dominates |
| 5d | Weekly | Short-term momentum/reversal |
| **21d** | Monthly | **Recommended primary signal** |
| 63d | Quarterly | Structural momentum/reversal regime |

---

## Dispersion Measures

| Measure | Formula | Use case |
|---------|---------|---------|
| `std` | Cross-sectional standard deviation | Standard (Stivers & Sun) |
| `mad` | Mean absolute deviation | Robust to outliers |
| `iqr` | Interquartile range | Most robust |

Default: `std` with `volume` weighting.

---

## Output Files

| File | Tab | Content |
|------|-----|---------|
| `cross_sectional_dispersion_YYYY-MM-DD.json` | Tab 1 | Best window per ETF + regime summary |
| `cross_sectional_dispersion_windows_YYYY-MM-DD.json` | Tab 2 | Full ranking at every window |

---

## Streamlit App

**Tab 1** — Best window per ETF with regime label (MOMENTUM / REVERSAL) and CSD percentile per window.

**Tab 2** — Window selector (1d/5d/21d/63d) with regime indicator and full ranking table.

---

## Setup

1. Create GitHub repo `P2-ETF-CROSS-SECTIONAL-DISPERSION`
2. Create HuggingFace dataset `P2SAMAPA/p2-etf-cross-sectional-dispersion-results`
3. Add `HF_TOKEN` as a GitHub Actions secret
4. Push all files to `main`
5. Actions → Run workflow

---

## References

- Stivers, C., Sun, L. (2002). *Stock Return Comovements and Cross-Sectional Dispersion*. Working Paper.
- Stivers, C., Sun, L. (2010). *Cross-Sectional Return Dispersion and the Magnitude of Momentum*. JFQA.
- Ang, A., Hodrick, R., Xing, Y., Zhang, X. (2006). *The Cross-Section of Volatility and Expected Returns*. JF.

**HuggingFace Results:** `P2SAMAPA/p2-etf-cross-sectional-dispersion-results`
**Part of:** P2Quant Engine Suite · P2SAMAPA
