import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

import config
import data_manager
from dispersion_engine import (
    compute_dispersion_series,
    dispersion_regime,
    compute_etf_scores,
)


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, dict):  return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [convert_to_serializable(v) for v in obj]
    return obj


def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set"); return

    df    = data_manager.load_master_data()
    today = datetime.now().strftime("%Y-%m-%d")

    all_results = {}
    all_windows = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Cross-Sectional Dispersion) ===")

        prices = data_manager.prepare_prices(df, tickers)
        dvol   = data_manager.prepare_dvol(df, tickers, window=20)
        available = [t for t in tickers if t in prices.columns]

        if not available or len(prices) < config.REGIME_LOOKBACK + 30:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            all_windows[universe_name] = {"windows": {}}
            continue

        best_per_etf   = {}
        window_results = {}
        disp_summary   = {}

        for win in config.WINDOWS:
            if len(prices) < win + config.REGIME_LOOKBACK + 5:
                print(f"  Skipping window {win}d")
                continue

            print(f"  Processing window {win}d...")

            try:
                # Compute dispersion time series
                disp_series = compute_dispersion_series(
                    prices=prices,
                    tickers=available,
                    window=win,
                    measure=config.DISPERSION_MEASURE,
                    weighting=config.WEIGHTING,
                    dvol_df=dvol,
                )
                if disp_series.dropna().empty:
                    print(f"  No dispersion values for {win}d")
                    continue

                # Compute regime z-score
                disp_z = dispersion_regime(disp_series, config.REGIME_LOOKBACK)

                current_disp   = float(disp_series.dropna().iloc[-1])
                current_z      = float(disp_z.dropna().iloc[-1]) if not disp_z.dropna().empty else 0.0
                regime_label   = "MOMENTUM" if current_z > 0 else "REVERSAL"

                print(f"  CSD={current_disp:.5f}  z={current_z:.3f}  regime={regime_label}")

                disp_summary[win] = {
                    "csd_value":   current_disp,
                    "csd_zscore":  current_z,
                    "regime":      regime_label,
                    "csd_pct":     float((disp_series.dropna() <= current_disp).mean() * 100),
                }

                # Compute ETF scores
                scores = compute_etf_scores(
                    prices=prices,
                    tickers=available,
                    disp_z=disp_z,
                    momentum_window=config.MOMENTUM_WINDOW,
                    reversal_window=config.REVERSAL_WINDOW,
                    w_momentum=config.W_MOMENTUM,
                    w_reversal=config.W_REVERSAL,
                )
                if scores.empty:
                    continue

                score_dict = {t: float(s) for t, s in scores.items() if not np.isnan(s)}
                window_results[win] = score_dict

                top3 = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  Top 3: {[(t, round(s, 4)) for t, s in top3]}")

                for etf, score in score_dict.items():
                    if etf not in best_per_etf or score > best_per_etf[etf][0]:
                        best_per_etf[etf] = (float(score), win)

            except Exception as e:
                print(f"  Window {win}d failed: {e}")
                import traceback; traceback.print_exc()
                continue

        # ── Fallback ──────────────────────────────────────────────────────
        if not best_per_etf:
            print("  Falling back to historical mean return")
            for t in available:
                mean_ret = np.log(prices[t] / prices[t].shift(1)).iloc[-252:].mean()
                if not np.isnan(mean_ret):
                    best_per_etf[t] = (float(mean_ret), 0)

        if not best_per_etf:
            all_results[universe_name] = {"top_etfs": []}
            all_windows[universe_name] = {"windows": {}}
            continue

        # ── Tab 1 ─────────────────────────────────────────────────────────
        full_scores = {
            t: {"score": float(s), "best_window": int(w)}
            for t, (s, w) in best_per_etf.items()
        }
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs    = [
            {"ticker": t, "csd_score": float(s), "best_window": int(w)}
            for t, (s, w) in sorted_etfs[:config.TOP_N]
        ]
        print(f"  Top {config.TOP_N}: {[e['ticker'] for e in top_etfs]}")

        all_results[universe_name] = {
            "top_etfs":      top_etfs,
            "full_scores":   full_scores,
            "window_results":window_results,
            "disp_summary":  disp_summary,
            "run_date":      today,
        }

        # ── Tab 2 ─────────────────────────────────────────────────────────
        windows_tab2 = {}
        for win, sd in window_results.items():
            sw = sorted(sd.items(), key=lambda x: x[1], reverse=True)
            windows_tab2[str(win)] = {
                "top_etfs":    [{"ticker": t, "csd_score": float(s)} for t, s in sw[:config.TOP_N]],
                "full_ranking":[{"ticker": t, "csd_score": float(s)} for t, s in sw],
                "regime":      disp_summary.get(win, {}).get("regime", "UNKNOWN"),
                "csd_zscore":  disp_summary.get(win, {}).get("csd_zscore", 0.0),
            }
        all_windows[universe_name] = {"windows": windows_tab2, "run_date": today}

    # ── Write and push ────────────────────────────────────────────────────
    Path("results").mkdir(exist_ok=True)

    tab1_path = Path(f"results/cross_sectional_dispersion_{today}.json")
    with open(tab1_path, "w") as f:
        json.dump(convert_to_serializable({
            "run_date": today, "universes": all_results,
        }), f, indent=2)

    tab2_path = Path(f"results/cross_sectional_dispersion_windows_{today}.json")
    with open(tab2_path, "w") as f:
        json.dump(convert_to_serializable({
            "run_date": today, "universes": all_windows,
        }), f, indent=2)

    import push_results
    push_results.push_daily_result(tab1_path)
    push_results.push_daily_result(tab2_path)

    print(f"\n=== Cross-Sectional Dispersion Engine complete ===")
    print(f"  Tab 1: {tab1_path.name}")
    print(f"  Tab 2: {tab2_path.name}")


if __name__ == "__main__":
    main()
