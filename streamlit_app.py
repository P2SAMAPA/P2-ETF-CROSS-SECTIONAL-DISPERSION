import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Cross-Sectional Dispersion Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#1a5276; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #1a5276; }
.etf-card    { background:linear-gradient(135deg,#1a5276 0%,#2c3e50 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#117a65 0%,#1a5276 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.regime-mom  { background:#d5f5e3; border-radius:10px; padding:0.6rem 1rem;
               color:#1e8449; font-weight:700; display:inline-block; }
.regime-rev  { background:#fde8e8; border-radius:10px; padding:0.6rem 1rem;
               color:#922b21; font-weight:700; display:inline-block; }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Cross-Sectional Dispersion Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Stivers & Sun (2010) · CSD regime detection · '
    'High dispersion = momentum · Low dispersion = reversal · '
    'Volume-weighted · Multi-window</div>',
    unsafe_allow_html=True)

st.sidebar.markdown("## 📊 CSD Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(f"**Measure:** {config.DISPERSION_MEASURE.upper()}")
st.sidebar.markdown(f"**Weighting:** {config.WEIGHTING.capitalize()}")

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        return [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                          detail=True, recursive=True)
                if f["type"] == "file"]
    except Exception as e:
        return [f"Error: {e}"]


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


files     = list_repo_files()
tab1_path = find_latest(files, "cross_sectional_dispersion_2")
tab2_path = find_latest(files, "cross_sectional_dispersion_windows_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Cross-Sectional Dispersion Signal")

    with st.expander("📖 How CSD Works", expanded=True):
        st.markdown("""
**Cross-Sectional Dispersion (CSD)** measures how spread out returns are across ETFs.

| CSD Level | Regime | Signal |
|-----------|--------|--------|
| **High** (z > 0) | Idiosyncratic factors dominate | **Momentum** — overweight recent winners |
| **Low** (z < 0)  | Systematic/macro factor dominates | **Reversal** — overweight recent losers |

**Score** = regime_weight × momentum_signal + (1 − regime_weight) × reversal_signal

where regime_weight = sigmoid(CSD z-score) — a continuous switch between momentum and reversal.

**Volume weighting**: each ETF's return is weighted by its 20-day dollar volume share,
so high-turnover ETFs drive the dispersion measure more than low-turnover ones.
        """)

    for universe_name, uni_data in universes1.items():
        top_etfs    = uni_data.get("top_etfs", [])
        disp_summary= uni_data.get("disp_summary", {})
        if not top_etfs:
            continue

        st.markdown(
            f'<div class="uni-title">{universe_name.replace("_"," ").title()}</div>',
            unsafe_allow_html=True)

        # Show current regime for each window
        if disp_summary:
            rcols = st.columns(len(disp_summary))
            for idx, (win, ds) in enumerate(sorted(disp_summary.items())):
                with rcols[idx]:
                    regime = ds.get("regime", "?")
                    z      = ds.get("csd_zscore", 0.0)
                    csd    = ds.get("csd_value", 0.0)
                    pct    = ds.get("csd_pct", 50.0)
                    cls    = "regime-mom" if regime == "MOMENTUM" else "regime-rev"
                    st.markdown(
                        f"**{win}d window**<br>"
                        f'<span class="{cls}">{regime}</span><br>'
                        f"CSD={csd:.4f} | z={z:.2f} | {pct:.0f}th pct",
                        unsafe_allow_html=True)
            st.markdown("")

        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">CSD score = {etf['csd_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window','N/A')}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {universe_name}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = [{"ETF": t,
                         "CSD Score": info.get("score", info) if isinstance(info, dict) else info,
                         "Best Window (d)": info.get("best_window","N/A") if isinstance(info, dict) else "N/A"}
                        for t, info in full.items()]
                df = pd.DataFrame(rows).sort_values("CSD Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(f"Run date: {data1.get('run_date','?')} · "
               "Stivers & Sun (2010) · Volume-weighted CSD · "
               "Scores are cross-sectional z-scores.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data available.")
        st.stop()

    win_labels = {1: "1d (daily)", 5: "5d (weekly)",
                  21: "21d (monthly)", 63: "63d (quarterly)"}
    default_idx  = win_options.index(21) if 21 in win_options else 0
    selected_win = st.selectbox(
        "Select return window for dispersion calculation",
        options=win_options,
        index=default_idx,
        format_func=lambda w: win_labels.get(w, f"{w}d"),
    )
    win_key = str(selected_win)

    st.markdown(f"### Rankings at **{win_labels.get(selected_win, f'{selected_win}d')}** window")

    for universe_name in ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]:
        label = {"FI_COMMODITIES": "🏦 FI & Commodities",
                 "EQUITY_SECTORS": "📈 Equity Sectors",
                 "COMBINED":       "🌐 Combined"}.get(universe_name, universe_name)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        # Regime indicator
        regime = win_data.get("regime", "?")
        z      = win_data.get("csd_zscore", 0.0)
        cls    = "regime-mom" if regime == "MOMENTUM" else "regime-rev"
        st.markdown(
            f'Current regime: <span class="{cls}">{regime}</span> '
            f'(CSD z-score = {z:.2f})',
            unsafe_allow_html=True)
        st.markdown("")

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">CSD score = {etf['csd_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d · {regime}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df = pd.DataFrame(rows)
                df.columns = ["ETF", "CSD Score"]
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
