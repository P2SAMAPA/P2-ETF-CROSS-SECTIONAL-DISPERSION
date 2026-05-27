import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config


def load_master_data() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=config.DATA_REPO,
        filename="master_data.parquet",
        repo_type="dataset",
        token=config.HF_TOKEN,
    )
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def prepare_prices(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    prices = pd.DataFrame(index=df.index)
    for t in tickers:
        if t in df.columns:
            s = df[t].ffill()
            if not s.isna().all():
                prices[t] = s
    return prices.dropna(how="all")


def prepare_dvol(df: pd.DataFrame, tickers: list,
                  window: int = 20) -> pd.DataFrame:
    """
    Extract dollar volume columns for the given tickers.
    Returns DataFrame with columns <TICKER>_DVOL_<window>D.
    """
    dvol = pd.DataFrame(index=df.index)
    col  = f"DVOL_{window}D"
    for t in tickers:
        full_col = f"{t}_{col}"
        if full_col in df.columns:
            dvol[full_col] = df[full_col].ffill()
    return dvol
