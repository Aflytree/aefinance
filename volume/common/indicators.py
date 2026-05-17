import pandas as pd


def add_price_macd_columns(df: pd.DataFrame, close_col: str = "收盘") -> pd.DataFrame:
    """价格 MACD：DIF、DEA、MACD 柱（通达信习惯 2×(DIF−DEA)）。"""
    close = pd.to_numeric(df[close_col], errors="coerce")
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    out = df.copy()
    out["DIF"] = dif
    out["DEA"] = dea
    out["MACD"] = 2 * (dif - dea)
    return out


def macd_value_at(df: pd.DataFrame, idx: int, col: str) -> float | None:
    v = df.loc[idx, col]
    if pd.isna(v):
        return None
    return float(v)
