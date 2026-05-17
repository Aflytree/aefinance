from typing import Any, Dict, List

import pandas as pd

from .constants import (
    MA20_TO_MA5_FACTOR,
    MA5_TO_MA10_MAX_RATIO,
    MA5_TO_MA10_MIN_RATIO,
    MAX_SIGNAL_DAY_PCT_CHG,
    MIN_SIGNAL_DAY_PCT_CHG,
)
from .data import fetch_stock_df, normalize_stock_code


def method3_price_ma_bullish_expanding(df: pd.DataFrame, row_index: int = -1) -> bool:
    """
    方法：price_ma5 > price_ma20，且 MA5 与 MA10 的差值较上一交易日扩大。
    """
    if df is None or df.empty or "收盘" not in df.columns:
        return False
    close = pd.to_numeric(df["收盘"], errors="coerce")
    if row_index < 0:
        row_index = len(df) + row_index
    if row_index < 1 or row_index >= len(close):
        return False

    price_ma5 = close.rolling(window=5).mean()
    price_ma10 = close.rolling(window=10).mean()
    price_ma20 = close.rolling(window=20).mean()
    price_ma30 = close.rolling(window=30).mean()
    price_ma60 = close.rolling(window=60).mean()
    price_ma120 = close.rolling(window=120).mean()
    price_ma250 = close.rolling(window=250).mean()
    gap = price_ma5 - price_ma10
    gap_yesterday = gap.shift(1)

    p5 = price_ma5.iloc[row_index]
    p10 = price_ma10.iloc[row_index]
    p20 = price_ma20.iloc[row_index]
    p30 = price_ma30.iloc[row_index]
    p60 = price_ma60.iloc[row_index]
    p120 = price_ma120.iloc[row_index]
    p250 = price_ma250.iloc[row_index]
    g0 = gap.iloc[row_index]
    g1 = gap_yesterday.iloc[row_index]
    if any(pd.isna(v) for v in (p5, p10, p20, p30, p60, p120, p250, g0, g1)):
        return False
        
    ma_order_ok = p5 > p20
    gap_expanding = g0 > g1
    return bool(ma_order_ok and gap_expanding)


def prepare_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame | None:
    """清洗 OHLCV 并计算量能/价格均线，供单日命中判定使用。"""
    if df is None or df.empty:
        return None
    if not {"成交量", "收盘", "涨跌幅", "日期"}.issubset(df.columns):
        return None
    data = df.copy()
    data["成交量"] = pd.to_numeric(data["成交量"], errors="coerce")
    data["收盘"] = pd.to_numeric(data["收盘"], errors="coerce")
    data["涨跌幅"] = pd.to_numeric(data["涨跌幅"], errors="coerce")
    data = data.dropna(subset=["成交量", "收盘", "涨跌幅"]).reset_index(drop=True)
    if len(data) < 250:
        return None
    data["MA5量能"] = data["成交量"].rolling(5).mean()
    data["MA10量能"] = data["成交量"].rolling(10).mean()
    data["MA20量能"] = data["成交量"].rolling(20).mean()
    data["价格MA10"] = data["收盘"].rolling(10).mean()
    return data


def check_hit_at_row(
    data: pd.DataFrame,
    row_index: int,
    threshold: float,
    pre20_max_pct: float,
) -> Dict[str, Any] | None:
    """单日量能+价格筛选。"""
    if row_index < 249 or row_index >= len(data):
        return None
    row = data.iloc[row_index]
    ma5 = row["MA5量能"]
    ma10 = row["MA10量能"]
    ma20 = row["MA20量能"]
    price_ma10 = row["价格MA10"]
    if pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20) or pd.isna(price_ma10) or ma10 <= 0:
        return None

    ratio = float(ma5 / ma10)
    close = float(row["收盘"])
    day_pct_chg = float(row["涨跌幅"])
    pre20_pct_chg = (close / float(data.iloc[row_index - 20]["收盘"]) - 1) * 100

    if not (
        ratio > MA5_TO_MA10_MIN_RATIO
        and threshold <= ratio <= MA5_TO_MA10_MAX_RATIO
        and ma20 < ma5 * MA20_TO_MA5_FACTOR
        and day_pct_chg > MIN_SIGNAL_DAY_PCT_CHG
        and day_pct_chg < MAX_SIGNAL_DAY_PCT_CHG
        and pre20_pct_chg < pre20_max_pct
        and close > float(price_ma10)
        and method3_price_ma_bullish_expanding(data, row_index=row_index)
    ):
        return None

    return {
        "日期": str(row["日期"]),
        "MA5量能": float(ma5),
        "MA10量能": float(ma10),
        "MA20量能": float(ma20),
        "MA5/MA10": ratio,
        "收盘": close,
        "价格MA10": float(price_ma10),
        "当天涨跌幅%": day_pct_chg,
        "信号日前20日涨跌幅%": pre20_pct_chg,
    }


def screen_by_volume_ma(
    stock_codes: List[str],
    threshold: float,
    pre20_max_pct: float,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for stock_code in stock_codes:
        code = normalize_stock_code(stock_code)
        raw = fetch_stock_df(code)
        data = prepare_ohlcv_df(raw)
        if data is None:
            continue
        hit = check_hit_at_row(data, len(data) - 1, threshold, pre20_max_pct)
        if hit is None:
            continue
        hit["股票代码"] = code
        results.append(hit)
    return results
