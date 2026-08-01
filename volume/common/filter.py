from datetime import date
from typing import Any, Dict, List

import pandas as pd

from .constants import (
    AVG_DAILY_AMOUNT_LOOKBACK,
    DEFAULT_PRE20_MAX_PCT,
    DEFAULT_THRESHOLD,
    HITS_PERIOD_DAYS,
    MA20_TO_MA5_FACTOR,
    MA5_TO_MA10_MAX_RATIO,
    MA5_TO_MA10_MIN_RATIO,
    MAX_HIT_CLOSE_PRICE,
    MAX_HIT_PE,
    MAX_SIGNAL_DAY_PCT_CHG,
    MIN_AVG_DAILY_AMOUNT,
    MIN_HIT_CLOSE_PRICE,
    MIN_SIGNAL_DAY_PCT_CHG,
)
from .data import fetch_stock_df, normalize_stock_code


def describe_method3_condition() -> str:
    """与 method3_price_ma_bullish_expanding 实际判定一致。"""
    return "价格method3(价格MA5>价格MA20且MA5-MA10价差较昨日扩大)"


def describe_filter_conditions_text(
    *,
    threshold: float | None = None,
    pre20_max_pct: float | None = None,
) -> str:
    """从 constants 与 check_hit_at_row / method3 生成筛选条件说明（供邮件/报告引用）。"""
    th = DEFAULT_THRESHOLD if threshold is None else threshold
    pre20 = DEFAULT_PRE20_MAX_PCT if pre20_max_pct is None else pre20_max_pct
    return (
        f"MA5量能>=MA10量能*{th} 且 MA5/MA10>{MA5_TO_MA10_MIN_RATIO} 且 MA5/MA10<={MA5_TO_MA10_MAX_RATIO} "
        f"且 MA20量能<MA5*{MA20_TO_MA5_FACTOR} "
        f"且 当天涨跌幅∈({MIN_SIGNAL_DAY_PCT_CHG},{MAX_SIGNAL_DAY_PCT_CHG})% "
        f"且 信号日前20日涨跌幅<{pre20}% 且 收盘>价格MA10 且 {describe_method3_condition()} "
        f"且 二次过滤(剔银行/剔净利润亏损/"
        f"收盘∈[{MIN_HIT_CLOSE_PRICE},{MAX_HIT_CLOSE_PRICE}]/"
        f"近{AVG_DAILY_AMOUNT_LOOKBACK}日均成交额>={MIN_AVG_DAILY_AMOUNT / 1e4:.0f}万/"
        f"PE∈(0,{MAX_HIT_PE}])"
    )


def format_filter_conditions_line(
    *,
    threshold: float | None = None,
    pre20_max_pct: float | None = None,
    label: str = "条件",
) -> str:
    return f"{label}: {describe_filter_conditions_text(threshold=threshold, pre20_max_pct=pre20_max_pct)}"


def describe_filter_conditions_bullets(
    *,
    threshold: float | None = None,
    pre20_max_pct: float | None = None,
) -> str:
    """多行条件说明（前瞻报告头等）。"""
    th = DEFAULT_THRESHOLD if threshold is None else threshold
    pre20 = DEFAULT_PRE20_MAX_PCT if pre20_max_pct is None else pre20_max_pct
    lines = [
        f"  1) MA10量能 > 0",
        f"  2) MA5量能/MA10量能 > {MA5_TO_MA10_MIN_RATIO} 且 {th} <= MA5量能/MA10量能 <= {MA5_TO_MA10_MAX_RATIO}",
        f"  3) MA20量能 < MA5量能 × {MA20_TO_MA5_FACTOR}",
        f"  4) 当天涨跌幅: {MIN_SIGNAL_DAY_PCT_CHG}% < 涨跌幅 < {MAX_SIGNAL_DAY_PCT_CHG}%",
        f"  5) 信号日前20个交易日涨跌幅 < {pre20}%",
        "  6) 当日收盘 > 当日价格MA10（收盘10日均线）",
        f"  7) {describe_method3_condition()}",
        "  8) 二次过滤: 剔除银行（名称/行业含银行）",
        "  9) 二次过滤: 剔除最近完整报告期净利润 < 0",
        (
            f"  10) 二次过滤: 剔除信号日收盘价 < {MIN_HIT_CLOSE_PRICE} "
            f"或 > {MAX_HIT_CLOSE_PRICE}"
        ),
        (
            f"  11) 二次过滤: 剔除近{AVG_DAILY_AMOUNT_LOOKBACK}日日均成交额 "
            f"< {MIN_AVG_DAILY_AMOUNT / 1e4:.0f}万"
        ),
        f"  12) 二次过滤: 剔除动态市盈率 PE<=0 或 PE>{MAX_HIT_PE}",
    ]
    return "\n".join(lines)


def describe_signal_dedupe_rule() -> str:
    return f"同股任意连续{HITS_PERIOD_DAYS}个自然日内仅保留首次命中"


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
    if "成交额" in data.columns:
        data["成交额"] = pd.to_numeric(data["成交额"], errors="coerce")
    else:
        # 无成交额时用 收盘×成交量 近似（Baostock 成交量一般为股）
        data["成交额"] = data["收盘"] * data["成交量"]
    data = data.dropna(subset=["成交量", "收盘", "涨跌幅"]).reset_index(drop=True)
    if len(data) < 250:
        return None
    data["MA5量能"] = data["成交量"].rolling(5).mean()
    data["MA10量能"] = data["成交量"].rolling(10).mean()
    data["MA20量能"] = data["成交量"].rolling(20).mean()
    data["价格MA10"] = data["收盘"].rolling(10).mean()
    data["价格MA120"] = data["收盘"].rolling(120).mean()
    data["价格MA250"] = data["收盘"].rolling(250).mean()
    data["日均成交额"] = data["成交额"].rolling(AVG_DAILY_AMOUNT_LOOKBACK).mean()
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

    avg_amount = row.get("日均成交额")
    try:
        avg_amount_f = float(avg_amount) if pd.notna(avg_amount) else float("nan")
    except (TypeError, ValueError):
        avg_amount_f = float("nan")

    def _ma_f(col: str) -> float:
        v = row.get(col)
        try:
            return float(v) if pd.notna(v) else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    return {
        "日期": str(row["日期"]),
        "MA5量能": float(ma5),
        "MA10量能": float(ma10),
        "MA20量能": float(ma20),
        "MA5/MA10": ratio,
        "收盘": close,
        "价格MA10": float(price_ma10),
        "价格MA120": _ma_f("价格MA120"),
        "价格MA250": _ma_f("价格MA250"),
        "当天涨跌幅%": day_pct_chg,
        "信号日前20日涨跌幅%": pre20_pct_chg,
        "日均成交额": avg_amount_f,
    }


def parse_hit_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def dedupe_hits_first_within_days(
    hits: List[Dict[str, Any]],
    *,
    period_days: int = HITS_PERIOD_DAYS,
) -> List[Dict[str, Any]]:
    """每只股票在任意连续 period_days 个自然日内只保留首次命中。"""
    kept: List[Dict[str, Any]] = []
    last_kept_date: Dict[str, date] = {}
    for h in sorted(hits, key=lambda x: (parse_hit_date(x["日期"]), x["股票代码"])):
        code = h["股票代码"]
        d = parse_hit_date(h["日期"])
        prev = last_kept_date.get(code)
        if prev is not None and (d - prev).days < period_days:
            continue
        kept.append(h)
        last_kept_date[code] = d
    return kept


def dedupe_signal_indices(
    data: pd.DataFrame,
    indices: List[int],
    *,
    period_days: int = HITS_PERIOD_DAYS,
) -> List[int]:
    """单股信号行 index：任意连续 period_days 个自然日内只保留首次。"""
    kept: List[int] = []
    last_date: date | None = None
    for idx in sorted(indices):
        d = parse_hit_date(data.loc[idx, "日期"])
        if last_date is not None and (d - last_date).days < period_days:
            continue
        kept.append(idx)
        last_date = d
    return kept


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
