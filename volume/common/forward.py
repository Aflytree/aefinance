from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd

from stock_history import get_quote_history_baostock

from .data import normalize_stock_code
from .filter import check_hit_at_row, dedupe_signal_indices, prepare_ohlcv_df
from .indicators import add_price_macd_columns, macd_value_at


def get_signal_forward_returns(
    stock_code: str,
    threshold: float,
    years: int,
    pre20_max_pct: float,
    *,
    baostock_manage_login: bool = True,
    baostock_verbose: bool = False,
) -> pd.DataFrame:
    end = datetime.now().strftime("%Y%m%d")
    beg = (datetime.now() - timedelta(days=365 * years + 30)).strftime("%Y%m%d")
    raw = get_quote_history_baostock(
        stock_code,
        beg=beg,
        end=end,
        manage_login=baostock_manage_login,
        verbose=baostock_verbose,
    )
    data = prepare_ohlcv_df(raw)
    if data is None:
        return pd.DataFrame()

    data = add_price_macd_columns(data)
    signal_idx: List[int] = []
    for idx in range(249, len(data)):
        if check_hit_at_row(data, idx, threshold, pre20_max_pct) is not None:
            signal_idx.append(idx)

    filtered_signal_idx = dedupe_signal_indices(data, signal_idx)

    rows: List[Dict[str, Any]] = []
    code = normalize_stock_code(stock_code)
    for idx in filtered_signal_idx:
        close_now = float(data.loc[idx, "收盘"])
        ma5_ma10 = float(data.loc[idx, "MA5量能"] / data.loc[idx, "MA10量能"])
        ret3 = ret5 = ret10 = ret20 = None
        if idx + 3 < len(data):
            ret3 = (float(data.loc[idx + 3, "收盘"]) / close_now - 1) * 100
        if idx + 5 < len(data):
            ret5 = (float(data.loc[idx + 5, "收盘"]) / close_now - 1) * 100
        if idx + 10 < len(data):
            ret10 = (float(data.loc[idx + 10, "收盘"]) / close_now - 1) * 100
        if idx + 20 < len(data):
            ret20 = (float(data.loc[idx + 20, "收盘"]) / close_now - 1) * 100
        hit = check_hit_at_row(data, idx, threshold, pre20_max_pct)

        rows.append(
            {
                "股票代码": code,
                "日期": str(data.loc[idx, "日期"]),
                "MA5/MA10": ma5_ma10,
                "信号收盘": close_now,
                "价格MA10": hit["价格MA10"] if hit else None,
                "当天涨跌幅%": hit["当天涨跌幅%"] if hit else None,
                "信号日前20日涨跌幅%": hit["信号日前20日涨跌幅%"] if hit else None,
                "3日涨跌幅%": ret3,
                "5日涨跌幅%": ret5,
                "10日涨跌幅%": ret10,
                "20日涨跌幅%": ret20,
                "DIF": macd_value_at(data, idx, "DIF"),
                "DEA": macd_value_at(data, idx, "DEA"),
                "MACD": macd_value_at(data, idx, "MACD"),
            }
        )

    return pd.DataFrame(rows)


def build_summary_rows(
    df: pd.DataFrame, stock_code: str, threshold: float, years: int
) -> List[Dict[str, Any]]:
    code = normalize_stock_code(stock_code)
    rows: List[Dict[str, Any]] = []
    for col, name in [
        ("3日涨跌幅%", "3日"),
        ("5日涨跌幅%", "5日"),
        ("10日涨跌幅%", "10日"),
        ("20日涨跌幅%", "20日"),
    ]:
        s = df[col].dropna()
        if len(s) == 0:
            rows.append(
                {
                    "股票代码": code,
                    "历史年数": years,
                    "阈值": threshold,
                    "周期": name,
                    "样本数": 0,
                    "上涨平均涨幅(%)": None,
                    "下跌平均跌幅(%)": None,
                    "中位数涨跌幅(%)": None,
                    "胜率(%)": None,
                    "最大涨跌幅(%)": None,
                    "最小涨跌幅(%)": None,
                }
            )
            continue

        up = s[s > 0]
        down = s[s < 0]
        rows.append(
            {
                "股票代码": code,
                "历史年数": years,
                "阈值": threshold,
                "周期": name,
                "样本数": int(len(s)),
                "上涨平均涨幅(%)": round(float(up.mean()), 2) if len(up) else None,
                "下跌平均跌幅(%)": round(float(down.mean()), 2) if len(down) else None,
                "中位数涨跌幅(%)": round(float(s.median()), 2),
                "胜率(%)": round(float((s > 0).mean() * 100), 2),
                "最大涨跌幅(%)": round(float(s.max()), 2),
                "最小涨跌幅(%)": round(float(s.min()), 2),
            }
        )
    return rows


def format_forward_summary(
    df: pd.DataFrame, stock_code: str, threshold: float, years: int
) -> str:
    code = normalize_stock_code(stock_code)
    if df.empty:
        return f"{code} 最近{years}年无可用信号数据。"

    lines = [
        f"{code} 最近{years}年信号统计（阈值: {threshold}）",
        f"总信号数: {len(df)}",
    ]
    for col, name in [
        ("3日涨跌幅%", "3日"),
        ("5日涨跌幅%", "5日"),
        ("10日涨跌幅%", "10日"),
        ("20日涨跌幅%", "20日"),
    ]:
        s = df[col].dropna()
        if len(s) == 0:
            lines.append(f"{name}: 无有效样本")
            continue
        up = s[s > 0]
        down = s[s < 0]
        up_avg = f"{up.mean():.4f}%" if len(up) else "-"
        down_avg = f"{down.mean():.4f}%" if len(down) else "-"
        lines.append(
            f"{name}: 样本={len(s)} 上涨均值={up_avg} 下跌均值={down_avg} 中位数={s.median():.4f}% "
            f"胜率={(s > 0).mean() * 100:.2f}% 最大={s.max():.4f}% 最小={s.min():.4f}%"
        )
    return "\n".join(lines)


def prettify_detail_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    out = out.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")

    for col in [
        "MA5/MA10",
        "信号收盘",
        "价格MA10",
        "当天涨跌幅%",
        "信号日前20日涨跌幅%",
        "3日涨跌幅%",
        "5日涨跌幅%",
        "10日涨跌幅%",
        "20日涨跌幅%",
        "DIF",
        "DEA",
        "MACD",
    ]:
        if col in out.columns:
            decimals = 4 if col in ("DIF", "DEA", "MACD") else 2
            out[col] = pd.to_numeric(out[col], errors="coerce").round(decimals)
    return out
