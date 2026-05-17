from datetime import datetime
from typing import Any, Dict, List

from .constants import (
    MA20_TO_MA5_FACTOR,
    MA5_TO_MA10_MAX_RATIO,
    MA5_TO_MA10_MIN_RATIO,
    MAX_SIGNAL_DAY_PCT_CHG,
    MIN_SIGNAL_DAY_PCT_CHG,
)


def format_email_body(
    results: List[Dict[str, Any]],
    threshold: float,
    pre20_max_pct: float,
) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    cond = (
        f"条件: MA5量能 >= MA10量能 * {threshold} 且 MA5/MA10 > {MA5_TO_MA10_MIN_RATIO} "
        f"且 MA5/MA10 <= {MA5_TO_MA10_MAX_RATIO} 且 MA20量能 < MA5量能 * {MA20_TO_MA5_FACTOR} "
        f"且 当天涨跌幅 > {MIN_SIGNAL_DAY_PCT_CHG}% 且 当天涨跌幅 < {MAX_SIGNAL_DAY_PCT_CHG}% "
        f"且 信号日前20日涨跌幅 < {pre20_max_pct}% 且 收盘 > 价格MA10 "
        f"且 方法1(MA20>MA30/60/120/250) 且 方法三(MA5>MA10>MA20且价差扩大)"
    )
    if not results:
        return f"{date_str} 量能筛选结果\n{cond}\n无符合条件股票。"

    lines = [
        f"{date_str} 量能筛选结果",
        cond,
        f"命中数量: {len(results)}",
        "",
    ]
    for item in results:
        lines.append(
            f"{item['股票代码']} | 日期:{item['日期']} | MA5:{item['MA5量能']:.0f} | "
            f"MA10:{item['MA10量能']:.0f} | MA20:{item['MA20量能']:.0f} | "
            f"比值:{item['MA5/MA10']:.3f} | 收盘/价格MA10:{item['收盘']:.2f}/{item['价格MA10']:.2f} | "
            f"当天涨跌幅:{item['当天涨跌幅%']:.2f}% | "
            f"前20日涨跌幅:{item['信号日前20日涨跌幅%']:.2f}% | 收盘:{item['收盘']}"
        )
    return "\n".join(lines)


def send_result_email(body: str) -> None:
    send_func = None
    try:
        from email import send as _send  # type: ignore

        send_func = _send
    except Exception:
        pass

    if send_func is None:
        from efi_email import send as _send

        send_func = _send

    send_func(body)


def parse_stock_codes(stocks_arg: str) -> List[str]:
    return [s.strip() for s in stocks_arg.split(",") if s.strip()]
