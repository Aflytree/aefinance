import unicodedata
from typing import Any, List

import pandas as pd

from .forward import prettify_detail_df


def export_forward_result(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: str,
    strategy_header: str = "",
) -> str:
    pretty_detail = prettify_detail_df(detail_df)

    def _fmt_value(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "-"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _display_width(text: str) -> int:
        width = 0
        for ch in text:
            width += 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
        return width

    def _pad_to_width(text: str, width: int) -> str:
        pad = max(0, width - _display_width(text))
        return text + (" " * pad)

    def _build_pipe_table(df: pd.DataFrame, columns: List[tuple]) -> str:
        col_widths: List[int] = []
        for name, col in columns:
            max_w = _display_width(name)
            for _, row in df.iterrows():
                cell = _fmt_value(row[col])
                max_w = max(max_w, _display_width(cell))
            col_widths.append(max_w)

        header_cells = [
            _pad_to_width(name, col_widths[i]) for i, (name, _) in enumerate(columns)
        ]
        header = "| " + " | ".join(header_cells) + " |"
        sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
        lines = [header, sep]
        for _, row in df.iterrows():
            data_cells = [
                _pad_to_width(_fmt_value(row[col]), col_widths[i])
                for i, (_, col) in enumerate(columns)
            ]
            lines.append("| " + " | ".join(data_cells) + " |")
        return "\n".join(lines)

    text_lines: List[str] = []
    if strategy_header.strip():
        text_lines.append(strategy_header.rstrip())
        text_lines.append("")
    text_lines.append("=== 前瞻统计汇总 ===")
    if summary_df.empty:
        text_lines.append("无汇总数据")
    else:
        summary_columns = [
            ("股票代码", "股票代码"),
            ("历史年数", "历史年数"),
            ("阈值", "阈值"),
            ("周期", "周期"),
            ("样本数", "样本数"),
            ("上涨平均涨幅(%)", "上涨平均涨幅(%)"),
            ("下跌平均跌幅(%)", "下跌平均跌幅(%)"),
            ("中位数涨跌幅(%)", "中位数涨跌幅(%)"),
            ("胜率(%)", "胜率(%)"),
            ("最大涨跌幅(%)", "最大涨跌幅(%)"),
            ("最小涨跌幅(%)", "最小涨跌幅(%)"),
        ]
        text_lines.append(_build_pipe_table(summary_df, summary_columns))

    text_lines.append("")
    text_lines.append("=== 前瞻统计明细 ===")
    if pretty_detail.empty:
        text_lines.append("无明细数据")
    else:
        detail_columns = [
            ("股票代码", "股票代码"),
            ("日期", "日期"),
            ("MA5/MA10", "MA5/MA10"),
            ("信号收盘", "信号收盘"),
            ("价格MA10", "价格MA10"),
            ("当天涨跌幅%", "当天涨跌幅%"),
            ("信号日前20日涨跌幅%", "信号日前20日涨跌幅%"),
            ("DIF", "DIF"),
            ("DEA", "DEA"),
            ("MACD", "MACD"),
            ("3日涨跌幅%", "3日涨跌幅%"),
            ("5日涨跌幅%", "5日涨跌幅%"),
            ("10日涨跌幅%", "10日涨跌幅%"),
            ("20日涨跌幅%", "20日涨跌幅%"),
        ]
        detail_columns = [(n, c) for n, c in detail_columns if c in pretty_detail.columns]
        text_lines.append(_build_pipe_table(pretty_detail, detail_columns))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines) + "\n")

    return output_path
