import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

_vol_dir = Path(__file__).resolve().parent
_root = _vol_dir.parent
for _p in (_vol_dir, _root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import (
    DEFAULT_HISTORY_YEARS,
    DEFAULT_PRE20_MAX_PCT,
    DEFAULT_THRESHOLD,
    build_summary_rows,
    describe_filter_conditions_bullets,
    describe_signal_dedupe_rule,
    export_forward_result,
    format_email_body,
    format_forward_summary,
    get_signal_forward_returns,
    normalize_stock_code,
    parse_stock_codes,
    prettify_detail_df,
    screen_by_volume_ma,
    send_result_email,
)

DEFAULT_STOCK_CODES = [
    "000001", "002119", "002448", "002629", "002506", "600885", "600191", "002379", "600539", "600184", "600397",
    "002927", "603686", "603881", "600967", "002361", "600415", "002278", "600689", "603336", "603839", "600601",
]


def build_strategy_export_text(
    threshold: float,
    pre20_max_pct: float,
    history_years: int,
    stock_codes: List[str],
) -> str:
    codes = ", ".join(normalize_stock_code(c) for c in stock_codes)
    return (
        "=== 策略与参数说明 ===\n"
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"股票: {codes}\n"
        f"历史区间: 约最近 {history_years} 年日线（Baostock，起算含约 30 日缓冲）\n"
        f"命令行可调: --threshold={threshold}  --pre20-max-pct={pre20_max_pct}  --history-years={history_years}\n"
        "\n"
        "【数据长度】日线至少约 250 根已收盘 K 线（满足价格 MA250 与量能条件）。\n"
        "\n"
        "【信号日】某一交易日同时满足下列全部条件:\n"
        f"{describe_filter_conditions_bullets(threshold=threshold, pre20_max_pct=pre20_max_pct)}\n"
        "\n"
        f"【样本去重】{describe_signal_dedupe_rule()}。\n"
        "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="股票量能筛选（逻辑同 volume_ma_filter_daily_all）")
    parser.add_argument("--stocks", default=",".join(DEFAULT_STOCK_CODES))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--pre20-max-pct", type=float, default=DEFAULT_PRE20_MAX_PCT)
    parser.add_argument("--send-email", action="store_true", default=True)
    parser.add_argument("--no-send-email", dest="send_email", action="store_false")
    parser.add_argument("--analyze-forward", action="store_true", default=False)
    parser.add_argument("--no-analyze-forward", dest="analyze_forward", action="store_false")
    parser.add_argument("--history-years", type=int, default=DEFAULT_HISTORY_YEARS)
    parser.add_argument(
        "--export-csv",
        default=f"volume_ma_forward_report_{datetime.now().strftime('%Y%m%d')}.txt",
    )
    args = parser.parse_args()

    stock_codes = parse_stock_codes(args.stocks)
    results = screen_by_volume_ma(stock_codes, args.threshold, args.pre20_max_pct)
    body = format_email_body(results, args.threshold, args.pre20_max_pct)
    print(body)

    summary_lines: List[str] = []
    exported_path: str | None = None
    if args.analyze_forward:
        all_detail: List[pd.DataFrame] = []
        all_summary_rows: List[Dict[str, Any]] = []
        for stock_code in stock_codes:
            detail_df = get_signal_forward_returns(
                stock_code, args.threshold, args.history_years, args.pre20_max_pct
            )
            if detail_df.empty:
                continue
            all_detail.append(detail_df)
            all_summary_rows.extend(
                build_summary_rows(detail_df, stock_code, args.threshold, args.history_years)
            )
            summary_lines.append(
                format_forward_summary(detail_df, stock_code, args.threshold, args.history_years)
            )
            print(prettify_detail_df(detail_df).to_string(index=False, col_space=12, justify="center"))

        if summary_lines:
            body = body + "\n\n" + "\n\n".join(summary_lines)
            print("\n" + "\n\n".join(summary_lines))

        if all_detail:
            output_file = args.export_csv.strip() or f"volume_ma_forward_report_{datetime.now().strftime('%Y%m%d')}.txt"
            if not output_file.lower().endswith(".txt"):
                output_file = f"{output_file}.txt"
            strategy_txt = build_strategy_export_text(
                args.threshold, args.pre20_max_pct, args.history_years, stock_codes
            )
            exported_path = export_forward_result(
                pd.concat(all_detail, ignore_index=True),
                pd.DataFrame(all_summary_rows),
                output_file,
                strategy_txt,
            )
            print(f"\n格式化结果已导出: {exported_path}")
            if exported_path:
                body = body + f"\n\n前瞻报告文件: {exported_path}\n"

    if args.send_email and (results or summary_lines):
        send_result_email(body)


if __name__ == "__main__":
    main()
