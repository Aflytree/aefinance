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
    export_forward_result,
    format_email_body,
    format_forward_summary,
    get_signal_forward_returns,
    parse_stock_codes,
    prettify_detail_df,
    screen_by_volume_ma,
    send_result_email,
)

DEFAULT_STOCK_CODES = ["600601"]


def main() -> None:
    parser = argparse.ArgumentParser(description="股票量能筛选：MA5量能大于MA10量能的指定倍数")
    parser.add_argument(
        "--stocks",
        default=",".join(DEFAULT_STOCK_CODES),
        help="股票代码，多个用逗号分隔",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--send-email", action="store_true", default=True)
    parser.add_argument("--no-send-email", dest="send_email", action="store_false")
    parser.add_argument("--analyze-forward", action="store_true", default=False)
    parser.add_argument("--no-analyze-forward", dest="analyze_forward", action="store_false")
    parser.add_argument("--history-years", type=int, default=DEFAULT_HISTORY_YEARS)
    parser.add_argument("--pre20-max-pct", type=float, default=DEFAULT_PRE20_MAX_PCT)
    parser.add_argument(
        "--export-csv",
        default=f"volume_ma_forward_report_{datetime.now().strftime('%Y%m%d')}.txt",
    )
    args = parser.parse_args()

    stock_codes = parse_stock_codes(args.stocks)
    results = screen_by_volume_ma(stock_codes, args.threshold, args.pre20_max_pct)
    body = format_email_body(results, args.threshold, args.pre20_max_pct)
    print(body)

    if args.analyze_forward:
        all_detail: List[pd.DataFrame] = []
        all_summary_rows: List[Dict[str, Any]] = []
        summary_lines: List[str] = []
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
            path = export_forward_result(
                pd.concat(all_detail, ignore_index=True),
                pd.DataFrame(all_summary_rows),
                output_file,
            )
            print(f"\n格式化结果已导出: {path}")

    detail_txt = f"volume_ma_filter_detail_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(detail_txt, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"\n结果已导出: {detail_txt}")

    if args.send_email and results:
        send_result_email(body)


if __name__ == "__main__":
    main()
