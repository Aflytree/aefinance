"""全量回测验证（新 buy_signal + 6% ATR 止损），不写邮件、不 sleep。"""
import io
import sys
import time
from datetime import datetime

import backtest_strategy
import util

# 与 main.py 相同的股票池
STOCK_CODES = list(
    set(
        [
            "002119",
            "002448",
            "002629",
            "002506",
            "600885",
            "600191",
            "002379",
            "600539",
            "600184",
            "600397",
            "002927",
            "603686",
            "603881",
            "600967",
            "002361",
            "600415",
            "002278",
            "600689",
            "603336",
            "603839",
            "600601",
            "600595",
            "600549",
            "603228",
        ]
    )
)
STOCK_CODES.sort()


def _collect_stock_codes():
    return STOCK_CODES


def main():
    log_path = f"main_backtest_{datetime.now().strftime('%Y%m%d')}_v4_utf8.log"
    buf = io.StringIO()
    tee = _Tee(sys.stdout, buf)

    start = time.time()
    with _redirect_stdout(tee):
        print(f"=== 全量回测验证 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        print("策略: compute_signal_scores buy_ok + MAX_DYN_STOP_LOSS=-6%")
        print(f"股票数: {len(STOCK_CODES)}")
        print()

        all_results = []
        for code in STOCK_CODES:
            print(f"\n--- 回测 {code} ---")
            results = backtest_strategy.backtest_strategy(
                code,
                bg="20240323",
                initial_capital_=1_000_000,
                target_return_=0.11,
                stop_loss_=-0.03,
                init_stop_n_times=0,
            )
            if results is None:
                print(f"{code}: 回测失败")
                continue
            util.calculate_holding_days_stats(results)
            util.print_backtest_results(results)
            all_results.append(results)

        print()
        util.print_summary_statistics(all_results)

        # 汇总表
        print("\n=== 全池明细 ===")
        print(f"{'代码':<8} {'名称':<8} {'总收益%':>8} {'年化%':>8} {'胜率%':>7} {'回撤%':>7} {'次数':>6}")
        print("-" * 62)
        rows = []
        for r in sorted(all_results, key=lambda x: x["total_return"], reverse=True):
            dd = util.calculate_max_drawdown(r) * 100
            name = util.get_stock_name(r["stock_code"])[:6]
            row = (
                r["stock_code"],
                name,
                r["total_return"] * 100,
                r["annual_return"] * 100,
                r["win_rate"] * 100,
                dd,
                r["number_of_trades"],
            )
            rows.append(row)
            print(
                f"{row[0]:<8} {row[1]:<8} {row[2]:8.2f} {row[3]:8.2f} "
                f"{row[4]:7.1f} {row[5]:7.2f} {row[6]:6.1f}"
            )

        pos = sum(1 for r in rows if r[2] > 0)
        print("-" * 62)
        print(
            f"盈利 {pos}/{len(rows)} | "
            f"平均收益 {sum(r[2] for r in rows) / len(rows):.2f}% | "
            f"中位 {sorted(r[2] for r in rows)[len(rows) // 2]:.2f}% | "
            f"平均胜率 {sum(r[4] for r in rows) / len(rows):.1f}%"
        )
        print(f"\n耗时: {time.time() - start:.1f}s")

    text = buf.getvalue()
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n日志已写入: {log_path}", file=sys.__stdout__)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


class _redirect_stdout:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        self._old = sys.stdout
        sys.stdout = self.stream
        return self.stream

    def __exit__(self, *args):
        sys.stdout = self._old


if __name__ == "__main__":
    main()
