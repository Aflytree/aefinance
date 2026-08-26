import efi_email
import util
import time
import backtest_dragon_stocks
import backtest_strategy
import os
import sys
import logging
from datetime import datetime, timedelta

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

# Windows 环境才跑本地 pip 升级；Linux/WSL 跳过，避免无效路径和多余邮件
if sys.platform.startswith("win"):
    os.system(r"C:\Users\DELL\PyCharmMiscProject\.venv\Scripts\python.exe -m pip install --upgrade akshare")
    os.system(r"C:\Users\DELL\PyCharmMiscProject\.venv\Scripts\python.exe -m pip install --upgrade baostock")
    os.system(r"C:\Users\DELL\PyCharmMiscProject\.venv\Scripts\python.exe -m pip install --upgrade efinance")
    efi_email.send("akshare update done")


def _describe_data_freshness(all_results):
    """汇总各票最新K线日期，生成邮件标注。"""
    today = datetime.now().date()
    as_of_dates = []
    stale_codes = []
    today_codes = []
    for r in all_results:
        d = r.get("data_as_of")
        if d is None:
            continue
        as_of_dates.append(d)
        code = r.get("stock_code", "")
        if d >= today:
            today_codes.append(code)
        else:
            stale_codes.append(f"{code}({d})")

    if not as_of_dates:
        return today, [
            "【数据新鲜度】未能判定最新K线日期，请人工核对",
        ], False

    # 多数票的最新日期作为信号日
    from collections import Counter
    signal_day = Counter(as_of_dates).most_common(1)[0][0]
    using_today = signal_day >= today

    lines = [
        f"【数据新鲜度】日历今日: {today}",
        f"【数据新鲜度】信号日(多数票最新K线): {signal_day}",
    ]
    if using_today:
        lines.append("【数据新鲜度】使用的是【今日】股票数据")
    else:
        yesterday = today - timedelta(days=1)
        if signal_day == yesterday:
            lines.append(
                f"【数据新鲜度】注意：使用的是【昨日 {signal_day}】股票数据"
                f"（非今日 {today}，盘中/数据源尚未提供当日K线）"
            )
        else:
            lines.append(
                f"【数据新鲜度】注意：使用的是【非今日】股票数据，最新K线={signal_day}"
                f"（日历今日={today}）"
            )
    if today_codes:
        lines.append(f"【数据新鲜度】已含当日K线: {len(today_codes)} 只")
    if stale_codes:
        lines.append(
            f"【数据新鲜度】缺少当日K线: {len(stale_codes)} 只 -> "
            + ", ".join(stale_codes[:12])
            + (" ..." if len(stale_codes) > 12 else "")
        )
    return signal_day, lines, using_today


def efi_backtesting():
    efi_email.send("Start Stock Backtesting")
    # 记录开始时间
    start_time = time.time()
    for i  in range(1):
        stock_codes = []
        #常规关注股票
        stock_codes =list(set(stock_codes + [ '002119', '002448',
                          '002629', '002506',
                        '600885',
                       '600191']))
        #龙虎榜最近4个月符合条件股票  0.27 0.90, win_rate > 0.47, 交易>6
        stock_codes = list(set(stock_codes + ['002379', '600539', '002119', '600184',
                                              '600397','002927', '603686', '603881', '600967',
                                                '002361']))
        #20251029 add
        stock_codes = list(set(stock_codes + ['600415', '002278', '600689', '603336', '603839','603336']))
        #20260817 add
        stock_codes = list(set(stock_codes + ['600601', '600595', '600549', '603228']))
        # stock_codes = ['600397', '603336','002379', '603881', '002448', '600689','600415','600539', '603839', ]
        # stock_codes = ['600539' ]

        # stock_codes, day_dragons = util.get_dragon_tiger_stocks(date="20251022")
        # stock_codes = util.get_recent_days_lhb_stocks(days=120)
        all_results = []
        daily_trades = []
        last_buys = []
        logging.info("\n开始回测买入信号股票...")
        for code in stock_codes:
            results = backtest_strategy.backtest_strategy(code,
                                        # bg = '20210223',
                                        bg = '20240323',
                                        initial_capital_ = 1000000,
                                        target_return_ = 0.11,
                                        stop_loss_ = -0.03,
                                        init_stop_n_times = 0
                                        )
            if results is None:
                continue
            util.calculate_holding_days_stats(results)
            util.print_backtest_results(results)
            all_results.append(results)

        signal_day, freshness_lines, using_today = _describe_data_freshness(all_results)
        logging.info("信号日=%s 使用今日数据=%s", signal_day, using_today)
        for line in freshness_lines:
            logging.info("%s", line)

        for results in all_results:
            code = results["stock_code"]
            daily_trades.append(util.trade_daily(code, results, signal_day=signal_day))
            last_buys.append(util.last_busy(code, results, signal_day=signal_day))

        one_d_list = [
            item for sublist in daily_trades if sublist for item in sublist
        ]
        last_buys_list = [
            item for sublist in last_buys if sublist for item in sublist
        ]
        mail_lines = [
            f"回测运行日: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"股票数: {len(stock_codes)}, 有效回测: {len(all_results)}",
            "",
        ]
        mail_lines.extend(freshness_lines)
        mail_lines.extend([
            "",
            "--------------------------------------",
            f" signal-day trades ({signal_day})",
            "--------------------------------------",
        ])
        if one_d_list:
            mail_lines.extend(one_d_list)
        else:
            mail_lines.append(f"(无 {signal_day} 买卖)")
        mail_lines.extend([
            "",
            "--------------------------------------",
            " current holds",
            "--------------------------------------",
        ])
        if last_buys_list:
            mail_lines.extend(last_buys_list)
        else:
            mail_lines.append("(无当前持仓)")
        logging.info(
            "信号日买卖 %s 条, 当前持仓 %s 条",
            len(one_d_list),
            len(last_buys_list),
        )
        for line in last_buys_list:
            logging.info("%s", line)
        efi_email.send(mail_lines, subject="test")
        # 打印汇总统计
        # util.print_summary_statistics(all_results)
        # filtered_stocks = util.get_and_print_ideal_codes(all_results,                                                                                                                                nnnnnxnzn
        #                                                  total_return_lower_bound=0.21,
        #                                                  total_return_upper_bound=0.91,
        #                                                  win_rate=0.47,
        #                                                  num_of_trades=6
        #                                                  )
        # util.get_and_print_execution_time(start_time)
        # time.sleep(400)
        # # # # # # # # # # 可视化结果
        # util.visualize_backtest_results(all_results)
        # # # 打印统计摘要
        # util.logging.info_signal_summary(buy_signals, sell_signals, neutral_signals)

        # # 可视化结果
        # util.visualize_signals(buy_signals, sell_signals, neutral_signals)


def do_lhb_efi_backtesting():
    try:
        # 执行回测
        backtest_dragon_stocks.backtest_recent_dragon_stocks(
                                                            total_lhb_days = 30,
                                                            single_stock_start_date = '20240323',
                                                            win_rate_th = 0.50,
                                                            total_return = 0.50)
    except Exception as e:
        logging.error(f"回测执行过程中发生错误: {str(e)}")


if __name__ == "__main__":
    # do_lhb_efi_backtesting()
    efi_backtesting()
