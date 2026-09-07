import efi_email
import util
import time
import backtest_dragon_stocks
import backtest_strategy
import os
import sys
import logging

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

# Windows 环境才跑本地 pip 升级；Linux/WSL 跳过，避免无效路径和多余邮件
if sys.platform.startswith("win"):
    os.system(r"C:\Users\DELL\PyCharmMiscProject\.venv\Scripts\python.exe -m pip install --upgrade akshare")
    os.system(r"C:\Users\DELL\PyCharmMiscProject\.venv\Scripts\python.exe -m pip install --upgrade baostock")
    os.system(r"C:\Users\DELL\PyCharmMiscProject\.venv\Scripts\python.exe -m pip install --upgrade efinance")
    efi_email.send("akshare update done")


def efi_backtesting():
    efi_email.send("Start Stock Backtesting")
    # 记录开始时间
    start_time = time.time()
    for i  in range(50):
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

        signal_day, freshness_lines, using_today = util.describe_data_freshness(
            all_results
        )
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
        pattern_lines = util.format_double_bottom_mail_section(
            all_results, signal_day
        )
        for line in pattern_lines:
            logging.info("%s", line)

        efi_email.send_backtest_result_mail(
            stock_codes,
            all_results,
            signal_day,
            freshness_lines,
            one_d_list,
            last_buys_list,
            pattern_lines,
            subject="test",
        )
        # 打印汇总统计
        # util.print_summary_statistics(all_results)
        # filtered_stocks = util.get_and_print_ideal_codes(all_results,                                                                                                                                nnnnnxnzn
        #                                                  total_return_lower_bound=0.21,
        #                                                  total_return_upper_bound=0.91,
        #                                                  win_rate=0.47,
        #                                                  num_of_trades=6
        #                                                  )
        # util.get_and_print_execution_time(start_time)
        time.sleep(400)
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
