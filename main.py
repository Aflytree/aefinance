import efi_email
import util
import  time
import backtest_dragon_stocks
import backtest_strategy

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

def efi_backtesting():
    efi_email.send("Start Stock Backtesting")
    # 记录开始时间
    start_time = time.time()
    for i  in range(50):
        stock_codes = []
        #常规关注股票
        stock_codes =list(set(stock_codes + ['002119', '002448',
                         '002156', '002629','688041', '002506',
                       '002594', '600882', '600885',
                       '600191']))
        #龙虎榜最近4个月符合条件股票  0.27 0.90, win_rate > 0.47, 交易>6
        stock_codes = list(set(stock_codes + ['002379', '600539', '002119', '600184',
                                              '600397', '603228','002927', '603686', '603881', '600967',
                                              '002594', '600595', '002361']))
        #20251029 add
        stock_codes = list(set(stock_codes + ['600415', '002278', '600689', '603336', '603839','603336']))
        # stock_codes = ['600689']
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
            daily_trades.append(util.trade_daily(code, results))
            # import pdb;pdb.set_trace()
            last_buys.append(util.last_busy(code, results))

        one_d_list = [item for sublist in daily_trades for item in sublist]
        last_buys_list = [item for sublist in last_buys for item in sublist]
        one_d_list.append("\n -------------------------------------- \n"
                          " current holds \n "
                           "--------------------------------------\n")
        lt = one_d_list + last_buys_list
        efi_email.send(lt)
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
        # # efi_email.send(  "Next round ...")
        # util.draw_stock_code_price(all_results)
        # # # # # # # # # # 可视化结果
        # util.visualize_backtest_results(all_results)
        # # # 打印统计摘要
        # util.logging.info_signal_summary(buy_signals, sell_signals, neutral_signals)

        # 可视化结果
        # util.visualize_signals(buy_signals, sell_signals, neutral_signals)

def do_lhb_efi_backtesting():
    try:
        # 执行回测
        backtest_dragon_stocks.backtest_recent_dragon_stocks(
                                                            total_lhb_days = 60,
                                                            single_stock_start_date = '20210323',
                                                            win_rate_th = 0.50,
                                                            total_return = 3.00)
    except Exception as e:
        logging.error(f"回测执行过程中发生错误: {str(e)}")


if __name__ == "__main__":
    # do_lhb_efi_backtesting()
    efi_backtesting()