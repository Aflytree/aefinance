import efi_email
import util
import technical_indicator_analysis
import  time

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.ERROR)

def backtest_strategy(stock_code,
                      days_=252,
                      initial_capital_ = 1000000,
                      target_return_ = 0.11,
                      stop_loss_ = -0.03,
                      init_stop_n_times = 1):
    """
    对单只股票进行回测
    :param stock_code: 股票代码
    :param days_: 回测天数
    :return: 回测结果
    """
    try:
        # 使用 StockAnalyzer 类获取数据和计算指标
        analyzer = technical_indicator_analysis.StockAnalyzer(stock_code=stock_code, beg='20240223')
        df = analyzer.df

        if df.empty:
            print(f"未获取到股票 {stock_code} 的历史数据")
            return None

        # 初始化回测参数
        # initial_capital = 1000000  # 初始资金10万
        position = 0  # 持仓数量
        capital = initial_capital_  # 当前资金
        trades = []  # 交易记录
        holding_days = 0  # 持仓天数
        target_return = target_return_   # 目标收益率
        stop_loss = stop_loss_  # 止损线
        entry_price = 0  # 买入价格
        cumulative_gain = 0
        max_drawdown = 0
        peak_capital = initial_capital_  # 记录历史最高资金
        peak_return = 0  # 记录历史最高收益率
        drawdown_threshold = 0.10  # 回撤阈值
        mode_flag = 0
        buy_trades_holdings = []
        # init_stop_n_times = 1
        stop_n_times = init_stop_n_times

        # 跳过前20天，确保有足够数据计算指标
        for i in range(20, len(df)):
            try:
                # 更新分析器的数据窗口
                analyzer.df = df[i - 20:i + 1]

                date = df.index[i]
                current_price = df['收盘'].iloc[i]
                logging.info(f"*****************date {date} **********************")
                # 获取交易建议
                advice = analyzer.get_trading_advice1()

                # 解析交易建议中的信号
                buy_signal = 0
                sell_signal = 0

                # 根据建议内容判断买卖信号
                if "强烈买入信号" in advice:
                    buy_signal += 2
                elif "买入信号" in advice:
                    buy_signal += 1
                elif "强烈卖出信号" in advice:
                    sell_signal += 2
                elif "卖出信号" in advice:
                    sell_signal += 1

                # 分析建议中的具体理由
                if "价格处于上升趋势" in advice:
                    # pric_trend = 1
                    buy_signal += 1
                if "价格处于下降趋势" in advice:
                    sell_signal += 1
                if "量能配合良好" in advice:
                    buy_signal += 1
                    pric_trend = 1
                if "量能配合显示卖压" in advice:
                    sell_signal += 1
                if "技术指标显示买入信号" in advice:
                    buy_signal += 1
                if "技术指标显示卖出信号" in advice:
                    sell_signal += 1


                # 交易逻辑
                if position == 0:  # 没有持仓
                    logging.info(f"[没有仓位] max_drawdown {max_drawdown} ")
                    logging.info(f"[没有仓位] drawdown_threshold {drawdown_threshold} ")

                    if buy_signal >= 2:  # 至少达到有效买入信号
                        buy_trades_holdings.append(
                            {
                                'date': date,
                                'type': 'buy',
                                'price': current_price,
                                'quantity': position,
                                'advice': advice,
                                'reason': '：\n- ' + '\n- '.join("xxx")
                            }
                        )
                        # #
                        # logging.info(f"[bypass] stop_n_times {stop_n_times} ")
                        if abs(max_drawdown) > drawdown_threshold and stop_n_times > 0:
                            logging.info(f"[bypass] mode_flag {mode_flag} ")
                            logging.info(f"[bypass] max_drawdown {max_drawdown} ")
                            logging.info(f"[bypass] stop_n_times {stop_n_times} ")
                            stop_n_times = stop_n_times - 1
                            continue
                        #
                        # if stop_n_times == 0:
                        #     stop_n_times = init_stop_n_times

                        logging.info(f"[执行买入] buy_signal {buy_signal} ")
                        position = int(capital / current_price)  # 全仓买入
                        entry_price = current_price
                        capital -= position * current_price
                        trades.append({
                            'date': date,
                            'type': 'buy',
                            'price': current_price,
                            'quantity': position,
                            'signals': buy_signal,
                            'advice': advice,
                            'reason': '买入理由：\n' + '\n'.join([
                                line for line in advice.split('\n')
                                if line.startswith('- ') and '买入' in line
                            ]
                            )
                        })
                        holding_days = 0

                elif position > 0:  # 持有仓位
                    if buy_signal >= 2:
                        logging.info(f"[持有仓位，但技术显示可以买入] buy_signal {buy_signal} ")
                        buy_trades_holdings.append(
                            {
                                'date': date,
                                'type': 'buy',
                                'price': current_price,
                                'quantity': position,
                                'advice': advice,
                                'reason': '卖出理由：\n- ' + '\n- '.join("xxx")
                            }
                        )

                    holding_days += 1
                    # logging.info(f" sell_signal {buy_signal} ")
                    current_return = (current_price - entry_price) / entry_price

                    # 卖出条件
                    sell_reason = []
                    if current_return >= target_return:
                        sell_reason.append(f"达到目标收益：{current_return * 100:.2f}%")
                    if current_return <= stop_loss:
                        sell_reason.append(f"触及止损线：{current_return * 100:.2f}%")
                    if sell_signal >= 1 and holding_days > 5:
                        sell_reason.append("出现强烈卖出信号且持有超过5天")

                    # drawdown = (current_capital - peak_capital) / peak_capital  # 计算回撤
                    # logging.info(f" peak_capital {peak_capital} ")
                    # logging.info(f" drawdown {drawdown} ")

                    # # 检查最大回撤
                    # if drawdown > 0.10:  # 如果回撤超过10%
                    #     sell_reason.append("触及最大回撤限制")

                    if sell_reason:  # 有卖出理由时执行卖出
                        logging.info(f"[执行卖出] current_return {current_return} ")

                        # 计算当前资本
                        current_capital = capital + position * current_price
                        if current_capital >= peak_capital:
                            peak_return = (current_capital - initial_capital_) / initial_capital_
                            cumulative_gain += current_return
                            max_drawdown = 0
                            mode_flag = 0
                            stop_n_times = init_stop_n_times
                        else:
                            max_drawdown += current_return
                            mode_flag += 1

                        if current_capital > peak_capital:
                            logging.info(f"[执行卖出] update peak capital")

                        peak_capital = max(peak_capital, current_capital)  # 更新历史最高资金


                        logging.info(f"[执行卖出] current_return {current_return}")
                        logging.info(f"[执行卖出] cumulative_gain {cumulative_gain}")
                        logging.info(f"[执行卖出] peak_capital {peak_capital}")
                        logging.info(f"[执行卖出] current_capital {current_capital}")
                        logging.info(f"[执行卖出] max_drawdown {max_drawdown}")
                        logging.info(f"[执行卖出] mode_flag {mode_flag}")

                        capital += position * current_price
                        trades.append({
                            'date': date,
                            'type': 'sell',
                            'price': current_price,
                            'quantity': position,
                            'return': current_return,
                            'holding_days': holding_days,
                            'signals': sell_signal,
                            'advice': advice,
                            'reason': '卖出理由：\n- ' + '\n- '.join(sell_reason),
                            'capital' : capital
                        })
                        position = 0

            except Exception as e:
                logging.info(f"处理第 {i} 天数据时出错: {str(e)}")
                continue
        # 计算回测结果
        final_capital = capital + position * df['收盘'].iloc[-1]
        total_return = (final_capital - initial_capital_) / initial_capital_
        # 计算其他指标
        winning_trades = [t for t in trades if t.get('return', 0) > 0]
        losing_trades = [t for t in trades if t.get('return', 0) <= 0]
        win_rate = len(winning_trades) / (len(trades) / 2) if trades else 0

        # 计算年化收益率
        days_held = (df.index[-1] - df.index[0]).days
        annual_return = (1 + total_return) ** (365 / days_held) - 1 if days_held > 0 else 0

        # ============= 新增夏普比率计算部分 =============
        # 收集每日收益率
        daily_returns = []
        portfolio_value = initial_capital_

        for i in range(20, len(df)):
            current_price = df['收盘'].iloc[i]
            # 计算每日组合价值
            current_value = capital + position * current_price
            daily_return = (current_value - portfolio_value) / portfolio_value
            daily_returns.append(daily_return)
            portfolio_value = current_value

        # 计算夏普比率（年化）
        risk_free_rate = 0.03  # 假设无风险利率为3%
        sharpe_ratio = util.calculate_sharpe_ratio(
            daily_returns,
            risk_free_rate=risk_free_rate / 252,  # 转换为日无风险利率
            annualized=True
        )

        return {
            'stock_code': stock_code,
            'stock_name' : util.get_stock_name(stock_code),
            'initial_capital': initial_capital_,
            'final_capital': final_capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,  # 新增夏普比率
            'number_of_trades': len(trades) / 2,
            'win_rate': win_rate,
            'trades': trades,
            'position': position,  # 添加当前持仓状态
            'buy_trades_holdings':buy_trades_holdings
        }

    except Exception as e:
        logging.info(f"回测股票 {stock_code} 时出错: {str(e)}")
        import traceback
        logging.info(traceback.format_exc())
        return None

#20250317本周重点康强电子（调整到位？），盛和资源，
        # 华资实业（均线走势好，调整到位，7元考虑进？），海南椰岛（量能放大）
       # 仁智股份（是否突破 均线好，回撤会很大）
       # 协鑫集成 箱体震荡

def efi_backtesting():
    # recent_lhb_codes = util.get_recent_days_lhb_stocks()
    efi_email.send("Start Stock Backtesting")
    # 记录开始时间
    start_time = time.time()
    for i  in range(50):
        stock_codes = []
        # stock_codes = ['002506', '600178', '002119', '002122', '002448',
        #                '002703', '002673', '600392', '600489', '002261',
        #                '002264', '002861', '002881', '002629',
        #                '002506', '688041']
        #常规关注股票
        stock_codes =list(set(stock_codes + ['600178', '002119', '002448',
                       '002703', '600392', '002156', '002629','688041', '002506',
                       '002594', '000710', '600882', '600885', '600894',
                       '600191', ]))
        #龙虎榜最近4个月符合条件股票  0.27 0.90, win_rate > 0.47, 交易>6
        stock_codes = list(set(stock_codes + ['603121', '002379', '002765', '600539', '002119', '000429', '600184',
                                              '600397', '603228','002927', '603686', '600255', '603881', '600967',
                                              '002594', '002488', '600595','002112', '002361']))
        #days=90, 龙虎榜最近4个月符合条件股票  0.21 0.71, win_rate > 0.47, 交易>4
        stock_codes = list(set(stock_codes + ['603322', '600698', '600967', '002765', '600601']))
        # code = '002119'
        # stock_codes = [code]
        # stock_codes = ['600438', '603893', '000062', '002600', '000972', '002583', '000016',
        #                '600600','002031','300718','002611', '603166']
        # stock_codes = ['600178', '002629', '002119']
        # stock_codes = ['002594', '002119', '002861', '603986']
        # stock_codes, day_dragons = util.get_dragon_tiger_stocks(date="20250312")
        # stock_codes = util.get_recent_days_lhb_stocks(days=120)

        all_results = []
        daily_trades = []

        logging.info("\n开始回测买入信号股票...")

        for code in stock_codes:
            results = backtest_strategy(code,
                                        days_= 252,
                                        initial_capital_ = 1000000,
                                        target_return_ = 0.11,
                                        stop_loss_ = -0.03,
                                        init_stop_n_times = 0
                                        )
            if results is None:
                continue
            util.print_backtest_results(results)
            all_results.append(results)
            daily_trades.append(util.trade_daily(code, results['trades']))

        one_d_list = [item for sublist in daily_trades for item in sublist]
        efi_email.send(one_d_list)
        # 打印汇总统计
        util.print_summary_statistics(all_results)
        filtered_stocks = util.get_and_print_ideal_codes(all_results,
                                                         total_return_lower_bound=0.21,
                                                         total_return_upper_bound=0.91,
                                                         win_rate=0.47,
                                                         num_of_trades=6
                                                         )
        print(filtered_stocks)
        util.get_and_print_execution_time(start_time)
    # exit()
        # import pdb;pdb.set_trace()
        # time.sleep(400)
        # efi_email.send(  "Next round ...")
        # util.draw_stock_code_price(all_results)
        # # # # # # # # # # 可视化结果
        # util.visualize_backtest_results(all_results)
        # # # 打印统计摘要
        # util.logging.info_signal_summary(buy_signals, sell_signals, neutral_signals)

        # 可视化结果
        #util.visualize_signals(buy_signals, sell_signals, neutral_signals)


if __name__ == "__main__":
    efi_backtesting()