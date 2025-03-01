import efinance as ef
from datetime import datetime, timedelta
import efi_email
import util
import technical_indicator_analysis

def backtest_strategy(stock_code, days=200):
    """
    对单只股票进行回测
    :param stock_code: 股票代码
    :param days: 回测天数
    :return: 回测结果
    """
    try:
        # 使用 StockAnalyzer 类获取数据和计算指标
        analyzer = technical_indicator_analysis.StockAnalyzer(stock_code=stock_code, days=days)
        df = analyzer.df

        if df.empty:
            print(f"未获取到股票 {stock_code} 的历史数据")
            return None

        # 初始化回测参数
        initial_capital = 1000000  # 初始资金10万
        position = 0  # 持仓数量
        capital = initial_capital  # 当前资金
        trades = []  # 交易记录
        holding_days = 0  # 持仓天数
        target_return = 0.11   # 目标收益率
        stop_loss = -0.03  # 止损线
        entry_price = 0  # 买入价格

        # 跳过前20天，确保有足够数据计算指标
        for i in range(20, len(df)):
            try:
                # 更新分析器的数据窗口
                analyzer.df = df[i - 20:i + 1]

                date = df.index[i]
                current_price = df['收盘'].iloc[i]
                print(f"date {date}")
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
                    buy_signal += 1
                if "价格处于下降趋势" in advice:
                    sell_signal += 1
                if "量能配合良好" in advice:
                    buy_signal += 1
                if "量能配合显示卖压" in advice:
                    sell_signal += 1
                if "技术指标显示买入信号" in advice:
                    buy_signal += 1
                if "技术指标显示卖出信号" in advice:
                    sell_signal += 1


                # 交易逻辑
                if position == 0:  # 没有持仓
                    # print(f" buy_signal {buy_signal} ")
                    # print(f" sell_signal {buy_signal} 时出错: {str(sell_signal)}")
                    # print(sell_signal)
                    if buy_signal >= 2:  # 至少两个买入信号
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
                            ])
                        })
                        holding_days = 0

                elif position > 0:  # 持有仓位
                    holding_days += 1
                    # print(f" sell_signal {buy_signal} ")

                    current_return = (current_price - entry_price) / entry_price

                    # 卖出条件
                    sell_reason = []
                    if current_return >= target_return:
                        sell_reason.append(f"达到目标收益：{current_return * 100:.2f}%")
                    if current_return <= stop_loss:
                        sell_reason.append(f"触及止损线：{current_return * 100:.2f}%")
                    if sell_signal >= 2 and holding_days > 5:
                        sell_reason.append("出现强烈卖出信号且持有超过5天")

                    if sell_reason:  # 有卖出理由时执行卖出
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
                            'reason': '卖出理由：\n- ' + '\n- '.join(sell_reason)
                        })
                        position = 0

            except Exception as e:
                print(f"处理第 {i} 天数据时出错: {str(e)}")
                continue

        # 计算回测结果
        final_capital = capital + position * df['收盘'].iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital

        # 计算其他指标
        winning_trades = [t for t in trades if t.get('return', 0) > 0]
        losing_trades = [t for t in trades if t.get('return', 0) <= 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0

        # 计算年化收益率
        days_held = (df.index[-1] - df.index[0]).days
        annual_return = (1 + total_return) ** (365 / days_held) - 1 if days_held > 0 else 0

        return {
            'stock_code': stock_code,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'number_of_trades': len(trades),
            'win_rate': win_rate,
            'trades': trades,
            'position': position  # 添加当前持仓状态
        }

    except Exception as e:
        print(f"回测股票 {stock_code} 时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None



def efi_backtesting():
    # efi_email.send("begin efi backtesting ...")
    for i  in range(1):
        # stock_codes = ['002506', '600178', '002119', '002122', '002448',
        #                '002703', '002673', '600392', '600489', '002261',
        #                '002264', '002861', '002881', '002629',
        #                '002506', '688041']
        stock_codes = ['600178', '002119', '002122', '002448',
                       '002703', '600392', '002156',
                       '002264', '002861', '002629',
                       '688041', '002506', '002594']
        stock_codes = ['002119']
        # stock_codes = ['600438', '603893', '000062', '002600', '000972', '002583', '000016',
        #                '600600','002031','300718','002611', '603166']
        # stock_codes = ['600178', '002629', '002119']
        # stock_codes = ['002594', '002119', '002861', '603986']
        # stock_codes, day_dragons = util.get_dragon_tiger_stocks(date="20250228")

        all_results = []
        today_trades = []

        print("\n开始回测买入信号股票...")

        for code in stock_codes:
            results = backtest_strategy(code)
            util.print_backtest_results(results)
            all_results.append(results)
            # today_trades = util.trade_daily(code, results['trades'])

        efi_email.send(today_trades)

        # 打印汇总统计
        util.print_summary_statistics(all_results)
        # time.sleep(600)

        # # 可视化结果
        # util.visualize_backtest_results(all_results)
        # 打印统计摘要
        # util.print_signal_summary(buy_signals, sell_signals, neutral_signals)

        # 可视化结果
        #util.visualize_signals(buy_signals, sell_signals, neutral_signals)


if __name__ == "__main__":
    efi_backtesting()