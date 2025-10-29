import technical_indicator_analysis
import logging
import util

def backtest_strategy(stock_code,
                      bg = '20240223',
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
        analyzer = technical_indicator_analysis.StockAnalyzer(stock_code=stock_code, beg= bg)
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

        # 在回测初始化部分添加
        cumulative_loss = 0.0  # 累计亏损
        stop_loss_count = 0  # 止损次数
        max_drawdown = 0.0  # 最大回撤
        # 在回测参数中添加
        consecutive_stop_loss = 0
        max_consecutive_stop_loss = 3
        # 计算波动率指标 (提前计算)
        df['volatility'] = (df['最高'] - df['最低']).rolling(14).mean() / df['收盘']
        # 跳过前20天，确保有足够数据计算指标
        for i in range(20, len(df)):
            try:
                # 更新分析器的数据窗口
                analyzer.df = df[i - 20:i + 1]

                date = df.index[i]
                current_price = df['收盘'].iloc[i]
                current_volatility = df['volatility'].iloc[i]
                # # import pdb;pdb.set_trace()
                # # ========== 新增：计算波动率指标 ==========
                # # 计算真实波幅(True Range)
                # df['prev_close'] = df['收盘'].shift(1)
                # df['tr'] = np.maximum(
                #     df['最高'] - df['最低'],
                #     np.maximum(
                #         abs(df['最高'] - df['prev_close']),
                #         abs(df['最低'] - df['prev_close'])
                #     )
                # )
                # # 计算14日ATR(平均真实波幅)
                # df['atr'] = df['tr'].rolling(14).mean()
                # # 标准化波动率(ATR/收盘价)
                # df['volatility'] = df['atr'] / df['收盘']
                # # 清理临时列
                # df.drop(['prev_close', 'tr'], axis=1, inplace=True)
                # ========================================

                # logging.info(f"*****************date {date} **********************")
                # 获取交易建议
                advice = analyzer.get_trading_advice1()
                # import pdb;pdb.set_trace()
                buy_signal, sell_signal = util.parse_trading_signals(advice)

                # 交易逻辑
                if position == 0:  # 没有持仓
                    # logging.info(f"[没有仓位] max_drawdown {max_drawdown} ")
                    # logging.info(f"[没有仓位] drawdown_threshold {drawdown_threshold} ")

                    # if buy_signal >= 2 and util.market_condition_filter(df, i):  # 至少达到有效买入信号
                    if buy_signal >= 2 :  # 至少达到有效买入信号
                        # position_size = util.calculate_position_size(capital, current_price, df['volatility'].illoc[i])
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
                        logging.debug(f"[执行买入] buy_signal {buy_signal} ")
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
                            'reason': buy_signal
                        })
                        holding_days = 0




                elif position > 0:  # 持有仓位

                    if buy_signal >= 2:
                        logging.debug(f"[持有仓位，但技术显示可以买入] buy_signal {buy_signal}")

                        buy_trades_holdings.append(

                            {

                                'date': date,

                                'type': 'buy',

                                'price': current_price,

                                'quantity': position,

                                'advice': advice,

                                'reason': ''.join("xxx")

                            }

                        )

                    holding_days += 1

                    current_return = (current_price - entry_price) / entry_price

                    # 卖出条件 - 严格止损优先

                    sell_reason = []

                    is_stop_loss = False

                    actual_sell_price = current_price  # 默认按当前价格卖出

                    actual_return = current_return  # 默认按当前收益率

                    # 严格止损逻辑：如果当前亏损超过止损点，按止损点价格计算

                    if current_return <= stop_loss:

                        # 按止损点价格卖出，而不是实际收盘价

                        actual_sell_price = entry_price * (1 + stop_loss)

                        actual_return = stop_loss  # 亏损严格按止损点计算

                        sell_reason.append(f"严格执行止损：{actual_return * 100:.2f}%")

                        is_stop_loss = True

                        logging.debug(
                            f"[严格止损] 避免更大亏损：收盘亏损{current_return * 100:.2f}%，按止损点{stop_loss * 100:.2f}%执行")


                    elif current_return >= target_return:

                        sell_reason.append(f"达到目标收益：{current_return * 100:.2f}%")

                    elif sell_signal >= 1 and holding_days > 5:

                        sell_reason.append("出现强烈卖出信号且持有超过5天")

                    if sell_reason:

                        # 计算当前资本 - 使用实际卖出价格

                        current_capital = capital + position * actual_sell_price

                        # 严格的止损统计

                        if is_stop_loss:
                            cumulative_loss += actual_return  # 按止损点计算的亏损

                            stop_loss_count += 1
                            # 连续止损保护
                            if consecutive_stop_loss >= max_consecutive_stop_loss:
                                logging.info(f"[连续止损保护] 暂停交易，连续{consecutive_stop_loss}次止损")
                                # 可以跳过接下来几天的买入信号

                            logging.debug(f"[严格止损执行] 实际亏损: {actual_return * 100:.2f}%")

                            logging.debug(f"[严格止损执行] 避免亏损: {(current_return - actual_return) * 100:.2f}%")

                            logging.debug(f"[止损统计] 累计亏损: {cumulative_loss * 100:.2f}%")

                            logging.debug(f"[止损统计] 止损次数: {stop_loss_count}")
                        else:
                            consecutive_stop_loss = 0  # 盈利交易重置计数

                        # 更新最大回撤 - 基于资金曲线，而不是单笔交易收益率

                        if current_capital > peak_capital:

                            peak_capital = current_capital

                            max_drawdown = 0  # 创新高时重置最大回撤

                            logging.debug(f"[执行卖出] 更新峰值资金: {peak_capital:.2f}")

                        else:

                            # 计算当前回撤

                            current_drawdown = (peak_capital - current_capital) / peak_capital

                            max_drawdown = max(max_drawdown, current_drawdown)  # 注意这里是max不是min

                        # 盈利交易的累计收益统计

                        if not is_stop_loss and actual_return > 0:
                            cumulative_gain += actual_return

                        # 记录交易 - 使用实际执行的价格和收益率

                        trade_record = {

                            'date': date,

                            'type': 'sell',

                            'price': actual_sell_price,  # 记录实际卖出价格

                            'quantity': position,

                            'return': actual_return,  # 记录实际收益率

                            'holding_days': holding_days,

                            'signals': sell_signal,

                            'advice': advice,

                            'reason': ''.join(sell_reason),

                            'capital': current_capital,  # 记录卖出后的资金

                            'is_stop_loss': is_stop_loss,

                            'avoided_loss': (current_return - actual_return) if is_stop_loss else 0  # 避免的额外亏损

                        }

                        trades.append(trade_record)

                        # 执行卖出 - 使用实际卖出价格

                        capital += position * actual_sell_price

                        position = 0

                        holding_days = 0

                        logging.debug(f"[执行卖出] 实际收益率: {actual_return * 100:.2f}%")

                        logging.debug(f"[执行卖出] 当前资金: {current_capital:.2f}")

                        logging.debug(f"[执行卖出] 峰值资金: {peak_capital:.2f}")

                        logging.debug(f"[执行卖出] 当前回撤: {current_drawdown * 100:.2f}%")

                        logging.debug(f"[执行卖出] 最大回撤: {max_drawdown * 100:.2f}%")

                        if is_stop_loss:
                            logging.debug(f"[止损效果] 避免额外亏损: {(current_return - actual_return) * 100:.2f}%")
            except Exception as e:
                logging.debug(f"处理第 {i} 天数据时出错: {str(e)}")
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
        logging.debug(f"回测股票 {stock_code} 时出错: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return None
