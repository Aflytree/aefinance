from gc import enable

import technical_indicator_analysis
import logging
import util

STRICT_STOP_LOSS = False

# 优化开关
POSITION_FRACTION = 0.5          # 半仓
REQUIRE_WEEKLY_FILTER = True     # 周线硬过滤
USE_ATR_STOP = True              # ATR 动态止损
ATR_STOP_MULT = 2.0              # 止损 = max(固定止损, ATR倍数)
ATR_BUFFER_DAYS = 3              # 买入后前 N 天止损放宽
ATR_BUFFER_MULT = 1.5            # 缓冲期额外倍数
ENABLE_STOP_COOLDOWN = True      # 连续止损冷静期
MAX_CONSECUTIVE_STOP_LOSS = 3
COOLDOWN_DAYS = 10

# A股交易成本（近似）
ENABLE_TRADE_FEE = True
COMMISSION_RATE = 0.00025        # 佣金万2.5（买卖各收）
COMMISSION_MIN = 5.0             # 单笔佣金最低 5 元
STAMP_TAX_RATE = 0.0005          # 印花税万5（仅卖出）
TRANSFER_FEE_RATE = 0.00001      # 过户费万0.1（买卖，沪市为主，简化统一收）
SLIPPAGE_RATE = 0.0001           # 滑点万1（买上浮/卖下浮）


def calc_buy_fee(amount: float) -> float:
    """买入费用 = 佣金 + 过户费（金额按成交额）。"""
    if amount <= 0:
        return 0.0
    commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
    transfer = amount * TRANSFER_FEE_RATE
    return commission + transfer


def calc_sell_fee(amount: float) -> float:
    """卖出费用 = 佣金 + 印花税 + 过户费。"""
    if amount <= 0:
        return 0.0
    commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
    stamp = amount * STAMP_TAX_RATE
    transfer = amount * TRANSFER_FEE_RATE
    return commission + stamp + transfer


def apply_buy_slippage(price: float) -> float:
    return price * (1 + SLIPPAGE_RATE) if ENABLE_TRADE_FEE else price


def apply_sell_slippage(price: float) -> float:
    return price * (1 - SLIPPAGE_RATE) if ENABLE_TRADE_FEE else price


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
        weekly_df = analyzer.weekly_df

        if df.empty:
            print(f"未获取到股票 {stock_code} 的历史数据")
            return None

        # 初始化回测参数
        position = 0  # 持仓数量
        capital = initial_capital_  # 当前资金
        trades = []  # 交易记录
        holding_days = 0  # 持仓天数
        target_return = target_return_   # 目标收益率
        stop_loss = stop_loss_  # 固定止损下限（与 ATR 取更宽者）
        entry_price = 0  # 买入价格（含滑点后成交价）
        entry_atr = 0.0
        entry_cost = 0.0  # 买入本金+买入费用
        entry_fee = 0.0
        total_fees = 0.0
        cumulative_gain = 0
        peak_capital = initial_capital_  # 记录历史最高资金
        buy_trades_holdings = []
        cumulative_loss = 0.0  # 累计亏损
        stop_loss_count = 0  # 止损次数
        max_drawdown = 0.0  # 最大回撤
        consecutive_stop_loss = 0
        cooldown_until_idx = -1

        # 波动率 / ATR
        df['volatility'] = (df['最高'] - df['最低']).rolling(14).mean() / df['收盘']
        prev_close = df['收盘'].shift(1)
        tr = (df['最高'] - df['最低']).combine(
            (df['最高'] - prev_close).abs(), max
        ).combine(
            (df['最低'] - prev_close).abs(), max
        )
        df['atr'] = tr.rolling(14).mean()

        # 净值路径（用于更合理的夏普）
        equity_curve = []

        # 跳过前20天，确保有足够数据计算指标
        for i in range(20, len(df)):
            try:
                analyzer.df = df[i - 20:i + 1]
                date = df.index[i]
                current_price = df['收盘'].iloc[i]
                mark_to_market = capital + position * current_price
                equity_curve.append(mark_to_market)

                advice = analyzer.get_trading_advice1()
                buy_signal, sell_signal = util.parse_trading_signals(advice)

                if position == 0:  # 没有持仓
                    if ENABLE_STOP_COOLDOWN and i < cooldown_until_idx:
                        continue
                    if buy_signal >= 2:  # 至少达到有效买入信号
                        weekly_analysis = analyzer.analyze_weekly_moving_averages(date)
                        weekly_advice = ""
                        if REQUIRE_WEEKLY_FILTER:
                            if not weekly_analysis or 'error' in weekly_analysis:
                                continue
                            vals = weekly_analysis.get('current_values', {})
                            ma5 = vals.get('MA5')
                            ma10 = vals.get('MA10')
                            ma20 = vals.get('MA20')
                            if ma5 is None or ma10 is None or ma20 is None:
                                continue
                            if not (ma5 > ma20 and ma5 > ma10):
                                continue
                            weekly_advice = '五周线在20/10周线上方'

                        buy_trades_holdings.append(
                            {
                                'date': date,
                                'type': 'buy',
                                'price': current_price,
                                'quantity': position,
                                'advice': advice,
                                'reason': '：\n- ' + '\n- '.join(advice)
                            }
                        )

                        logging.debug(f"[执行买入] buy_signal {buy_signal} ")
                        buy_px = apply_buy_slippage(current_price)
                        # 半仓：预留买入费用后再算股数
                        spend_budget = capital * POSITION_FRACTION
                        if ENABLE_TRADE_FEE:
                            # 反推可买金额，使 成交额+费用 ≈ budget
                            # 近似：成交额 ≈ budget / (1 + 费率)，再按最低佣金修正
                            est_amount = spend_budget / (1 + COMMISSION_RATE + TRANSFER_FEE_RATE)
                            position = int(est_amount / buy_px)
                        else:
                            position = int(spend_budget / buy_px)
                        if position <= 0:
                            continue
                        buy_amount = position * buy_px
                        buy_fee = calc_buy_fee(buy_amount) if ENABLE_TRADE_FEE else 0.0
                        total_cost = buy_amount + buy_fee
                        if total_cost > capital:
                            # 再缩一手
                            position = int((capital - COMMISSION_MIN) / (
                                buy_px * (1 + COMMISSION_RATE + TRANSFER_FEE_RATE)
                            )) if ENABLE_TRADE_FEE else int(capital / buy_px)
                            if position <= 0:
                                continue
                            buy_amount = position * buy_px
                            buy_fee = calc_buy_fee(buy_amount) if ENABLE_TRADE_FEE else 0.0
                            total_cost = buy_amount + buy_fee
                            if total_cost > capital:
                                continue

                        entry_price = buy_px
                        entry_fee = buy_fee
                        entry_cost = total_cost
                        atr_now = df['atr'].iloc[i]
                        entry_atr = float(atr_now) if atr_now == atr_now else 0.0
                        capital -= total_cost
                        total_fees += buy_fee
                        trades.append({
                            'date': date,
                            'type': 'buy',
                            'price': buy_px,
                            'raw_price': current_price,
                            'quantity': position,
                            'signals': buy_signal,
                            'advice': advice,
                            'reason': ''.join(weekly_advice),
                            'position_fraction': POSITION_FRACTION,
                            'entry_atr': entry_atr,
                            'fee': buy_fee,
                            'amount': buy_amount,
                        })
                        holding_days = 0
                elif position > 0:  # 持有仓位
                    if buy_signal >= 2:
                        logging.debug(f"[持有仓位，但技术显示可以买入] buy_signal {buy_signal}")
                        buy_trades_holdings.append({
                            'date': date,
                            'type': 'buy',
                            'price': current_price,
                            'quantity': position,
                            'advice': advice,
                            'reason': ''.join(advice)
                        })

                    holding_days += 1
                    # 价格涨跌幅（不含费）用于触发止损/止盈阈值
                    price_return = (current_price - entry_price) / entry_price
                    sell_reason = []
                    is_stop_loss = False
                    actual_sell_price = apply_sell_slippage(current_price)
                    # 动态止损阈值
                    dyn_stop = stop_loss
                    if USE_ATR_STOP and entry_price > 0 and entry_atr > 0:
                        atr_pct = entry_atr / entry_price
                        mult = ATR_STOP_MULT
                        if holding_days <= ATR_BUFFER_DAYS:
                            mult *= ATR_BUFFER_MULT
                        atr_stop = -mult * atr_pct
                        dyn_stop = max(min(stop_loss, atr_stop), -0.08)

                    if STRICT_STOP_LOSS and price_return <= dyn_stop:
                        actual_sell_price = apply_sell_slippage(entry_price * (1 + dyn_stop))
                        sell_reason.append(f"严格执行止损：{dyn_stop * 100:.2f}%")
                        is_stop_loss = True
                    elif not STRICT_STOP_LOSS and price_return <= dyn_stop:
                        sell_reason.append(
                            f"触发止损：{price_return * 100:.2f}%(阈值{dyn_stop * 100:.2f}%)"
                        )
                        is_stop_loss = True
                    elif price_return >= target_return:
                        sell_reason.append(f"达到目标收益：{price_return * 100:.2f}%")
                    elif sell_signal >= 4 and holding_days > 7:
                        sell_reason.append("出现强烈卖出信号且持有超过7天")

                    if sell_reason:
                        sell_amount = position * actual_sell_price
                        sell_fee = calc_sell_fee(sell_amount) if ENABLE_TRADE_FEE else 0.0
                        proceeds = sell_amount - sell_fee
                        actual_return = (proceeds - entry_cost) / entry_cost if entry_cost else 0.0
                        current_capital = capital + proceeds
                        total_fees += sell_fee

                        if is_stop_loss:
                            cumulative_loss += actual_return
                            stop_loss_count += 1
                            consecutive_stop_loss += 1
                            if ENABLE_STOP_COOLDOWN and consecutive_stop_loss >= MAX_CONSECUTIVE_STOP_LOSS:
                                cooldown_until_idx = i + COOLDOWN_DAYS
                                logging.info(
                                    f"[连续止损保护] {stock_code} 暂停{COOLDOWN_DAYS}天，"
                                    f"连续{consecutive_stop_loss}次止损"
                                )
                                consecutive_stop_loss = 0
                        else:
                            consecutive_stop_loss = 0

                        if current_capital > peak_capital:
                            peak_capital = current_capital
                            max_drawdown = 0
                        else:
                            current_drawdown = (peak_capital - current_capital) / peak_capital
                            max_drawdown = max(max_drawdown, current_drawdown)

                        if not is_stop_loss and actual_return > 0:
                            cumulative_gain += actual_return

                        trade_record = {
                            'date': date,
                            'type': 'sell',
                            'price': actual_sell_price,
                            'raw_price': current_price,
                            'quantity': position,
                            'return': actual_return,
                            'price_return': price_return,
                            'holding_days': holding_days,
                            'signals': sell_signal,
                            'advice': advice,
                            'reason': ''.join(sell_reason),
                            'capital': current_capital,
                            'is_stop_loss': is_stop_loss,
                            'strict_stop_loss': STRICT_STOP_LOSS,
                            'avoided_loss': (price_return - dyn_stop) if (
                                        is_stop_loss and STRICT_STOP_LOSS) else 0,
                            'dyn_stop': dyn_stop,
                            'fee': sell_fee,
                            'amount': sell_amount,
                            'entry_cost': entry_cost,
                        }

                        trades.append(trade_record)
                        capital += proceeds
                        position = 0
                        holding_days = 0
                        entry_cost = 0.0
                        entry_fee = 0.0
            except Exception as e:
                logging.debug(f"处理第 {i} 天数据时出错: {str(e)}")
                continue

        final_capital = capital + position * df['收盘'].iloc[-1]
        # 未平仓不计卖出费用（与市值盯市一致）
        total_return = (final_capital - initial_capital_) / initial_capital_
        winning_trades = [t for t in trades if t.get('return', 0) > 0]
        losing_trades = [t for t in trades if t.get('return', 0) <= 0]
        win_rate = len(winning_trades) / (len(trades) / 2) if trades else 0

        days_held = (df.index[-1] - df.index[0]).days
        annual_return = (1 + total_return) ** (365 / days_held) - 1 if days_held > 0 else 0

        daily_returns = []
        if equity_curve:
            prev_v = equity_curve[0]
            for v in equity_curve[1:]:
                daily_returns.append((v - prev_v) / prev_v if prev_v else 0.0)
                prev_v = v

        risk_free_rate = 0.03
        sharpe_ratio = util.calculate_sharpe_ratio(
            daily_returns,
            risk_free_rate=risk_free_rate / 252,
            annualized=True
        ) if daily_returns else 0.0

        data_as_of = df.index.max().date() if len(df) else None
        today = __import__('datetime').datetime.now().date()
        return {
            'stock_code': stock_code,
            'stock_name' : util.get_stock_name(stock_code),
            'initial_capital': initial_capital_,
            'final_capital': final_capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'number_of_trades': len(trades) / 2,
            'win_rate': win_rate,
            'trades': trades,
            'position': position,
            'buy_trades_holdings':buy_trades_holdings,
            'total_fees': total_fees,
            'data_as_of': data_as_of,
            'data_is_today': bool(data_as_of and data_as_of >= today),
        }

    except Exception as e:
        logging.debug(f"回测股票 {stock_code} 时出错: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return None
