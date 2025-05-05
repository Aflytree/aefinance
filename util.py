import time

import matplotlib.pyplot as plt
import akshare as ak
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import efinance as ef
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]


stock_code_name_dicts = {
            '600178': '东安动力',
            '002122': '汇洲智能',
            '002448': '中原内配',
            '002703': '浙江世宝',
            '600392': '盛和资源',
            '002156': '通富微电',
            '002264': '新 华 都',
            '002861': '瀛通通讯',
            '002629': '仁智股份',
            '688041': '海光信息',
            '002506': '协鑫集成',
            '002594': '比亚迪',
            '000710': '贝瑞基因',
}

def calculate_max_drawdown(result):
    """
    计算最大回撤
    :param portfolio_values: 投资组合在每个时间点的总资产价值列表
    :return: 最大回撤值
    """

    trades = result['trades']
    current_capity = 1000000
    portfolio_values = []


    for trade in trades:
        if trade['type'] == 'sell':
            current_capity = trade['capital']

        portfolio_values.append(current_capity)

    peak = 0
    if len(trades) == 0:
        peck = current_capity
    else:
        peak = portfolio_values[0]  # 初始峰值

    max_drawdown = 0

    for value in portfolio_values:
        if value > peak:
            peak = value  # 更新峰值
        drawdown = (peak - value) / peak  # 计算当前回撤
        max_drawdown = max(max_drawdown, drawdown)  # 更新最大回撤
    return max_drawdown


# 在文件顶部添加夏普比率计算函数
def calculate_sharpe_ratio(returns, risk_free_rate=0.0, annualized=False, periods_per_year=252):
    """
    计算夏普比率
    参数:
        returns: 收益率序列
        risk_free_rate: 无风险利率(默认0.0)
        annualized: 是否年化(默认False)
        periods_per_year: 年化周期数(默认252)
    返回:
        夏普比率
    """
    returns = np.asarray(returns)
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)  # 使用样本标准差

    if std_return == 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / std_return

    if annualized:
        sharpe = sharpe * np.sqrt(periods_per_year)

    return sharpe

def print_backtest_results(results):
    """
    打印回测结果
    """
    if not results:
        print("回测失败")
        return
    max_drawdown = calculate_max_drawdown(results)
    print("\n=== 回测结果 ===")
    print(f"股票代码: {results['stock_code']}")
    print(f"初始资金: {results['initial_capital']:,.2f}")
    print(f"最终资金: {results['final_capital']:,.2f}")
    print(f"总收益率: {results['total_return'] * 100:.2f}%")
    print(f"年化收益率: {results['annual_return'] * 100:.2f}%")
    print(f"夏普比率: {results['sharpe_ratio']}")
    print(f"交易次数: {results['number_of_trades']}")
    print(f"胜率: {results['win_rate'] * 100:.2f}%")
    print(f"最大回撤: {max_drawdown * 100:.2f}%")
    print("\n交易明细:")
    for trade in results['trades']:
        if trade['type'] == 'buy':
            print(f"买入 - 日期: {trade['date'].strftime('%Y-%m-%d')}, "
                  f"价格: {trade['price']:.2f}, "
                  f"数量: {trade['quantity']}")
                  # f"reason: {trade['reason']}"                   )
        else:
            print(f"卖出 - 日期: {trade['date'].strftime('%Y-%m-%d')}, "
                  f"价格: {trade['price']:.2f}, "
                  f"数量: {trade['quantity']}, "
                  f"收益率: {trade.get('return', 0) * 100:.2f}%, "
                  f"持仓天数: {trade.get('holding_days', 0)}")

def get_stock_names(stock_codes):
    """
    获取股票名称
    """
    try:
        stock_names = {}
        for code in stock_codes:
            # 使用akshare获取股票信息
            if code in stock_code_name_dicts.keys():
                stock_names[code] = stock_code_name_dicts[code]
                continue
            try:
                # 根据股票代码前缀判断市场
                if code.startswith('6'):
                    market = 'sh'
                else:
                    market = 'sz'
                stock_info = ak.stock_individual_info_em(symbol=f"{code}")
                # import pdb;pdb.set_trace()
                if not stock_info.empty:
                    stock_names[code] = stock_info.iloc[1]['value']
            except:
                stock_names[code] = ''
        return stock_names
    except Exception as e:
        print(f"获取股票名称时出错: {str(e)}")
        return {}

def visualize_backtest_results(all_results):
    """
    可视化多只股票的回测结果
    :param all_results: 包含多只股票回测结果的列表
    """
    # 过滤掉None结果
    valid_results = [r for r in all_results if r is not None]

    if not valid_results:
        print("没有有效的回测结果可供显示")
        return
    # 获取股票名称
    stock_codes = [r['stock_code'] for r in valid_results]
    stock_names = get_stock_names(stock_codes)
    print(stock_names)
    # 准备数据，添加股票名称
    stock_labels = [f"{code} {stock_names.get(code, '')}" for code in stock_codes]
    returns = [r['total_return'] * 100 for r in valid_results]
    annual_returns = [r['annual_return'] * 100 for r in valid_results]
    win_rates = [r['win_rate'] * 100 for r in valid_results]
    trade_counts = [r['number_of_trades'] for r in valid_results]

    # 使用默认样式
    plt.style.use('default')

    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    except:
        print("警告：可能无法正确显示中文")

    # 创建图表
    fig = plt.figure(figsize=(15, 10))

    # 1. 收益率对比
    ax1 = plt.subplot(221)
    bars = ax1.bar(stock_labels, returns, color='lightblue')
    ax1.set_title('总收益率对比')
    ax1.set_ylabel('收益率 (%)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%', ha='center', va='bottom')

    # 2. 年化收益率对比
    ax2 = plt.subplot(222)
    bars = ax2.bar(stock_labels, annual_returns, color='lightgreen')
    ax2.set_title('年化收益率对比')
    ax2.set_ylabel('年化收益率 (%)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%', ha='center', va='bottom')

    # 3. 胜率对比
    ax3 = plt.subplot(223)
    bars = ax3.bar(stock_labels, win_rates, color='salmon')
    ax3.set_title('交易胜率对比')
    ax3.set_ylabel('胜率 (%)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%', ha='center', va='bottom')

    # 4. 交易次数对比
    ax4 = plt.subplot(224)
    bars = ax4.bar(stock_labels, trade_counts, color='plum')
    ax4.set_title('交易次数对比')
    ax4.set_ylabel('交易次数')
    ax4.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}', ha='center', va='bottom')

    # 调整布局
    plt.tight_layout()

    # 显示图表
    plt.show()

    # 创建收益率曲线图
    plt.figure(figsize=(12, 6))

    # 设置不同的颜色和线型
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    line_styles = ['-', '--', ':', '-.']

    # 为每只股票绘制收益率曲线
    for i, result in enumerate(valid_results):
        trades = result['trades']
        dates = [t['date'] for t in trades]
        cumulative_returns = []
        current_return = 0

        for trade in trades:
            if trade['type'] == 'sell':
                current_return += trade['return'] * 100
            cumulative_returns.append(current_return)
        # print(cumulative_returns)
        # 使用股票代码和名称作为标签
        stock_code = result['stock_code']
        stock_name = stock_names.get(stock_code, '')
        label = f"{stock_code} {stock_name}"

        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        plt.plot(dates, cumulative_returns,
                 label=label,
                 marker='o',
                 color=color,
                 linestyle=line_style,
                 linewidth=2,
                 markersize=6)

    plt.title('累计收益率曲线')
    plt.xlabel('交易日期')
    plt.ylabel('累计收益率 (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)

    # 调整布局以适应图例
    plt.tight_layout()
    plt.show()

def print_summary_statistics(all_results):
    """
    打印汇总统计信息
    """
    valid_results = [r for r in all_results if r is not None]
    # import pdb;pdb.set_trace()
    if not valid_results:
        print("没有有效的回测结果可供统计")
        return

    print("\n=== 回测汇总统计 ===")
    print(f"测试股票数量: {len(valid_results)}")

    # 计算平均值
    avg_return = np.mean([r['total_return'] * 100 for r in valid_results])
    avg_annual_return = np.mean([r['annual_return'] * 100 for r in valid_results])
    avg_win_rate = np.mean([r['win_rate'] * 100 for r in valid_results])
    avg_trades = np.mean([r['number_of_trades'] for r in valid_results])

    print(f"\n平均统计:")
    print(f"平均收益率: {avg_return:.2f}%")
    print(f"平均年化收益率: {avg_annual_return:.2f}%")
    print(f"平均胜率: {avg_win_rate:.2f}%")
    print(f"平均交易次数: {avg_trades:.1f}")

    # 最佳表现
    best_return = max(valid_results, key=lambda x: x['total_return'])
    print(f"\n最佳表现股票:")
    print(f"股票代码: {best_return['stock_code']}")
    print(f"总收益率: {best_return['total_return'] * 100:.2f}%")
    print(f"年化收益率: {best_return['annual_return'] * 100:.2f}%")
    print(f"胜率: {best_return['win_rate'] * 100:.2f}%")
    print(f"交易次数: {best_return['number_of_trades']}")



def has_broken_high(stock_code, recent_years = 3):
    """
    recent_years: 是否突破最近recent_years的历史高点
    """

    # 获取股票的历史数据
    historical_data = ef.stock.get_quote_history(stock_code)  # 替换为实际股票代码
    if historical_data.empty:
        print(f"未找到股票 {stock_code} 的历史数据。")
        return False
    # 将日期列转换为 datetime 格式
    historical_data['日期'] = pd.to_datetime(historical_data['日期'])

    # 获取最近三年的数据
    three_years_ago = pd.Timestamp.today() - pd.DateOffset(years=recent_years)
    recent_data = historical_data[historical_data['日期'] >= three_years_ago]

    # 获取当前价格（假设为最新的收盘价）
    current_price = historical_data['收盘'].iloc[-1]

    # 获取历史最高点
    historical_high = recent_data['收盘'].max()

    # 判断是否突破历史最高点
    if current_price > historical_high:
        print(f"股票 {stock_code} 的当前价格 {current_price} 突破了最近 {recent_years} 年最高点 {historical_high}。")
        return True
    elif current_price > historical_high * 0.9:
        print(f"股票 {stock_code} 的当前价格 {current_price} 接近了最近 {recent_years} 年最高点 {historical_high}。")
        return True
    else:
        print(f"股票 {stock_code} 的当前价格 {current_price} 未突破最近 {recent_years} 年历史最高点 {historical_high}。")
        return False


def get_and_print_ideal_codes(all_results, total_return_lower_bound = 0.27,
                      total_return_upper_bound = 0.90,
                      win_rate = 0.47,
                      num_of_trades = 6
                      ):
    """
    打印汇总统计信息
    """
    valid_results = [r for r in all_results if r is not None]
    """
        获取total_return <30, 90>
            win_gate > 43%
            num_of_trades > 6
            的所有股票
    """
    # 筛选符合条件的股票
    filtered_stocks = []
    for stock in valid_results:
        if (total_return_lower_bound <= stock['total_return'] <= total_return_upper_bound and
                stock['win_rate'] >= win_rate and
                stock['number_of_trades'] > num_of_trades):
            if has_broken_high(stock['stock_code']):
                filtered_stocks.append({
                    'stock_code' : stock['stock_code'],
                    'total_return': stock['total_return'],
                    'win_rate': stock['win_rate'],
                    'number_of_trades': stock['number_of_trades']
                })

    # 打印符合条件的股票信息
    print("\n=== 符合条件的股票信息 ===")
    for index, stock in enumerate(filtered_stocks, start=1):
        print(f"股票 {index}:")
        print(f"  stock_code {stock['stock_code']}:")
        print(f"  总收益率: {stock['total_return'] * 100:.2f}%")
        print(f"  胜率: {stock['win_rate'] * 100:.2f}%")
        print(f"  交易次数: {stock['number_of_trades']}")
        print()  # 打印空行以便于阅读
    if len(filtered_stocks) == 0 :
        print("没有符合条件的理想股票")
    return filtered_stocks

def get_recent_trading_days(num_days=30):
    # 获取最近的交易日
    trading_days = pd.date_range(end=pd.Timestamp.today(), periods=num_days, freq='B')  # 'B' 表示工作日
    return trading_days.tolist()

def is_shenzhen_or_shanghai(stock_code):
    # 检查股票代码是否属于深市或沪市
    return stock_code.startswith(('0', '2', '6'))

def get_recent_days_lhb_stocks(days = 30):
    """
    获取最近多少天的龙虎榜数据
    """
    # 计算最近日期
    end_date = datetime.now()
    start_date = end_date - timedelta(days)
    trades_dates = get_recent_trading_days(days)

    stock_codes = []
    for dte in trades_dates:

        try:
            # 获取最近一个月的龙虎榜数据
            lhb_datas = ak.stock_lhb_detail_daily_sina(date=dte.strftime('%Y%m%d'))
            if not lhb_datas.empty:
                # 根据实际的列名调整
                stock_info = lhb_datas[['股票代码', '股票名称']].drop_duplicates()
                result = list(zip(stock_info['股票代码'], stock_info['股票名称']))

                # 打印获取到的数据数量
                print(f"\n {dte} 获取到 {len(result)} 只龙虎榜股票")

                for code, name in result:
                    if is_shenzhen_or_shanghai(code) :
                        stock_codes.append(code)
                # import pdb;pdb.set_trace()
                pass
        except Exception as e:
            print(f"获取数据失败，日期: {dte}，错误: {e}")
            time.sleep(3)

    return list(set(stock_codes))

def get_stock_name(code):
    """
    获取股票名称
    """
    stock_name = None
    try:
        if code in stock_code_name_dicts.keys():
            return  stock_code_name_dicts[code]
        # 根据股票代码前缀判断市场
        if code.startswith('6'):
            market = 'sh'
        else:
            market = 'sz'
        stock_info = ak.stock_individual_info_em(symbol=f"{code}")
        if not stock_info.empty:
            stock_name = stock_info.iloc[1]['value']
    except:
        stock_name = ''
    return stock_name

def get_dragon_tiger_stocks(date="20250210"):
    """
    获取最新龙虎榜股票
    """
    try:
        # 使用龙虎榜每日明细接口
        dragon_tiger_data = ak.stock_lhb_detail_daily_sina(date=date)
        print(dragon_tiger_data)
        # 打印数据结构信息
        print("\n数据列名:", dragon_tiger_data.columns.tolist())
        print("\n数据前几行:")
        print(dragon_tiger_data.head())

        # 提取股票代码和名称并去重
        if not dragon_tiger_data.empty:
            # 根据实际的列名调整
            stock_info = dragon_tiger_data[['股票代码', '股票名称']].drop_duplicates()
            result = list(zip(stock_info['股票代码'], stock_info['股票名称']))

            # 打印获取到的数据数量
            print(f"\n获取到 {len(result)} 只龙虎榜股票")
            stock_codes = []

            for code, name in result:
                stock_codes.append(code)

            return stock_codes, result
        else:
            print("未获取到龙虎榜数据")
            return []

    except Exception as e:
        print(f"获取龙虎榜数据时出错: {str(e)}")
        # 如果出错，打印所有可用的接口
        print("\n可用的龙虎榜相关接口:")
        for method in dir(ak):
            if 'lhb' in method.lower():
                print(f"- {method}")
        return []

def visualize_signals(buy_signals, sell_signals, neutral_signals):
    """可视化买卖信号统计"""
    # 准备数据
    categories = ['买入信号', '卖出信号', '观望信号']
    values = [len(buy_signals), len(sell_signals), len(neutral_signals)]

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 绘制柱状图
    bars = plt.bar(categories, values)

    # 设置颜色
    bars[0].set_color('red')
    bars[1].set_color('green')
    bars[2].set_color('gray')

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    # 在柱状图下方添加股票列表
    plt.figtext(0.1, 0.02, f"买入: {', '.join([f'{code} {name}' for code, name in buy_signals])}",
                wrap=True, fontsize=8)
    plt.figtext(0.4, 0.02, f"卖出: {', '.join([f'{code} {name}' for code, name in sell_signals])}",
                wrap=True, fontsize=8)
    plt.figtext(0.7, 0.02, f"观望: {', '.join([f'{code} {name}' for code, name in neutral_signals])}",
                wrap=True, fontsize=8)

    # 设置标题和标签
    plt.title('股票信号分布')
    plt.xlabel('信号类型')
    plt.ylabel('股票数量')

    # 调整布局以适应底部文本
    plt.subplots_adjust(bottom=0.2)

    # 显示图表
    plt.show()

def print_signal_summary(buy_signals, sell_signals, neutral_signals):
    """打印信号统计摘要"""
    total_stocks = len(buy_signals) + len(sell_signals) + len(neutral_signals)

    print("\n=== 股票信号统计 ===")
    print(f"总分析股票数: {total_stocks}")

    print(f"\n买入信号 ({len(buy_signals)}只):")
    for code, name in buy_signals:
        print(f"- {code} {name}")

    print(f"\n卖出信号 ({len(sell_signals)}只):")
    for code, name in sell_signals:
        print(f"- {code} {name}")

    print(f"\n观望信号 ({len(neutral_signals)}只):")
    for code, name in neutral_signals:
        print(f"- {code} {name}")


def trade_daily(code, trades):
    today_trades = []
    for trade in trades:
        # print(trade['trades'])
        today_trade = ""
        # import pdb;pdb.set_trace()
        if trade['type'] == 'buy':
            if trade['date'].strftime('%Y-%m-%d') == datetime.now().date().strftime('%Y-%m-%d'):
                stock_name = get_stock_name(code)
                today_trade += f" 买入 - 日期: {trade['date'].strftime('%Y-%m-%d')}, " \
                               f" 价格: {trade['price']:.2f} " \
                               f" 数量: {trade['quantity']}" \
                               f" reason: {trade.get('reason', 0)}\n" \
                               f" code: {code}" \
                               f" name: {stock_name}"
                today_trades.append(today_trade)
        else:
            if trade['date'].strftime('%Y-%m-%d') == datetime.now().date().strftime('%Y-%m-%d'):
                stock_name = get_stock_name(code)
                today_trade += f" 卖出 - 日期: {trade['date'].strftime('%Y-%m-%d')}, " \
                               f" 价格: {trade['price']:.2f}, " \
                               f" 数量: {trade['quantity']}, " \
                               f" 收益率: {trade.get('return', 0) * 100:.2f}%, " \
                               f" 持仓天数: {trade.get('holding_days', 0)}" \
                               f" reason: {trade.get('reason', 0)}\n" \
                               f" code: {code}" \
                               f" name: {stock_name}"
                today_trades.append(today_trade)
                # print("sell today")
    return today_trades
    # print(today_trades)



def draw_stock_code_price(all_results):

    for result in all_results:
        stock_code = result['stock_code']
        # 获取股票历史数据
        df = ef.stock.get_quote_history(stock_code)  # 替换为实际股票代码
        # 假设返回的数据是一个 DataFrame，包含 'date' 和 'close' 列
        # 将日期列转换为 datetime 格式
        df['日期'] = pd.to_datetime(df['日期'])

        # 筛选最近 200 天的数据
        end_date = df['日期'].max()
        start_date = end_date - timedelta(days=700)
        recent_data = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]

        # 绘制价格曲线图
        plt.figure(figsize=(10, 5))
        plt.plot(recent_data['日期'], recent_data['收盘'], label='收盘价', color='blue')

        # 标注买入和卖出
        for trade in result['trades']:
            trade_date = pd.to_datetime(trade['date'])
            if trade['type'] == 'buy':
                plt.scatter(trade_date, trade['price'], color='red',marker='x',
                            label='买入' if '买入' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif trade['type'] == 'sell':
                plt.scatter(trade_date, trade['price'], color='green',marker='s',
                            label='卖出' if '卖出' not in plt.gca().get_legend_handles_labels()[1] else "")

        for trade in result['buy_trades_holdings']:
            trade_date = pd.to_datetime(trade['date'])
            if trade['type'] == 'buy':
                plt.scatter(trade_date, trade['price'], color='black',alpha=0.3,
                            label='买入' if '买入' not in plt.gca().get_legend_handles_labels()[1] else "")
        gp_name = result['stock_name'] + '股票价格曲线图'
        plt.title(gp_name)
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.grid()
        plt.show()

def get_and_print_execution_time(start_time = 0):
    # 记录结束时间
    end_time = time.time()

    # 计算并打印执行时间
    execution_time = end_time - start_time
    # 将执行时间转换为小时、分钟和秒
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # 打印执行时间
    print(f"程序执行时间: {int(hours)} 小时 {int(minutes)} 分钟 {seconds:.6f} 秒")
    return execution_time