import time

import matplotlib.pyplot as plt
import akshare as ak
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
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
            '002927': '泰永长征',
            '600539': '狮头股份',
            '600255': '鑫科材料',
            '002361': '神剑股份',
            '002488': '金固股份',
            '000429': '粤高速A',
            '603881': '数据港',
            '600882': '妙可蓝多',
            '600601': '方正科技',
            '603121': '华培动力',
            '002112': '三变科技',
            '600595': '中孚实业',
            '600397': '江钨装备',
            '600191': '华资实业',
            '600894': '广日股份',
            '002379': '宏创控股',
            '603228': '景旺电子',
            '600698': '湖南天雁',
            '603322': '超讯通信',
            '603686': '福龙马',
            '600967': '内蒙一机',
            '002765': '蓝黛科技',
            '600184': '光电股份',
            '002119': '康强电子',
            '600885': '宏发股份',
            '600415': '小商品城',
            '600689': '上海三毛',
            '002278': '神开股份',
            '603336': '宏辉果蔬',
            '603839': '安正时尚',
}

# 修改信号解析部分，与 compute_signal_scores 输出的标签对齐
def parse_trading_signals(advice):
    buy_signal = 0
    sell_signal = 0

    signal_weights = {
        "强烈买入信号": 3,
        "买入信号": 2,
        "强烈卖出信号": 3,
        "卖出信号": 2,
        "均线多头排列": 1,
        "均线多头共振": 1,
        "MA金叉": 1,
        "价格处于上升趋势": 1,
        "价格处于下降趋势": -2,
        "量能配合良好": 1,
        "量能配合显示卖压": -1,
        "MACD金叉": 2,
        "MACD死叉": -2,
        "KDJ金叉": 1,
        "KDJ超买": -2,
        "RSI健康区间": 1,
        "超卖区域": 1,
        "超买区域": -2,
        "布林带下轨支撑": 1,
        "布林带上轨压力": -1,
        "技术指标显示买入信号": 1,
        "技术指标显示卖出信号": -1,
        "双底形态": 2,
        "双头形态": -2,
        "长上影线压力": -1,
        "突破阻力位": 1,
        "跌破支撑位": -1,
    }

    for pattern, weight in signal_weights.items():
        if pattern not in advice:
            continue
        if weight > 0:
            buy_signal += weight
        else:
            sell_signal += abs(weight)

    return buy_signal, sell_signal


# 修改买入部分的仓位管理
def calculate_position_size(capital, current_price, volatility, max_risk=0.02):
    """
    基于波动率和风险管理的仓位计算
    max_risk: 单笔交易最大风险比例(默认2%)
    """
    # 计算ATR或其他波动率指标
    atr = volatility * current_price  # 假设volatility是标准化波动率

    # 计算风险调整后的仓位
    risk_per_share = atr
    max_loss = capital * max_risk
    position = int(max_loss / risk_per_share)

    # 确保不超过可用资金
    max_by_capital = int(capital * 0.9 / current_price)  # 最多使用90%资金
    return min(position, max_by_capital)


# 在买入信号前增加市场环境判断
def market_condition_filter(df, current_index):
    """
    检查市场整体趋势和波动性
    返回: Boolean (True表示适合交易)
    """
    # 检查大盘趋势 (示例使用20日均线)
    ma_20 = df['收盘'].rolling(20).mean()
    current_ma = ma_20.iloc[current_index]
    price_above_ma = df['收盘'].iloc[current_index] > current_ma

    # 检查波动率 (示例使用ATR)
    atr = (df['最高'] - df['最低']).rolling(14).mean()
    current_atr = atr.iloc[current_index]
    normalized_atr = current_atr / df['收盘'].iloc[current_index]

    # 过滤条件
    if not price_above_ma:
        return False
    if normalized_atr > 0.1:  # 波动过大
        return False
    if normalized_atr < 0.02:  # 波动过小
        return False

    return True


# 动态止盈止损策略
def dynamic_exit_strategy(current_return, holding_days, sell_signal, volatility):
    sell_reasons = []

    # 基础止盈止损
    base_target = 0.11
    base_stop_loss = -0.03

    # 根据波动率调整目标
    adjusted_target = base_target * (1 + volatility)
    # adjusted_stop_loss = base_stop_loss * (1 + volatility)
    adjusted_stop_loss = base_stop_loss

    # 根据持有天数调整 (时间衰减效应)
    time_factor = min(1.0, holding_days / 10)  # 10天后不再增加
    final_target = adjusted_target * (1 + time_factor * 0.2)  # 最多增加20%

    if current_return >= final_target:
        sell_reasons.append(f"达到动态目标收益：{current_return * 100:.2f}%")
    elif current_return <= adjusted_stop_loss:
        sell_reasons.append(f"触及动态止损线：{current_return * 100:.2f}%")
    elif sell_signal >= 2 and holding_days > 3:  # 强烈信号且持有3天以上
        sell_reasons.append("强烈卖出信号")
    elif sell_signal >= 1 and holding_days > 7:  # 普通信号且持有7天以上
        sell_reasons.append("卖出信号且持有周期足够")

    return sell_reasons


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
    # import pdb;pdb.set_trace()
    # return np.float16(sharpe)
    return sharpe


def calculate_holding_days_stats(results):
    """
    计算平均持股天数统计并写入results
    """
    # 计算持股天数统计
    all_holding_days = []
    winning_days = []
    losing_days = []

    for trade in results['trades']:
        if trade['type'] == 'sell':
            days = trade.get('holding_days', 0)
            all_holding_days.append(days)
            if trade.get('return', 0) > 0:
                winning_days.append(days)
            else:
                losing_days.append(days)

    # 计算平均值
    avg_all_days = sum(all_holding_days) / len(all_holding_days) if all_holding_days else 0
    avg_win_days = sum(winning_days) / len(winning_days) if winning_days else 0
    avg_loss_days = sum(losing_days) / len(losing_days) if losing_days else 0

    # 将结果写入results
    results['avg_holding_days'] = avg_all_days
    results['avg_winning_holding_days'] = avg_win_days
    results['avg_losing_holding_days'] = avg_loss_days
    results['total_trades_count'] = len(all_holding_days)
    results['winning_trades_count'] = len(winning_days)
    results['losing_trades_count'] = len(losing_days)

    return results

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
    print(f"股票名称: {get_stock_name(results['stock_code'])}")
    print(f"初始资金: {results['initial_capital']:,.2f}")
    print(f"最终资金: {results['final_capital']:,.2f}")
    print(f"总收益率: {results['total_return'] * 100:.2f}%")
    print(f"年化收益率: {results['annual_return'] * 100:.2f}%")
    print(f"夏普比率: {format(results['sharpe_ratio'], '.4f')}")
    print(f"交易次数: {results['number_of_trades']}")
    print(f"胜率: {results['win_rate'] * 100:.2f}%")
    print(f"最大回撤: {max_drawdown * 100:.2f}%")
    print(f"平均持股天数: {results['avg_holding_days']:.1f}天")
    print(f"盈利平均持股天数: {results['avg_winning_holding_days']:.1f}天")
    print(f"亏损平均持股天数: {results['avg_losing_holding_days']:.1f}天")
    if results.get('total_fees') is not None:
        print(f"累计交易费用: {results['total_fees']:,.2f}")
    print("\n交易明细:")
    for trade in results['trades']:
        fee = trade.get('fee', 0) or 0
        if trade['type'] == 'buy':
            print(f"买入 - 日期: {trade['date'].strftime('%Y-%m-%d')}, "
                  f"价格: {trade['price']:.2f}, "
                  f"数量: {trade['quantity']}, "
                  f"费用: {fee:.2f}, "
                  f"reason: {trade['reason']}")
        else:
            print(f"卖出 - 日期: {trade['date'].strftime('%Y-%m-%d')}, "
                  f"价格: {trade['price']:.2f}, "
                  f"数量: {trade['quantity']}, "
                  f"收益率: {trade.get('return', 0) * 100:.2f}%, "
                  f"费用: {fee:.2f}, "
                  f"持仓天数: {trade.get('holding_days', 0)}, "
                  f"reason: {trade['reason']}")

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


def _signal_day_str(results, signal_day=None):
    """信号日：优先显式传入，其次结果里的 data_as_of，否则日历今天。"""
    if signal_day is not None:
        if hasattr(signal_day, "strftime"):
            return signal_day.strftime("%Y-%m-%d")
        return str(signal_day)
    data_as_of = results.get("data_as_of")
    if data_as_of is not None:
        if hasattr(data_as_of, "strftime"):
            return data_as_of.strftime("%Y-%m-%d")
        return str(data_as_of)
    return datetime.now().date().strftime("%Y-%m-%d")


def trade_daily(code, results, signal_day=None):
    """统计信号日（默认=最新K线日）的买卖。"""
    trades = results['trades']
    day = _signal_day_str(results, signal_day)
    today_trades = []
    for trade in trades:
        today_trade = ""
        if trade['type'] == 'buy':
            if trade['date'].strftime('%Y-%m-%d') == day:
                stock_name = get_stock_name(code)
                today_trade += f" 买入 - 日期: {trade['date'].strftime('%Y-%m-%d')}, " \
                               f" 价格: {trade['price']:.2f} （信号日开仓，已计入当前持仓）\n" \
                               f" 总收益 : {results['total_return'] * 100: .2f}%" \
                               f" 夏普: {format(results['sharpe_ratio'], '.4f')}" \
                               f" reson: {results['trades'][-1]['reason']}\n" \
                               f" code: {code}" \
                               f" name: {stock_name}"
                today_trades.append(today_trade)
        else:
            if trade['date'].strftime('%Y-%m-%d') == day:
                stock_name = get_stock_name(code)
                today_trade += f" 卖出 - 日期: {trade['date'].strftime('%Y-%m-%d')}, " \
                               f" 价格: {trade['price']:.2f}, \n" \
                               f" 本次收益率: {trade.get('return', 0) * 100:.2f}%, " \
                               f" 持仓天数: {trade.get('holding_days', 0)}\n" \
                               f" reason: {trade.get('reason', 0)}\n" \
                               f" code: {code}" \
                               f" name: {stock_name}"
                today_trades.append(today_trade)
    return today_trades


def last_busy(code, results, signal_day=None):
    """当前持仓：最后一笔是买入即视为仍持仓（含信号日当天开仓）。"""
    trades = results['trades']
    current_hold_ = []
    if not trades:
        return current_hold_
    day = _signal_day_str(results, signal_day)
    if trades[-1]['type'] == 'buy':
        stock_name = get_stock_name(code)
        buy_date = trades[-1]['date'].strftime('%Y-%m-%d')
        tag = "（今日开仓）" if buy_date == day else ""
        last_buy_ = (
            f" 买入: {buy_date}, "
            f" 价格: {trades[-1]['price']:.2f}{tag}\n"
            f" 总收益 : {results['total_return'] * 100: .2f}%"
            f" 夏普: {format(results['sharpe_ratio'], '.2f')}"
            f" 胜率: {results['win_rate'] * 100:.2f}%\n"
            f" code: {code}"
            f" name: {stock_name}"
        )
        current_hold_.append(last_buy_)
    return current_hold_


def describe_data_freshness(all_results):
    """汇总各票最新K线日期，生成邮件标注。

    Returns:
        signal_day, freshness_lines, using_today
    """
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


def format_double_bottom_mail_section(all_results, signal_day):
    """邮件专段：未达严格双底的雏形观察 + 已确认但滞后/未买入提示。"""
    watch_lines = []
    lag_lines = []
    confirmed_idle = []

    for r in all_results:
        note = r.get("double_bottom_note") or {}
        if not note:
            continue
        code = r.get("stock_code", "")
        name = r.get("stock_name", "")
        label = f"{code} {name}".strip()

        trades = r.get("trades") or []
        bought_today = False
        bought_with_db = False
        for t in trades:
            if t.get("type") != "buy":
                continue
            td = t.get("date")
            if td is None:
                continue
            d = td.date() if hasattr(td, "date") else td
            if d != signal_day:
                continue
            bought_today = True
            if "双底" in str(t.get("reason", "")):
                bought_with_db = True

        b1, b2 = note.get("b1_date"), note.get("b2_date")
        bottoms = ""
        if b1 and b2:
            bottoms = f"两底 {b1}/{b2}"
            if note.get("b1_close") is not None and note.get("b2_close") is not None:
                bottoms += f"({note['b1_close']:.2f}/{note['b2_close']:.2f})"

        if note.get("ok_loose") and not note.get("ok_strict"):
            fails = "、".join(note.get("fails") or []) or "严格条件未过"
            watch_lines.append(
                f" 【观察·未达标】{label}: {bottoms}；原因: {fails}"
                f"（不计入买入加分，仅形态提示）"
            )

        first = note.get("first_strict_date")
        pct_first = note.get("pct_from_first_strict")
        pct_b2 = note.get("pct_from_b2")
        pct_s = ""
        if pct_first is not None:
            pct_s = f"，自首次确认约{pct_first*100:+.1f}%"
        elif pct_b2 is not None:
            pct_s = f"，自二底约{pct_b2*100:+.1f}%"

        if note.get("ok_strict") and first is not None and first < signal_day:
            if bought_with_db or bought_today:
                lag_lines.append(
                    f" 【滞后提醒】{label}: 双底早在 {first} 已确认，"
                    f"信号日 {signal_day} 才买入{pct_s}"
                )
            else:
                confirmed_idle.append(
                    f" 【已确认未买】{label}: 双底自 {first} 已成立{pct_s}"
                    f"（其它买入条件未齐，仅提示）"
                )
        elif note.get("ok_strict") and first == signal_day and not bought_today:
            confirmed_idle.append(
                f" 【今日刚确认未买】{label}: {bottoms}；颈线={note.get('neckline')}"
                f"（其它买入条件未齐，仅提示）"
            )

    lines = [
        "",
        "--------------------------------------",
        " 双底形态提示（含未达买入标准的观察）",
        "--------------------------------------",
    ]
    if not (watch_lines or lag_lines or confirmed_idle):
        lines.append("(今日无双底观察/滞后提示)")
        return lines
    lines.extend(watch_lines)
    lines.extend(lag_lines)
    lines.extend(confirmed_idle)
    return lines


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