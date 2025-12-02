import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import stock_history

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def parse_date_input(date_str):
    """
    解析日期输入，支持多种格式
    """
    if not date_str:
        return None

    # 尝试多种日期格式
    date_formats = [
        '%Y-%m-%d',  # 2023-01-01
        '%Y%m%d',  # 20230101
        '%Y/%m/%d',  # 2023/01/01
        '%Y.%m.%d',  # 2023.01.01
    ]

    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # 如果是相对日期，如 "30d", "6m", "1y"
    if date_str.endswith('d'):  # 天数
        try:
            days = int(date_str[:-1])
            return datetime.now() - timedelta(days=days)
        except ValueError:
            pass
    elif date_str.endswith('m'):  # 月数
        try:
            months = int(date_str[:-1])
            return datetime.now() - timedelta(days=months * 30)
        except ValueError:
            pass
    elif date_str.endswith('y'):  # 年数
        try:
            years = int(date_str[:-1])
            return datetime.now() - timedelta(days=years * 365)
        except ValueError:
            pass

    print(f"无法解析日期: {date_str}，请使用格式: YYYY-MM-DD, YYYYMMDD, 30d, 6m, 1y 等")
    return None


def get_stock_data_flexible(stock_code, start_date='20230323', end_date=datetime.now().date().strftime('%Y%m%d')):
    """
    灵活获取股票历史数据
    """
    try:
        # 使用stock_history获取数据
        df = stock_history.get_stock_data_with_retry(stock_code, start_date, end_date)

        if df is None or df.empty:
            print("获取数据失败，请检查股票代码")
            return None

        # 确保日期列是datetime类型
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)

        print(f"成功获取 {len(df)} 个交易日数据")
        return df

    except Exception as e:
        print(f"获取数据时出错: {e}")
        return None


def calculate_price_changes(stock_data):
    """
    计算每日涨跌幅
    """
    stock_data['Daily_Return'] = stock_data['收盘'].pct_change() * 100
    stock_data = stock_data.dropna()
    return stock_data


def categorize_price_changes(daily_return):
    """
    将涨跌幅分类到简化的区间
    """
    if daily_return > 3:
        return '涨3%以上'
    elif daily_return > 0:
        return '涨0-3%'
    elif daily_return == 0:
        return '平盘'
    elif daily_return > -3:
        return '跌0-3%'
    else:
        return '跌3%以上'


def analyze_stock_price_distribution_flexible(stock_code, start_date=None):
    """
    灵活分析股票涨跌分布
    """
    # 获取股票数据
    stock_data = get_stock_data_flexible(stock_code, start_date)

    if stock_data is None:
        return None, None, None

    # 计算涨跌幅
    stock_data = calculate_price_changes(stock_data)

    # 分类涨跌幅
    stock_data['Change_Category'] = stock_data['Daily_Return'].apply(categorize_price_changes)

    # 统计分布
    distribution = stock_data['Change_Category'].value_counts().sort_index()

    # 计算百分比
    percentage_distribution = (distribution / len(stock_data) * 100).round(2)

    return distribution, percentage_distribution, stock_data


def plot_distribution(distribution, stock_code, date_info):
    """
    绘制涨跌分布图
    """
    plt.figure(figsize=(12, 8))

    # 设置颜色
    colors = []
    for category in distribution.index:
        if '涨' in category:
            colors.append('#ff4d4f')  # 红色系
        elif '跌' in category:
            colors.append('#52c41a')  # 绿色系
        else:
            colors.append('#bfbfbf')  # 灰色

    # 绘制柱状图
    bars = plt.bar(distribution.index, distribution.values, color=colors, alpha=0.8)

    # 添加数值标签
    for bar, value in zip(bars, distribution.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{value}天', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title(f'{stock_code} {date_info}涨跌分布统计', fontsize=16, fontweight='bold')
    plt.xlabel('涨跌区间', fontsize=12)
    plt.ylabel('出现天数', fontsize=12)
    plt.xticks(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pie_chart(distribution, stock_code, date_info):
    """
    绘制饼图展示分布
    """
    plt.figure(figsize=(10, 8))

    # 设置颜色
    colors = []
    labels = []
    for category in distribution.index:
        if '涨' in category:
            colors.append('#ff7875')  # 浅红色
        elif '跌' in category:
            colors.append('#73d13d')  # 浅绿色
        else:
            colors.append('#d9d9d9')  # 灰色
        # 在标签中显示百分比
        pct = (distribution[category] / distribution.sum() * 100).round(1)
        labels.append(f'{category}\n({pct}%)')

    # 绘制饼图
    plt.pie(distribution.values, labels=labels, colors=colors,
            startangle=90, textprops={'fontsize': 11})
    plt.title(f'{stock_code} {date_info}涨跌分布饼图', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.show()


def calculate_probability_statistics(stock_data):
    """
    计算各种涨跌情况的概率统计
    """
    total_days = len(stock_data)

    # 计算各区间概率
    prob_up_0_3 = len(
        stock_data[(stock_data['Daily_Return'] > 0) & (stock_data['Daily_Return'] <= 3)]) / total_days * 100
    prob_up_over_3 = len(stock_data[stock_data['Daily_Return'] > 3]) / total_days * 100
    prob_down_0_3 = len(
        stock_data[(stock_data['Daily_Return'] < 0) & (stock_data['Daily_Return'] >= -3)]) / total_days * 100
    prob_down_over_3 = len(stock_data[stock_data['Daily_Return'] < -3]) / total_days * 100
    prob_flat = len(stock_data[stock_data['Daily_Return'] == 0]) / total_days * 100

    # 计算总体涨跌概率
    prob_up_total = len(stock_data[stock_data['Daily_Return'] > 0]) / total_days * 100
    prob_down_total = len(stock_data[stock_data['Daily_Return'] < 0]) / total_days * 100

    return {
        '涨0-3%': prob_up_0_3,
        '涨3%以上': prob_up_over_3,
        '跌0-3%': prob_down_0_3,
        '跌3%以上': prob_down_over_3,
        '平盘': prob_flat,
        '总体上涨': prob_up_total,
        '总体下跌': prob_down_total
    }


def print_probability_statistics(prob_stats, total_days):
    """
    打印概率统计结果
    """
    print(f"\n🎯 涨跌概率统计 (基于{total_days}个交易日):")
    print("=" * 50)

    print(f"\n📈 上涨情况概率:")
    print("-" * 30)
    print(f"涨0-3%概率:    {prob_stats['涨0-3%']:>6.2f}%")
    print(f"涨3%以上概率:   {prob_stats['涨3%以上']:>6.2f}%")
    print(f"总体上涨概率:   {prob_stats['总体上涨']:>6.2f}%")

    print(f"\n📉 下跌情况概率:")
    print("-" * 30)
    print(f"跌0-3%概率:    {prob_stats['跌0-3%']:>6.2f}%")
    print(f"跌3%以上概率:   {prob_stats['跌3%以上']:>6.2f}%")
    print(f"总体下跌概率:   {prob_stats['总体下跌']:>6.2f}%")

    print(f"\n⚖️ 其他统计:")
    print("-" * 30)
    print(f"平盘概率:      {prob_stats['平盘']:>6.2f}%")

    # 计算涨跌比
    up_down_ratio = prob_stats['总体上涨'] / prob_stats['总体下跌'] if prob_stats['总体下跌'] > 0 else float('inf')
    print(f"涨跌比:        {up_down_ratio:>6.2f}")


def detailed_rise_fail_analysis_flexible(stock_code, start_date=None):
    """
    灵活详细分析函数
    """
    if start_date:
        start_dt = parse_date_input(start_date)
        end_dt = datetime.now()
        date_info = f"{start_dt.strftime('%Y-%m-%d')} 至 {end_dt.strftime('%Y-%m-%d')}"
    else:
        date_info = "近期"

    print(f"\n{'=' * 60}")
    print(f"股票 {stock_code} {date_info} 涨跌分布分析")
    print(f"{'=' * 60}")

    # 获取分析结果
    distribution, percentage, stock_data = analyze_stock_price_distribution_flexible(stock_code, start_date)

    if distribution is None:
        return

    # 打印分布统计
    print("\n📊 涨跌分布统计:")
    print("-" * 40)
    for category in ['涨3%以上', '涨0-3%', '平盘', '跌0-3%', '跌3%以上']:
        if category in distribution.index:
            count = distribution[category]
            pct = percentage[category]
            print(f"{category:<8}: {count:>4}天 ({pct:>5}%)")
        else:
            print(f"{category:<8}:    0天 (  0%)")

    # 计算概率统计
    prob_stats = calculate_probability_statistics(stock_data)

    # 打印概率统计
    print_probability_statistics(prob_stats, len(stock_data))

    # 基本统计信息
    total_days = len(stock_data)
    max_gain = stock_data['Daily_Return'].max()
    max_loss = stock_data['Daily_Return'].min()
    avg_gain = stock_data['Daily_Return'].mean()
    std_return = stock_data['Daily_Return'].std()

    print(f"\n📈 基本统计信息:")
    print("-" * 40)
    print(f"总交易日数: {total_days}")
    print(f"最大单日涨幅: {max_gain:.2f}%")
    print(f"最大单日跌幅: {max_loss:.2f}%")
    print(f"平均日涨跌幅: {avg_gain:.2f}%")
    print(f"日涨跌幅标准差: {std_return:.2f}%")

    # 绘制图表
    plot_distribution(distribution, stock_code, date_info)
    plot_pie_chart(distribution, stock_code, date_info)

    return distribution, percentage, stock_data, prob_stats

# 使用示例
if __name__ == "__main__":
    print("🎯 股票涨跌分布分析工具 (简化版)")
    print("区间分类: 涨0-3%, 涨3%以上, 跌0-3%, 跌3%以上")

    # 单个股票分析
    detailed_rise_fail_analysis_flexible('600415', start_date='20240323')
