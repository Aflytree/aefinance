import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import random
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings('ignore')


def create_session_with_retry(retries=3, backoff_factor=0.3):
    """创建带重试机制的session"""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def get_weekly_stock_data(stock_code, period="weekly", start_date=None, max_retries=3):
    """
    获取股票的周线数据 - 增强版，带重试机制
    """
    for retry in range(max_retries):
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

            # 添加随机延迟避免请求过快
            time.sleep(random.uniform(0.5, 1.5))

            # 使用akshare获取数据
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period=period,
                start_date=start_date,
                adjust="qfq",
                timeout=15
            )

            if df.empty:
                return None

            # 处理不同的列数
            if len(df.columns) == 12:
                df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量',
                              '成交额', '振幅', '涨跌幅', '涨跌额', '换手率', '未知']
            elif len(df.columns) == 11:
                df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量',
                              '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
            elif len(df.columns) >= 7:
                # 取前7列
                df = df.iloc[:, :7]
                df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额']
            else:
                return None

            # 确保有必要的列
            required_cols = ['日期', '收盘']
            if not all(col in df.columns for col in required_cols):
                return None

            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期').reset_index(drop=True)

            return df

        except requests.exceptions.RequestException as e:
            if retry < max_retries - 1:
                wait_time = (retry + 1) * 2
                print(f"重试 {stock_code} ({retry + 1}/{max_retries}), 等待{wait_time}秒...")
                time.sleep(wait_time)
                continue
            else:
                print(f"获取{stock_code}数据失败(网络错误): {e}")
                return None
        except Exception as e:
            print(f"获取{stock_code}数据失败: {str(e)[:50]}")
            return None

    return None


def get_stock_list(limit=50):
    """
    获取股票列表 - 限制数量避免过多请求
    """
    try:
        # 尝试从文件读取缓存
        cache_file = 'stock_list_cache.csv'
        try:
            # 检查缓存是否过期（1天）
            if os.path.exists(cache_file):
                mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if (datetime.now() - mtime).days < 1:
                    cached_df = pd.read_csv(cache_file)
                    stock_list = cached_df['code'].tolist()
                    print(f"从缓存读取{len(stock_list)}只股票列表")
                    return stock_list[:limit]
        except:
            pass

        print("获取最新股票列表...")
        time.sleep(2)  # 等待一下

        # 获取沪深300成分股作为样本（更稳定）
        try:
            hs300 = ak.index_stock_cons_csindex(symbol="000300")
            if not hs300.empty:
                stock_list = hs300['成分券代码'].tolist()
                print(f"获取到沪深300成分股 {len(stock_list)} 只")

                # 保存缓存
                pd.DataFrame({'code': stock_list}).to_csv(cache_file, index=False)
                return stock_list[:limit]
        except:
            pass

        # 如果沪深300失败，使用A股列表
        stock_info = ak.stock_info_a_code_name()
        if stock_info.empty:
            # 使用预设的稳定股票
            default_stocks = ['000001', '000002', '000858', '002415', '300750',
                              '600519', '601318', '000333', '000651', '300059',
                              '600036', '601888', '600276', '300760', '601012',
                              '000568', '002475', '300498', '600887', '600900',
                              '600030', '601166', '600048', '601328', '601288',
                              '000063', '002594', '000725', '600104', '000100']
            return default_stocks[:limit]

        # 过滤ST股票
        mask = ~stock_info['name'].str.contains('ST')
        filtered_stocks = stock_info[mask]['code'].tolist()
        filtered_stocks = [code for code in filtered_stocks if code.isdigit()]

        # 保存缓存
        pd.DataFrame({'code': filtered_stocks}).to_csv(cache_file, index=False)

        print(f"获取到 {len(filtered_stocks)} 只股票")
        return filtered_stocks[:limit]

    except Exception as e:
        print(f"获取股票列表失败: {e}")
        # 返回预设股票
        default_stocks = ['000001', '000002', '000858', '002415', '300750',
                          '600519', '601318', '000333', '000651', '300059']
        return default_stocks[:limit]


def calculate_moving_averages(df, periods=[5, 20, 60]):
    """计算移动平均线"""
    df = df.copy()
    for period in periods:
        df[f'MA{period}'] = df['收盘'].rolling(window=period, min_periods=period).mean()
    return df


def detect_ma_crossover(df, short_period=5, long_period=20, lookback_weeks=8):
    """
    检测均线交叉信号
    """
    if df is None or len(df) < max(short_period, long_period) + 5:
        return False, None, "数据不足"

    # 确保有均线数据
    ma_short = f'MA{short_period}'
    ma_long = f'MA{long_period}'

    if ma_short not in df.columns or ma_long not in df.columns:
        return False, None, "无均线数据"

    # 查看最近几周
    recent_data = df.tail(min(lookback_weeks + 5, len(df)))

    # 查找金叉
    for i in range(1, len(recent_data)):
        short_prev = recent_data.iloc[i - 1][ma_short]
        long_prev = recent_data.iloc[i - 1][ma_long]
        short_curr = recent_data.iloc[i][ma_short]
        long_curr = recent_data.iloc[i][ma_long]

        # 检查是否为有效数值
        if pd.isna(short_prev) or pd.isna(long_prev) or pd.isna(short_curr) or pd.isna(long_curr):
            continue

        # 金叉条件：短期均线上穿长期均线
        if short_prev <= long_prev and short_curr > long_curr:
            cross_date = recent_data.iloc[i]['日期']
            # 确认交叉后短期均线保持在长期均线之上
            if i < len(recent_data) - 1:
                for j in range(i + 1, min(i + 3, len(recent_data))):
                    if recent_data.iloc[j][ma_short] <= recent_data.iloc[j][ma_long]:
                        break
            return True, cross_date, "金叉"

    return False, None, "无金叉"


def get_stock_name(stock_code):
    """获取股票名称"""
    try:
        stock_info = ak.stock_info_a_code_name()
        name_row = stock_info[stock_info['code'] == stock_code]
        if not name_row.empty:
            return name_row['name'].iloc[0]
    except:
        pass
    return stock_code


def scan_stocks_with_progress(stock_list, lookback_weeks=12):
    """
    带进度条的股票扫描
    """
    results = []
    total = len(stock_list)

    print(f"\n开始扫描 {total} 只股票...")
    print("=" * 60)

    # 使用tqdm显示进度条
    for i, stock_code in enumerate(tqdm(stock_list, desc="扫描进度")):
        try:
            # 获取数据
            df = get_weekly_stock_data(stock_code)

            if df is not None and len(df) >= 30:
                # 计算均线
                df = calculate_moving_averages(df)

                # 检测金叉
                has_cross, cross_date, status = detect_ma_crossover(df, lookback_weeks=lookback_weeks)

                if has_cross and cross_date:
                    # 获取最新数据
                    last_row = df.iloc[-1]

                    # 计算距离现在的周数
                    weeks_ago = (datetime.now() - cross_date).days // 7

                    # 只保留最近8周内的金叉
                    if weeks_ago <= lookback_weeks:
                        stock_name = get_stock_name(stock_code)

                        results.append({
                            '股票代码': stock_code,
                            '股票名称': stock_name,
                            '金叉日期': cross_date,
                            '当前价格': round(last_row['收盘'], 2),
                            f'MA5': round(last_row.get('MA5', 0), 2),
                            f'MA20': round(last_row.get('MA20', 0), 2),
                            f'MA60': round(last_row.get('MA60', 0), 2),
                            '5周线上穿幅度(%)': round((last_row.get('MA5', 0) / last_row.get('MA20', 1) - 1) * 100, 2),
                            '距离当前(周)': weeks_ago,
                            '数据长度(周)': len(df)
                        })

            # 更长的延迟，避免请求过快
            if i % 10 == 0:
                time.sleep(random.uniform(2, 3))
            else:
                time.sleep(random.uniform(0.8, 1.5))

        except KeyboardInterrupt:
            print("\n用户中断扫描")
            break
        except Exception as e:
            # 静默处理错误
            continue

    return pd.DataFrame(results)


def analyze_and_display_results(results_df):
    """分析和显示结果"""
    if results_df.empty:
        print("\n" + "=" * 60)
        print("未发现近期5周线上穿20周线的股票")
        print("=" * 60)
        return results_df

    print("\n" + "=" * 80)
    print(f"🎯 发现 {len(results_df)} 只近期5周线上穿20周线的股票")
    print("=" * 80)

    # 按金叉日期排序（最近的在前）
    results_df = results_df.sort_values('金叉日期', ascending=False)

    # 显示结果
    print("\n📈 金叉股票列表：")
    print("-" * 120)
    print(f"{'代码':8} {'名称':15} {'金叉日期':12} {'当前价':8} {'MA5':8} {'MA20':8} {'上穿幅度%':10} {'周数':6}")
    print("-" * 120)

    for _, row in results_df.head(30).iterrows():
        print(f"{row['股票代码']:8} {row['股票名称'][:14]:15} "
              f"{row['金叉日期'].strftime('%m-%d'):12} "
              f"{row['当前价格']:8.2f} "
              f"{row['MA5']:8.2f} "
              f"{row['MA20']:8.2f} "
              f"{row['5周线上穿幅度(%)']:10.2f} "
              f"{row['距离当前(周)']:6}")

    # 统计信息
    print("\n" + "=" * 120)
    print("📊 统计信息：")
    print(f"  最近金叉: {results_df['金叉日期'].max().strftime('%Y-%m-%d')}")
    print(f"  最早金叉: {results_df['金叉日期'].min().strftime('%Y-%m-%d')}")
    print(f"  平均上穿幅度: {results_df['5周线上穿幅度(%)'].mean():.2f}%")
    print(f"  最大上穿幅度: {results_df['5周线上穿幅度(%)'].max():.2f}%")

    # 按上穿幅度排序
    print("\n🏆 上穿幅度最大的股票：")
    top_by_gap = results_df.nlargest(5, '5周线上穿幅度(%)')
    for _, row in top_by_gap.iterrows():
        print(f"  {row['股票代码']} {row['股票名称'][:12]:12} - 上穿幅度: {row['5周线上穿幅度(%)']:.2f}%")

    return results_df


def save_results_with_timestamp(results_df):
    """保存结果"""
    if results_df.empty:
        return None

    timestamp = datetime.now().strftime('%Y%m%d')
    filename_csv = f'5周线上穿20周线_{timestamp}.csv'
    filename_xlsx = f'5周线上穿20周线_{timestamp}.xlsx'

    # 保存CSV
    results_df.to_csv(filename_csv, index=False, encoding='utf-8-sig')

    # 保存Excel
    results_df.to_excel(filename_xlsx, index=False)

    print(f"\n💾 结果已保存：")
    print(f"  CSV文件: {filename_csv}")
    print(f"  Excel文件: {filename_xlsx}")

    return filename_csv


def main():
    """主函数"""
    print("=" * 60)
    print("🔍 5周线上穿20周线股票扫描程序")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 配置参数
    SCAN_LIMIT = 80  # 扫描股票数量（从稳定的股票开始）
    LOOKBACK_WEEKS = 12  # 回顾周数

    try:
        # 1. 获取股票列表
        print(f"\n1. 获取股票列表（限制{SCAN_LIMIT}只）...")
        stock_list = get_stock_list(SCAN_LIMIT)

        # 2. 扫描金叉
        print(f"\n2. 扫描金叉信号（回顾{LOOKBACK_WEEKS}周）...")
        results = scan_stocks_with_progress(stock_list, LOOKBACK_WEEKS)

        # 3. 分析结果
        print(f"\n3. 分析结果...")
        analyzed_results = analyze_and_display_results(results)

        # 4. 保存结果
        if not analyzed_results.empty:
            save_results_with_timestamp(analyzed_results)

            # 建议
            print(f"\n💡 投资建议：")
            print(f"  1. 金叉信号需结合成交量确认")
            print(f"  2. 建议关注上穿幅度适中（3-10%）的股票")
            print(f"  3. 注意60周线（MA60）的趋势")
            print(f"  4. 建议结合基本面分析")

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
    finally:
        print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)


# 如果安装了tqdm，使用它，否则用简单进度显示
try:
    from tqdm import tqdm
except ImportError:
    print("提示: 安装tqdm可以获得更好的进度条显示: pip install tqdm")


    # 简单的tqdm替代
    class tqdm:
        def __init__(self, iterable, desc=""):
            self.iterable = iterable
            self.desc = desc
            self.total = len(iterable)
            self.current = 0

        def __iter__(self):
            print(f"{self.desc}: 0/{self.total}", end="")
            for item in self.iterable:
                yield item
                self.current += 1
                print(f"\r{self.desc}: {self.current}/{self.total}", end="")
            print()

if __name__ == "__main__":
    # 创建必要的目录
    import os

    if not os.path.exists('reports'):
        os.makedirs('reports')

    # 运行主程序
    main()