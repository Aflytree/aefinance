import pdb

import efinance as ef
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import pandas as pd
import  stock_history



def disable_system_proxy():
    """禁用系统代理设置"""
    # 方法1：清空代理环境变量
    proxy_env_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']
    for var in proxy_env_vars:
        os.environ[var] = ''

    # 方法2：设置NO_PROXY
    os.environ['NO_PROXY'] = '*'
    os.environ['no_proxy'] = '*'

    # 方法3：禁用urllib3的警告
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    print("系统代理已禁用")

def setup_efinance_session():
    """设置efinance的session，禁用代理并添加重试机制"""
    # 创建自定义session
    session = requests.Session()

    # 禁用代理
    session.trust_env = False

    # 设置重试策略
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # 替换efinance的默认session
    ef.session = session

class StockAnalyzer:
    def __init__(self, stock_code='000875', beg='20230323', end=datetime.now().date().strftime('%Y%m%d')):
        self.stock_code = stock_code
        self.beg = beg
        self.end = end
        self.df = self._get_data()
        self.weekly_df = self._get_and_prepare_weekly_data(stock_code, beg, end)
        self._calculate_indicators()

    def _get_and_prepare_weekly_data(self, stock_code, beg, end):
        """获取并准备周线数据（获取更长的历史数据）"""
        # 为了正确计算MA20，需要至少提前40周（10个月）的数据
        # 将起始日期提前一年
        import datetime
        from dateutil.relativedelta import relativedelta

        # 将beg转换为datetime
        beg_date = datetime.datetime.strptime(beg, '%Y%m%d')
        # 提前10个月获取数据
        extended_beg = (beg_date - relativedelta(months=10)).strftime('%Y%m%d')

        print(f"📅 获取周线数据：")
        print(f"  请求日期: {beg} 到 {end}")
        print(f"  实际获取: {extended_beg} 到 {end}（提前10个月）")

        weekly_df = stock_history.get_quote_history_baostock(
            stock_code, extended_beg, end, day_or_week="w"
        )

        if weekly_df is None or weekly_df.empty:
            print("⚠️  未能获取周线数据")
            return None

        # 确保日期列存在并转换为datetime
        if '日期' in weekly_df.columns:
            weekly_df['日期'] = pd.to_datetime(weekly_df['日期'])
            weekly_df.set_index('日期', inplace=True)

        print(f"✅ 周线数据获取完成: {len(weekly_df)} 周")
        print(f"   实际日期范围: {weekly_df.index[0]} 到 {weekly_df.index[-1]}")

        # 验证数据量
        if len(weekly_df) < 20:
            print(f"⚠️  警告：只有{len(weekly_df)}周数据，至少需要20周计算MA20")

        return weekly_df

        return weekly_df
    def _get_data(self):
        """获取股票数据"""
        print("开始下载股票行情数据：", self.stock_code)
        # df = ef.stock.get_quote_history(self.stock_code, beg='20240123', end = '20250317')
        # import pdb;pdb.set_trace()
        """快速禁用代理并获取数据"""
        # 一键禁用所有代理
        for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            os.environ[var] = ''
        os.environ['NO_PROXY'] = '*'

        # 设置efinance不使用代理
        # disable_system_proxy()
        # disable_system_proxy()
        # # df = ef.stock.get_quote_history(self.stock_code, beg=self.beg, end=self.end )
        # setup_efinance_session()

        ############################################################################
        # 这里可以选择是使用akshare, efinance还是 baostock
        ############################################################################
        df = stock_history.get_stock_data_with_retry(self.stock_code, self.beg, self.end)
        # df = get_stock_data_akshare(self.stock_code, self.beg, self.end)
        # df = stock_history.get_quote_history_baostock(self.stock_code, self.beg, self.end)
        print("股票行情数据下载完毕")
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        #############################################
        ##################afly#######################
        #############################################
        print("显示股票行情数据")
        print(df)
        # import pdb;pdb.set_trace()
        # return df.tail(self.days)
        return df

    def _calculate_indicators(self):
        """计算技术指标"""
        # 移动平均线
        self.df['MA5'] = self.df['收盘'].rolling(window=5).mean()
        self.df['MA10'] = self.df['收盘'].rolling(window=10).mean()
        self.df['MA20'] = self.df['收盘'].rolling(window=20).mean()

        # RSI指标
        delta = self.df['收盘'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        self.df['RSI'] = 100 - (100 / (1 + gain / loss))

        # MACD指标
        exp1 = self.df['收盘'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['收盘'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2   #DIF
        self.df['Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean() #DEA
        self.df['MACD_Hist'] = self.df['MACD'] - self.df['Signal']

        # 布林带
        self.df['BB_middle'] = self.df['收盘'].rolling(window=20).mean()
        std = self.df['收盘'].rolling(window=20).std()
        self.df['BB_upper'] = self.df['BB_middle'] + (std * 2)
        self.df['BB_lower'] = self.df['BB_middle'] - (std * 2)

    def _identify_support_resistance(self):
        """识别支撑位和阻力位"""
        df = self.df
        window = 20  # 识别窗口
        threshold = 0.02  # 价格聚集阈值

        support_resistance = {
            'support': [],
            'resistance': []
        }

        # 获取最近的价格数据
        recent_prices = df['收盘'].tail(window)
        current_price = recent_prices.iloc[-1]
        # import pdb;pdb.set_trace()

        # 计算价格区间
        price_range = np.arange(
            min(recent_prices) * 0.95,
            max(recent_prices) * 1.05,
            (max(recent_prices) - min(recent_prices)) / 20
        )

        # 统计价格分布
        price_distribution = []
        for price in price_range:
            count = sum((recent_prices >= price * (1 - threshold)) &
                        (recent_prices <= price * (1 + threshold)))
            price_distribution.append((price, count))

        # 识别支撑位和阻力位
        for price, count in price_distribution:
            if count >= 3:  # 至少3天价格聚集
                if price < current_price:
                    support_resistance['support'].append(round(price, 2))
                else:
                    support_resistance['resistance'].append(round(price, 2))

        # 只保留最近的几个位置
        support_resistance['support'] = sorted(support_resistance['support'])[-3:]
        support_resistance['resistance'] = sorted(support_resistance['resistance'])[:3]

        return support_resistance

    def _analyze_macd(self):
        """分析MACD指标"""
        signal = {
            'signal': None,
            'strength': 0,
            'message': ''
        }

        latest_macd = self.df['MACD'].iloc[-1]
        latest_signal = self.df['Signal'].iloc[-1]
        prev_macd = self.df['MACD'].iloc[-2]
        prev_signal = self.df['Signal'].iloc[-2]

        # MACD金叉
        if latest_macd > latest_signal and prev_macd <= prev_signal:
            signal['signal'] = 'buy'
            signal['strength'] = 1
            signal['message'] = 'MACD金叉'
            if prev_macd > 0 and prev_signal > 0 and latest_macd > 0  and latest_signal > 0:
                # print("macd jin")
                signal['message_macd'] = 'MACD位于零轴上方'
                signal['strength'] = 3
        # MACD死叉
        elif latest_macd < latest_signal and prev_macd >= prev_signal:
            signal['signal'] = 'sell'
            signal['strength'] = 1
            signal['message'] = 'MACD死叉'
        # MACD位置
        elif latest_macd > 0:
            signal['signal'] = 'buy'
            signal['strength'] = 0.5
            signal['message'] = 'MACD位于零轴上方'
        else:
            signal['signal'] = 'sell'
            signal['strength'] = 0.5
            signal['message'] = 'MACD位于零轴下方'

        return signal

    def _analyze_kdj(self):
        """分析KDJ指标"""
        signal = {
            'signal': None,
            'strength': 0,
            'message': ''
        }

        # 计算KDJ
        low_min = self.df['最低'].rolling(window=9).min()
        high_max = self.df['最高'].rolling(window=9).max()
        rsv = (self.df['收盘'] - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        j = 3 * k - 2 * d

        latest_k = k.iloc[-1]
        latest_d = d.iloc[-1]
        latest_j = j.iloc[-1]

        # 超买超卖判断
        if latest_k < 20 and latest_d < 20:
            signal['signal'] = 'buy'
            signal['strength'] = 2
            signal['message'] = 'KDJ超卖'
        elif latest_k > 80 and latest_d > 80:
            signal['signal'] = 'sell'
            signal['strength'] = 2
            signal['message'] = 'KDJ超买'
        # 金叉死叉判断
        elif latest_k > latest_d and k.iloc[-2] <= d.iloc[-2]:
            signal['signal'] = 'buy'
            signal['strength'] = 1
            signal['message'] = 'KDJ金叉'
        elif latest_k < latest_d and k.iloc[-2] >= d.iloc[-2]:
            signal['signal'] = 'sell'
            signal['strength'] = 1
            signal['message'] = 'KDJ死叉'
        else:
            signal['signal'] = 'neutral'
            signal['strength'] = 0
            signal['message'] = 'KDJ处于中性位置'

        return signal

    def _analyze_rsi(self):
        """分析RSI指标"""
        signal = {
            'signal': None,
            'strength': 0,
            'message': ''
        }

        latest_rsi = self.df['RSI'].iloc[-1]

        if latest_rsi < 30:
            signal['signal'] = 'buy'
            signal['strength'] = 2
            signal['message'] = 'RSI超卖'
        elif latest_rsi > 70:
            signal['signal'] = 'sell'
            signal['strength'] = 2
            signal['message'] = 'RSI超买'
        elif latest_rsi < 50:
            signal['signal'] = 'buy'
            signal['strength'] = 0.5
            signal['message'] = 'RSI位于50以下'
        else:
            signal['signal'] = 'sell'
            signal['strength'] = 0.5
            signal['message'] = 'RSI位于50以上'

        return signal

    def _analyze_bollinger(self):
        """分析布林带指标"""
        signal = {
            'signal': None,
            'strength': 0,
            'message': ''
        }

        latest_price = self.df['收盘'].iloc[-1]
        latest_upper = self.df['BB_upper'].iloc[-1]
        latest_lower = self.df['BB_lower'].iloc[-1]
        latest_middle = self.df['BB_middle'].iloc[-1]

        # 价格突破布林带
        if latest_price > latest_upper:
            signal['signal'] = 'sell'
            signal['strength'] = 2
            signal['message'] = '价格突破布林带上轨'
        elif latest_price < latest_lower:
            signal['signal'] = 'buy'
            signal['strength'] = 2
            signal['message'] = '价格突破布林带下轨'
        # 价格在布林带中的位置
        elif latest_price > latest_middle:
            signal['signal'] = 'sell'
            signal['strength'] = 0.5
            signal['message'] = '价格位于布林带上方'
        else:
            signal['signal'] = 'buy'
            signal['strength'] = 0.5
            signal['message'] = '价格位于布林带下方'

        return signal

    def _is_double_bottom(self, prices):
        """识别双底形态"""
        if len(prices) < 20:
            return False

        # 寻找局部最低点
        bottoms = []
        for i in range(1, len(prices) - 1):
            if prices.iloc[i] < prices.iloc[i - 1] and prices.iloc[i] < prices.iloc[i + 1]:
                bottoms.append((i, prices.iloc[i]))

        if len(bottoms) < 2:
            return False
        # import pdb;pdb.set_trace()
        # 检查最后两个底部
        last_two_bottoms = bottoms[-2:]
        if len(last_two_bottoms) == 2:
            first_bottom, second_bottom = last_two_bottoms
            # 检查两个底部的价格接近程度
            price_diff = abs(first_bottom[1] - second_bottom[1]) / first_bottom[1]
            # 检查两个底部的时间间隔
            time_diff = second_bottom[0] - first_bottom[0]

            if price_diff < 0.05 and 5 <= time_diff <= 15:
                return True

        return False

    def _is_double_top(self, prices):
        """识别双头形态"""
        if len(prices) < 20:
            return False

        # 寻找局部最高点
        tops = []
        for i in range(1, len(prices) - 1):
            if prices.iloc[i] > prices.iloc[i - 1] and prices.iloc[i] > prices.iloc[i + 1]:
                tops.append((i, prices.iloc[i]))

        if len(tops) < 2:
            return False

        # 检查最后两个顶部
        last_two_tops = tops[-2:]
        if len(last_two_tops) == 2:
            first_top, second_top = last_two_tops
            # 检查两个顶部的价格接近程度
            price_diff = abs(first_top[1] - second_top[1]) / first_top[1]
            # 检查两个顶部的时间间隔
            time_diff = second_top[0] - first_top[0]

            if price_diff < 0.05 and 5 <= time_diff <= 15:
                return True

        return False

    def analyze_trading_signals(self):
        """分析交易信号"""
        signals = pd.DataFrame(index=self.df.index)
        signals['买入信号'] = 0
        signals['卖出信号'] = 0

        # 1. MA金叉死叉信号
        signals.loc[(self.df['MA5'] > self.df['MA20']) &
                    (self.df['MA5'].shift(1) <= self.df['MA20'].shift(1)), '买入信号'] += 1
        signals.loc[(self.df['MA5'] < self.df['MA20']) &
                    (self.df['MA5'].shift(1) >= self.df['MA20'].shift(1)), '卖出信号'] += 1

        # 2. RSI超买超卖信号
        signals.loc[self.df['RSI'] < 30, '买入信号'] += 1
        signals.loc[self.df['RSI'] > 70, '卖出信号'] += 1

        # 3. MACD金叉死叉信号
        signals.loc[(self.df['MACD'] > self.df['Signal']) &
                    (self.df['MACD'].shift(1) <= self.df['Signal'].shift(1)), '买入信号'] += 1
        signals.loc[(self.df['MACD'] < self.df['Signal']) &
                    (self.df['MACD'].shift(1) >= self.df['Signal'].shift(1)), '卖出信号'] += 1

        # 4. 布林带信号
        signals.loc[self.df['收盘'] < self.df['BB_lower'], '买入信号'] += 1
        signals.loc[self.df['收盘'] > self.df['BB_upper'], '卖出信号'] += 1

        return signals

    def get_weekly_data_from_daily(self):
        """从日线数据转换为周线数据"""
        # 确保日期索引是DatetimeIndex
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)

        # 按周重新采样（周一为一周的开始）
        weekly_df = self.df.resample('W-MON').agg({
            '开盘': 'first',  # 周一的开盘价
            '最高': 'max',  # 一周的最高价
            '最低': 'min',  # 一周的最低价
            '收盘': 'last',  # 周五的收盘价
            '成交量': 'sum'  # 一周的总成交量
        }).dropna()  # 删除空值

        return weekly_df

    def analyze_weekly_moving_averages(self, current_date=None):
        """分析周线均线系统（修复版本）"""
        if self.weekly_df is None or len(self.weekly_df) < 5:
            return {"error": "周线数据不足"}

        # 确保weekly_df有正确的日期索引
        if not isinstance(self.weekly_df.index, pd.DatetimeIndex):
            print("⚠️  周线数据索引不是日期类型，尝试修复...")
            # 尝试找到日期列
            for col in self.weekly_df.columns:
                if '日期' in str(col).lower() or 'date' in str(col).lower():
                    self.weekly_df.index = pd.to_datetime(self.weekly_df[col])
                    break

        # 计算周线均线
        weekly_df = self.weekly_df.copy()
        weekly_df['MA5'] = weekly_df['收盘'].rolling(window=5, min_periods=1).mean()
        weekly_df['MA10'] = weekly_df['收盘'].rolling(window=10, min_periods=1).mean()
        weekly_df['MA20'] = weekly_df['收盘'].rolling(window=20, min_periods=1).mean()

        # 如果传入日期，找到对应的周线数据
        if current_date is not None:
            # 转换为pandas Timestamp
            if not isinstance(current_date, pd.Timestamp):
                current_date = pd.to_datetime(current_date)

            # 找到包含current_date的周线（周线数据索引是周五日期）
            # 找到离current_date最近的周五
            week_end_date = self._find_week_end_date(current_date)
            #
            # print(f"📅 查找周线数据:")
            # print(f"  当前日期: {current_date.strftime('%Y-%m-%d')}")
            # print(f"  对应周五: {week_end_date.strftime('%Y-%m-%d')}")

            # 在周线数据中查找这个周五
            matching_dates = weekly_df.index[weekly_df.index == week_end_date]

            if len(matching_dates) > 0:
                target_date = matching_dates[0]
                # 找到这个日期在DataFrame中的位置
                mask = weekly_df.index == target_date
                latest_idx = weekly_df[mask].index[0]

                # 获取这一行的索引位置
                idx_position = weekly_df.index.get_loc(latest_idx)
                prev_idx = max(idx_position - 1, 0)

                latest = weekly_df.iloc[idx_position]
                prev = weekly_df.iloc[prev_idx]

                # print(f"✅ 找到对应周线数据: {latest_idx.strftime('%Y-%m-%d')}")
            else:
                # 如果找不到精确匹配，找最近的一周
                # print("⚠️  未找到精确匹配，使用最接近的周线数据")
                # 计算每个周线日期与目标日期的差值
                date_diffs = abs((weekly_df.index - week_end_date).days)
                closest_idx = date_diffs.idxmin()
                idx_position = weekly_df.index.get_loc(closest_idx)

                latest = weekly_df.iloc[idx_position]
                prev = weekly_df.iloc[max(idx_position - 1, 0)]

                # print(f"📊 使用最近周线: {closest_idx.strftime('%Y-%m-%d')}")
        else:
            # 使用最新数据
            latest = weekly_df.iloc[-1]
            prev = weekly_df.iloc[-2] if len(weekly_df) > 1 else latest
            # print(f"📊 使用最新周线数据: {weekly_df.index[-1].strftime('%Y-%m-%d')}")

        # import pdb;pdb.set_trace()
        # 分析信号
        return self._analyze_weekly_ma_signals(latest, prev)

    def _find_week_end_date(self, current_date):
        """找到包含当前日期的周五日期"""
        # 计算当前是星期几 (0=周一, 1=周二, ..., 4=周五)
        weekday = current_date.weekday()

        if weekday == 4:  # 已经是周五
            return current_date
        # elif weekday < 4:  # 周一至周四
        #     # 本周的周五还没到，返回上周五
        #     days_before = weekday + 3  # 周一需要退3天到上周五，周二退4天...
        #     return current_date - pd.Timedelta(days=days_before)
        elif weekday < 4:  # 周一至周四
            # 本周的周五还没到，返回本周五
            days_after = 4 - weekday  # 周一需要加4天到周五，周二加3天...
            return current_date + pd.Timedelta(days=days_after)
        else:  # 周六或周日
            # 周末，返回本周五（已经过去）
            days_before = weekday - 4  # 周六退1天，周日退2天
            return current_date - pd.Timedelta(days=days_before)

    def _analyze_weekly_ma_signals(self, latest, prev):
        """分析周线MA信号"""
        analysis = {
            'current_values': {
                '收盘价': latest['收盘'],
                'MA5': latest['MA5'],
                'MA10': latest['MA10'],
                'MA20': latest['MA20']
            },
            'signals': [],
            'strength': 0
        }
        # print()
        # 改进的信号判断逻辑
        current_diff = latest['MA5'] - latest['MA20']
        prev_diff = prev['MA5'] - prev['MA20']
        # print(f"MA5={latest['MA5']:.2f}, MA20={latest['MA20']:.2f}, diff={current_diff:.2f}")
        # 判断突破信号（使用相对阈值）
        threshold = max(abs(current_diff) * 0.3, 0.01)  # 30%的差值或最小0.01
        # import pdb;pdb.set_trace()
        # MA5突破MA20判断
        if current_diff > 0:
            if prev_diff <= 0 or (prev_diff > 0 and current_diff > prev_diff * 1.2):
                # 从下方突破或显著加强
                analysis['signals'].append('🟢 MA5上穿/强势突破MA20')
                analysis['strength'] += 3
                # print("🟢 MA5上穿/强势突破MA20")
            else:
                analysis['signals'].append('🟢 MA5在MA20上方延续')
                analysis['strength'] += 1
        elif current_diff < 0:
            if prev_diff >= 0 or (prev_diff < 0 and current_diff < prev_diff * 1.2):
                # 从上方跌破或显著恶化
                analysis['signals'].append('🔴 MA5下穿/弱势跌破MA20')
                analysis['strength'] -= 3

                # print("🔴 MA5下穿/弱势跌破MA20")

            else:
                analysis['signals'].append('🔴 MA5在MA20下方延续')
                analysis['strength'] -= 1
        else:
            analysis['signals'].append('🟡 MA5与MA20接近')

        # 均线排列判断
        ma5_above_ma20 = current_diff > 0
        ma5_above_ma20_prev = prev_diff > 0

        # 如果状态改变
        if ma5_above_ma20 != ma5_above_ma20_prev:
            if ma5_above_ma20:
                analysis['signals'].append('🚀 MA5刚刚突破MA20，趋势转多')
                analysis['strength'] += 2
            else:
                analysis['signals'].append('💀 MA5刚刚跌破MA20，趋势转空')
                analysis['strength'] -= 2

        return analysis

    def _analyze_support_resistance_breakthrough(self):
        """分析支撑位和阻力位的突破情况"""
        analysis = {
            'support_break': {'status': None, 'level': None, 'strength': 0},
            'resistance_break': {'status': None, 'level': None, 'strength': 0},
            'key_levels': {'support': [], 'resistance': []}
        }

        # 获取支撑阻力位
        sr = self._identify_support_resistance()
        analysis['key_levels']['support'] = sr['support']
        analysis['key_levels']['resistance'] = sr['resistance']

        # 当前价格和近期价格
        current_price = self.df['收盘'].iloc[-1]
        prev_price = self.df['收盘'].iloc[-2]
        recent_low = self.df['最低'].tail(5).min()
        recent_high = self.df['最高'].tail(5).max()
        # import pdb;pdb.set_trace()
        # 分析支撑位突破
        if sr['support']:
            for support_level in sr['support']:
                # 检查是否跌破支撑位（收盘价低于支撑位）
                if current_price < support_level and prev_price >= support_level:
                    analysis['support_break']['status'] = '跌破'
                    analysis['support_break']['level'] = support_level
                    analysis['support_break']['strength'] = 2  # 负向信号
                    break
                # 检查是否回踩支撑位
                elif current_price > support_level and recent_low <= support_level * 1.01:
                    analysis['support_break']['status'] = '回踩'
                    analysis['support_break']['level'] = support_level
                    analysis['support_break']['strength'] = -1  # 正向信号
                    break
                # 检查是否在支撑位附近获得支撑
                elif current_price > support_level and current_price <= support_level * 1.02:
                    analysis['support_break']['status'] = '获得支撑'
                    analysis['support_break']['level'] = support_level
                    analysis['support_break']['strength'] = -2  # 正向信号
                    break

        # 分析阻力位突破
        if sr['resistance']:
            for resistance_level in sr['resistance']:
                # 检查是否突破阻力位（收盘价高于阻力位）
                if current_price > resistance_level and prev_price <= resistance_level:
                    analysis['resistance_break']['status'] = '突破'
                    analysis['resistance_break']['level'] = resistance_level
                    analysis['resistance_break']['strength'] = -2  # 负向信号（对空头）
                    break
                # 检查是否测试阻力位
                elif current_price < resistance_level and recent_high >= resistance_level * 0.99:
                    analysis['resistance_break']['status'] = '测试阻力'
                    analysis['resistance_break']['level'] = resistance_level
                    analysis['resistance_break']['strength'] = 1  # 负向信号
                    break
                # 检查是否在阻力位附近受阻
                elif current_price < resistance_level and current_price >= resistance_level * 0.98:
                    analysis['resistance_break']['status'] = '受阻回落'
                    analysis['resistance_break']['level'] = resistance_level
                    analysis['resistance_break']['strength'] = 2  # 负向信号
                    break

        return analysis

    def get_trading_advice1(self):
        """生成更复杂的交易建议"""
        signals = self.analyze_trading_signals()
        latest_date = self.df.index[-1]
        # import pdb;pdb.set_trace()

        # 1. 价格趋势分析
        price_trend = self._analyze_price_trend()

        # 2. 成交量分析
        volume_analysis = self._analyze_volume()
        # import pdb;pdb.set_trace()

        # 3. 技术指标综合分析
        technical_analysis = self._analyze_technical_indicators()

        # 4. 形态识别
        pattern_analysis = self._analyze_patterns()

        # # 5. 支撑阻力位突破分析
        # breakthrough_analysis = self._analyze_support_resistance_breakthrough()
        # if breakthrough_analysis['resistance_break']['status'] == '突破':
        #     print(f"- 突破重要阻力位")
        # if breakthrough_analysis['support_break']['status'] == '获得支撑':
        #     print(f"- 在支撑位获得支撑")
        # if breakthrough_analysis['support_break']['status'] == '跌破':
        #     print("- 跌破重要支撑位" )
        # analysis1 = self.analyze_weekly_moving_averages()
        # 5. 生成综合建议
        return self._generate_comprehensive_advice(
            price_trend, volume_analysis, technical_analysis, pattern_analysis
        )

    def _analyze_price_trend(self):
        """分析价格趋势"""
        df = self.df
        current_price = df['收盘'].iloc[-1]

        analysis = {
            'trend': None,
            'strength': 0,
            'support_resistance': [],
            'details': []
        }

        # 计算各周期涨跌幅
        changes = {
            '日涨跅': (current_price - df['收盘'].iloc[-2]) / df['收盘'].iloc[-2],
            '周涨跌': (current_price - df['收盘'].iloc[-5]) / df['收盘'].iloc[-5] if len(df) >= 5 else 0,
            '月涨跌': (current_price - df['收盘'].iloc[-20]) / df['收盘'].iloc[-20] if len(df) >= 20 else 0
        }

        # 计算趋势强度
        trend_strength = 0
        for period, change in changes.items():
            if change > 0:
                trend_strength += 1
            elif change < 0:
                trend_strength -= 1

        # 识别支撑位和阻力位
        support_resistance = self._identify_support_resistance()
        # import pdb;pdb.set_trace()
        # 判断趋势
        if trend_strength >= 2:
            analysis['trend'] = '上升'
            analysis['strength'] = abs(trend_strength)
        elif trend_strength <= -2:
            analysis['trend'] = '下降'
            analysis['strength'] = trend_strength
            # analysis['strength'] = abs(trend_strength)
            analysis['strength'] = trend_strength
        else:
            analysis['trend'] = '震荡'
            analysis['strength'] = 0

        analysis['support_resistance'] = support_resistance
        analysis['changes'] = changes

        return analysis

    def _analyze_volume(self):
        """分析成交量"""
        df = self.df
        # import pdb;pdb.set_trace()
        current_volume = df['成交量'].iloc[-1]
        # import pdb;pdb.set_trace()
        analysis = {
            'volume_trend': None,
            'volume_signal': None,
            'details': []
        }

        # 计算成交量均线
        vol_ma5 = df['成交量'].rolling(5).mean()
        vol_ma10 = df['成交量'].rolling(10).mean()

        # 计算量比
        volume_ratio = current_volume / vol_ma5.iloc[-2]

        # 判断放量还是缩量
        if volume_ratio > 1.5:
            analysis['volume_trend'] = '放量'
            if df['收盘'].iloc[-1] > df['收盘'].iloc[-2]:
                analysis['volume_signal'] = '放量上涨'
            else:
                analysis['volume_signal'] = '放量下跌'
        elif volume_ratio < 0.7:
            analysis['volume_trend'] = '缩量'
            if df['收盘'].iloc[-1] > df['收盘'].iloc[-2]:
                analysis['volume_signal'] = '缩量上涨'
            else:
                analysis['volume_signal'] = '缩量下跌'
        else:
            analysis['volume_trend'] = '量能平稳'

        return analysis

    def _analyze_technical_indicators(self):
        """分析技术指标"""
        df = self.df
        analysis = {
            'indicators': {},
            'signals': [],
            'strength': 0
        }

        # 1. MACD分析
        macd_signal = self._analyze_macd()
        analysis['indicators']['MACD'] = macd_signal

        # 2. KDJ分析
        kdj_signal = self._analyze_kdj()
        analysis['indicators']['KDJ'] = kdj_signal

        # 3. RSI分析
        rsi_signal = self._analyze_rsi()
        analysis['indicators']['RSI'] = rsi_signal

        # 4. 布林带分析
        bollinger_signal = self._analyze_bollinger()
        analysis['indicators']['BOLL'] = bollinger_signal

        # 计算综合信号强度
        for indicator, signal in analysis['indicators'].items():
            if signal['signal'] == 'buy':
                analysis['strength'] += signal['strength']
            elif signal['signal'] == 'sell':
                analysis['strength'] -= signal['strength']

        return analysis

    def _analyze_patterns(self):
        """识别K线形态"""
        df = self.df
        patterns = {
            'candlestick': [],
            'price_patterns': [],
            'strength': 0
        }

        # 1. 识别单日K线形态
        latest_k = {
            'open': df['开盘'].iloc[-1],
            'high': df['最高'].iloc[-1],
            'low': df['最低'].iloc[-1],
            'close': df['收盘'].iloc[-1]
        }

        # 判断十字星
        if (latest_k['high'] - latest_k['low']) > 0:
            if abs(latest_k['open'] - latest_k['close']) / (latest_k['high'] - latest_k['low']) < 0.1:
                patterns['candlestick'].append('十字星')

        # 判断长上影线
        if (latest_k['high'] - latest_k['low']) > 0:
            if (latest_k['high'] - max(latest_k['open'], latest_k['close'])) / (latest_k['high'] - latest_k['low']) > 0.6:
                patterns['candlestick'].append('长上影线')

        # 判断长下影线
        if (latest_k['high'] - latest_k['low']) > 0:
            if (min(latest_k['open'], latest_k['close']) - latest_k['low']) / (latest_k['high'] - latest_k['low']) > 0.6:
                patterns['candlestick'].append('长下影线')

        # 2. 识别多日形态
        recent_prices = df['收盘'].tail(20)

        # 判断双底形态
        if self._is_double_bottom(recent_prices):
            patterns['price_patterns'].append('双底形态')
            patterns['strength'] += 2

        # 判断双头形态
        if self._is_double_top(recent_prices):
            patterns['price_patterns'].append('双头形态')
            patterns['strength'] -= 2

        return patterns

    def _generate_comprehensive_advice(self, price_trend, volume_analysis, technical_analysis, pattern_analysis):
        """生成综合建议"""
        latest_date = self.df.index[-1]
        latest_price = self.df['收盘'].iloc[-1]

        advice = f"\n=== 交易建议分析 ({latest_date.strftime('%Y-%m-%d')}) ===\n"
        advice += f"当前价格: {latest_price:.2f}\n"

        # 1. 趋势分析总结
        advice += "\n【趋势分析】\n"
        advice += f"主趋势: {price_trend['trend']} (强度: {price_trend['strength']})\n"
        for period, change in price_trend['changes'].items():
            advice += f"{period}: {change * 100:.2f}%\n"

        # 2. 量能分析
        advice += "\n【量能分析】\n"
        advice += f"成交量状态: {volume_analysis['volume_trend']}\n"
        if volume_analysis['volume_signal']:
            advice += f"量能信号: {volume_analysis['volume_signal']}\n"

        # 3. 技术指标分析
        advice += "\n【技术指标】\n"
        for indicator, signal in technical_analysis['indicators'].items():
            advice += f"{indicator}: {signal['message']}\n"

        # 4. 形态分析
        if pattern_analysis['candlestick'] or pattern_analysis['price_patterns']:
            advice += "\n【形态分析】\n"
            if pattern_analysis['candlestick']:
                advice += f"K线形态: {', '.join(pattern_analysis['candlestick'])}\n"
            if pattern_analysis['price_patterns']:
                advice += f"价格形态: {', '.join(pattern_analysis['price_patterns'])}\n"

        # 5. 综合建议
        total_strength = (
                price_trend['strength'] +
                technical_analysis['strength'] +
                pattern_analysis['strength']
        )
        # print("total_strength:", total_strength)
        # import pdb;pdb.set_trace()
        advice += "\n【交易建议】\n"
        if total_strength >= 3:
            advice += "强烈买入信号\n"
            advice += "理由:\n"
            if price_trend['trend'] == '上升':
                advice += "- 价格处于上升趋势\n"
            if volume_analysis['volume_signal'] == '放量上涨':
                advice += "- 量能配合良好\n"
            if technical_analysis['strength'] > 4:
                advice += "- 技术指标显示买入信号\n"
        elif total_strength <= -3:
            advice += "强烈卖出信号\n"
            advice += "理由:\n"
            if price_trend['trend'] == '下降':
                advice += "- 价格处于下降趋势\n"
            if volume_analysis['volume_signal'] == '放量下跌':
                advice += "- 量能配合显示卖压\n"
            if technical_analysis['strength'] < 0:
                advice += "- 技术指标显示卖出信号\n"
        else:
            advice += "观望信号\n"
            advice += "- 当前无明显买卖信号，建议观望\n"

        # 6. 风险提示
        advice += "\n【风险提示】\n"
        advice += "- 建议结合基本面分析\n"
        advice += "- 注意设置止损位置\n"
        advice += "- 控制仓位风险\n"

        return advice

    def get_trading_advice2(self):
        """生成交易建议"""
        signals = self.analyze_trading_signals()
        latest_date = self.df.index[-1]

        # 获取最新的信号强度
        buy_strength = signals['买入信号'].iloc[-1]
        sell_strength = signals['卖出信号'].iloc[-1]

        # 获取最新价格信息
        latest_price = self.df['收盘'].iloc[-1]
        prev_price = self.df['收盘'].iloc[-2]
        price_change = (latest_price - prev_price) / prev_price * 100

        advice = f"\n交易建议分析 ({latest_date.strftime('%Y-%m-%d')}):\n"
        advice += f"当前价格: {latest_price:.2f} (日涨跌: {price_change:.2f}%)\n"

        # 技术指标状态
        advice += "\n技术指标状态:\n"
        advice += f"RSI: {self.df['RSI'].iloc[-1]:.2f}\n"
        advice += f"MACD: {self.df['MACD'].iloc[-1]:.2f}\n"

        # 综合建议
        advice += "\n交易建议:\n"
        if buy_strength > sell_strength:
            strength = "强" if buy_strength >= 2 else "中等"
            advice += f"买入信号 ({strength})\n"
            advice += "理由:\n"
            if self.df['RSI'].iloc[-1] < 30:
                advice += "- RSI处于超卖区域\n"
            if self.df['收盘'].iloc[-1] < self.df['BB_lower'].iloc[-1]:
                advice += "- 价格触及布林带下轨\n"
            if self.df['MA5'].iloc[-1] > self.df['MA20'].iloc[-1]:
                # print(self.df['MA5'].iloc[-2])
                # print(self.df['MA20'].iloc[-2])
                advice += "- 短期均线上穿长期均线\n"
        elif sell_strength > buy_strength:
            strength = "强" if sell_strength >= 2 else "中等"
            advice += f"卖出信号 ({strength})\n"
            advice += "理由:\n"
            if self.df['RSI'].iloc[-1] > 70:
                advice += "- RSI处于超买区域\n"
            if self.df['收盘'].iloc[-1] > self.df['BB_upper'].iloc[-1]:
                advice += "- 价格触及布林带上轨\n"
            if self.df['MA5'].iloc[-1] < self.df['MA20'].iloc[-1]:
                advice += "- 短期均线下穿长期均线\n"
        else:
            advice += "观望信号\n"
            advice += "- 当前无明显买卖信号，建议观望\n"

        return advice

