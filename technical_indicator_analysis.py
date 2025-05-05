import efinance as ef
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class StockAnalyzer:
    def __init__(self, stock_code='000875', beg='20230323', end=datetime.now().date().strftime('%Y%m%d')):
        self.stock_code = stock_code
        self.beg = beg
        self.end = end
        self.df = self._get_data()
        self._calculate_indicators()

    def _get_data(self):
        """获取股票数据"""
        print("开始下载股票行情数据：", self.stock_code)
        # df = ef.stock.get_quote_history(self.stock_code, beg='20240123', end = '20250317')
        df = ef.stock.get_quote_history(self.stock_code, beg=self.beg, end=self.end )
        print("股票行情数据下载完毕")
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
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

    def get_trading_advice1(self):
        """生成更复杂的交易建议"""
        signals = self.analyze_trading_signals()
        latest_date = self.df.index[-1]

        # 1. 价格趋势分析
        price_trend = self._analyze_price_trend()

        # 2. 成交量分析
        volume_analysis = self._analyze_volume()

        # 3. 技术指标综合分析
        technical_analysis = self._analyze_technical_indicators()

        # 4. 形态识别
        pattern_analysis = self._analyze_patterns()

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

        # 判断趋势
        if trend_strength >= 2:
            analysis['trend'] = '上升'
            analysis['strength'] = abs(trend_strength)
        elif trend_strength <= -2:
            analysis['trend'] = '下降'
            analysis['strength'] = abs(trend_strength)
        else:
            analysis['trend'] = '震荡'
            analysis['strength'] = 0

        analysis['support_resistance'] = support_resistance
        analysis['changes'] = changes

        return analysis

    def _analyze_volume(self):
        """分析成交量"""
        df = self.df
        current_volume = df['成交量'].iloc[-1]

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

