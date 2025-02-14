import efinance as ef
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time


class StockRealTimeAnalyzer:
    def __init__(self, stock_code='002506', days=60):
        self.stock_code = stock_code
        self.days = days
        self.df = self._get_historical_data()
        self.realtime_data = None
        self.update_realtime_data()
        self._calculate_indicators()

    def _get_historical_data(self):
        """获取历史数据"""
        df = ef.stock.get_quote_history(self.stock_code)
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        return df.tail(self.days)

    def update_realtime_data(self):
        """获取实时数据"""
        try:
            realtime = ef.stock.get_realtime_quotes(self.stock_code)
            self.realtime_data = realtime.iloc[0]
            return True
        except Exception as e:
            print(f"获取实时数据失败: {str(e)}")
            return False

    def _calculate_indicators(self):
        """计算技术指标"""
        # 基础指标计算（与之前相同）
        self._calculate_basic_indicators()

        # 新增盘中实时指标
        if self.realtime_data is not None:
            self._calculate_realtime_indicators()

    def _calculate_basic_indicators(self):
        """计算基础技术指标"""
        # 移动平均线
        self.df['MA5'] = self.df['收盘'].rolling(window=5).mean()
        self.df['MA10'] = self.df['收盘'].rolling(window=10).mean()
        self.df['MA20'] = self.df['收盘'].rolling(window=20).mean()
        self.df['MA60'] = self.df['收盘'].rolling(window=60).mean()

        # 成交量均线
        self.df['VOL5'] = self.df['成交量'].rolling(window=5).mean()
        self.df['VOL10'] = self.df['成交量'].rolling(window=10).mean()

        # MACD
        exp1 = self.df['收盘'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['收盘'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Hist'] = self.df['MACD'] - self.df['Signal']

        # KDJ
        low_min = self.df['最低'].rolling(window=9).min()
        high_max = self.df['最高'].rolling(window=9).max()
        self.df['RSV'] = (self.df['收盘'] - low_min) / (high_max - low_min) * 100
        self.df['K'] = self.df['RSV'].ewm(com=2).mean()
        self.df['D'] = self.df['K'].ewm(com=2).mean()
        self.df['J'] = 3 * self.df['K'] - 2 * self.df['D']

    def _calculate_realtime_indicators(self):
        """计算实时指标"""
        current_price = float(self.realtime_data['最新价'])

        # 计算实时价格相对均线位置
        self.price_ma_position = {
            'MA5': current_price / self.df['MA5'].iloc[-1] - 1,
            'MA10': current_price / self.df['MA10'].iloc[-1] - 1,
            'MA20': current_price / self.df['MA20'].iloc[-1] - 1,
            'MA60': current_price / self.df['MA60'].iloc[-1] - 1
        }

        # 计算实时振幅
        self.daily_amplitude = (float(self.realtime_data['最高价']) - float(self.realtime_data['最低价'])) / float(
            self.realtime_data['昨收']) * 100

        # 计算实时量比
        self.volume_ratio = float(self.realtime_data['成交量']) / self.df['成交量'].mean()

    def analyze_realtime_signals(self):
        """分析实时交易信号"""
        if self.realtime_data is None:
            return "无法获取实时数据"

        current_price = float(self.realtime_data['最新价'])
        open_price = float(self.realtime_data['开盘'])
        prev_close = float(self.realtime_data['昨收'])

        signals = []

        # 1. 价格突破分析
        for ma, position in self.price_ma_position.items():
            if position > 0.02:  # 价格站上均线2%以上
                signals.append(f"价格强势站上{ma}")
            elif position < -0.02:  # 价格跌破均线2%以上
                signals.append(f"价格弱势跌破{ma}")

        # 2. 量价配合分析
        if self.volume_ratio > 2:  # 放量
            if current_price > prev_close:
                signals.append("放量上涨，可能突破")
            else:
                signals.append("放量下跌，注意风险")

        # 3. 盘中趋势分析
        intraday_trend = (current_price - open_price) / open_price * 100
        if abs(intraday_trend) > 2:
            trend_direction = "上涨" if intraday_trend > 0 else "下跌"
            signals.append(f"盘中{trend_direction}趋势明显")

        # 4. 振幅分析
        if self.daily_amplitude > 5:
            signals.append("日内波动剧烈，注意风险")

        return signals

    def get_trading_advice(self):
        """生成交易建议"""
        if not self.update_realtime_data():
            return "无法获取实时数据，建议稍后再试"

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_price = float(self.realtime_data['最新价'])

        advice = f"\n=== 交易建议 ({current_time}) ===\n"
        advice += f"股票代码: {self.stock_code}\n"
        advice += f"当前价格: {current_price}\n"
        advice += f"涨跌幅: {self.realtime_data['涨跌幅']}%\n"
        advice += f"成交量: {self.realtime_data['成交量']}\n"
        advice += f"换手率: {self.realtime_data['换手率']}%\n\n"

        # 获取实时信号
        realtime_signals = self.analyze_realtime_signals()

        advice += "实时市场信号:\n"
        for signal in realtime_signals:
            advice += f"- {signal}\n"

        # 交易建议
        advice += "\n交易建议:\n"
        buy_signals = 0
        sell_signals = 0

        # 统计买卖信号
        for signal in realtime_signals:
            if "突破" in signal or "强势" in signal or "放量上涨" in signal:
                buy_signals += 1
            if "跌破" in signal or "弱势" in signal or "放量下跌" in signal:
                sell_signals += 1

        # 结合历史指标和实时信号给出建议
        if buy_signals > sell_signals:
            advice += "建议买入:\n"
            advice += "- 多个技术指标显示强势\n"
            advice += "- 建议分批建仓，设置止损\n"
            if self.daily_amplitude > 5:
                advice += "- 注意日内波动风险，建议限价单交易\n"
        elif sell_signals > buy_signals:
            advice += "建议卖出:\n"
            advice += "- 多个技术指标显示弱势\n"
            advice += "- 建议分批减仓，注意风险\n"
        else:
            advice += "建议观望:\n"
            advice += "- 当前无明显买卖信号\n"
            advice += "- 建议等待更明确的市场信号\n"

        return advice


def monitor_stock(stock_code='002506', interval=60):
    """
    持续监控股票
    interval: 更新间隔（秒）
    """
    analyzer = StockRealTimeAnalyzer(stock_code)

    try:
        while True:
            print("\n" + "=" * 50)
            print(analyzer.get_trading_advice())
            print(f"\n{interval}秒后更新...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n停止监控")


if __name__ == "__main__":
    # 启动实时监控
    monitor_stock(interval=60)  # 每60秒更新一次