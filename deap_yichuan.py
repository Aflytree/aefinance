import numpy as np
import pandas as pd
import backtrader as bt
from deap import base, creator, tools, algorithms
import efinance  as ef
import akshare as ak
import matplotlib.pyplot as plt
from backtrader.plot import Plot_OldSync


# 4. 可视化函数
def plot_trades(df, signals, fast, slow):
    plt.figure(figsize=(16, 8))
    plt.plot(df['close'], label='Price', alpha=0.5)

    # 标记买卖点
    buy_dates = [s[0] for s in signals if s[1] == 'BUY']
    buy_prices = [s[2] for s in signals if s[1] == 'BUY']
    sell_dates = [s[0] for s in signals if s[1] == 'SELL']
    sell_prices = [s[2] for s in signals if s[1] == 'SELL']

    plt.scatter(buy_dates, buy_prices, marker='^', color='g', label='Buy', s=100)
    plt.scatter(sell_dates, sell_prices, marker='v', color='r', label='Sell', s=100)

    # 添加均线
    df['fast_ma'] = df['close'].rolling(fast).mean()
    df['slow_ma'] = df['close'].rolling(slow).mean()
    plt.plot(df['fast_ma'], label=f'{fast}日均线', linestyle='--')
    plt.plot(df['slow_ma'], label=f'{slow}日均线', linestyle='--')

    plt.title(f'双均线策略交易信号 (快线={fast}, 慢线={slow})')
    plt.legend()
    plt.grid()
    plt.show()

# 1. 准备历史数据（示例使用AAPL数据）
# data = bt.feeds.YahooFinanceData(dataname='AAPL',
#                                  fromdate=pd.to_datetime('2015-01-01'),
#                                  todate=pd.to_datetime('2020-12-31'))
# import pdb;pdb.set_trace()
# print(f"数据量: {len(data)}条")

# 检查列名是否匹配
# required_cols = ['open', 'high', 'low', 'close', 'volume']
# assert all(col in data.columns for col in required_cols), f"缺失列: {set(required_cols) - set(data.columns)}"
data = ef.stock.get_quote_history('002119', beg='20190323', end='20250323')

#
print("股票行情数据下载完毕")
data['日期'] = pd.to_datetime(data['日期'])
data.set_index('日期', inplace=True)

print("显示股票行情数据")
print(data)
print(f"数据量: {len(data)}条")
print(data.head(200))
# import pdb;pdb.set_trace()
# # 必须执行的列名转换和类型检查
data = data.rename(columns={
    '日期': 'datetime',
    '开盘': 'open',
    '最高': 'high',
    '最低': 'low',
    '收盘': 'close',
    '成交量': 'volume'
})
# print(data)

df1 = data.head(800)
df2 = data.tail(500)
# 转换为 Backtrader 数据格式
data1 = bt.feeds.PandasData(dataname=df1)
data2 = bt.feeds.PandasData(dataname=df2)

# 2. 定义交易策略
class DualMAStrategy(bt.Strategy):
    params = (
        ('fast', 10),
        ('slow', 30)
    )

    def __init__(self):
        # 正确初始化实例属性
        self.trade_signals = []  # 每个策略实例独立拥有
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
                self.trade_signals.append((
                    self.data.datetime.date(0),
                    'BUY',
                    self.data.close[0]
                ))
        elif self.crossover < 0:
            self.close()
            self.trade_signals.append((
                self.data.datetime.date(0),
                'SELL',
                self.data.close[0]
            ))

# 新增函数获取交易信号
def get_trade_signals(fast, slow):
    cerebro = bt.Cerebro()
    cerebro.adddata(data1)
    cerebro.addstrategy(DualMAStrategy, fast=fast, slow=slow)
    results = cerebro.run()
    return results[0].trade_signals

# 3. 遗传算法适应度函数
def evaluate(individual):
    try:
        fast, slow = int(individual[0]), int(individual[1])

        # 参数校验
        if fast >= slow or fast < 5 or slow > 50:
            return (float('nan'),)  # 必须返回单值元组

        cerebro = bt.Cerebro(stdstats=False)
        cerebro.adddata(data)
        cerebro.addstrategy(DualMAStrategy, fast=fast, slow=slow)
        cerebro.broker.set_cash(100000)
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

        results = cerebro.run()
        strat = results[0]

        # 仅返回适应度值（必须与weights长度一致）
        ann_return = strat.analyzers.returns.get_analysis().get('rnorm100', float('nan'))
        return (ann_return,)  # 注意逗号表示单元素元组

    except Exception as e:
        print(f"评估失败: {str(e)}")
        return (float('nan'),)  # 保持返回结构一致


# 4. 遗传算法配置
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", np.random.randint, 5, 50)  # 参数范围5-50天
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=5, up=50, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 5. 运行遗传算法
population = toolbox.population(n=20)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)

population, log = algorithms.eaSimple(population, toolbox,
                                      cxpb=0.5, mutpb=0.2,
                                      ngen=20, stats=stats,
                                      halloffame=hof, verbose=True)

# 6. 输出最优参数
best_params = hof[0]
best_fitness = evaluate(best_params)
best_signals = get_trade_signals(best_params[0], best_params[1])
# best_signals = strat.trade_signals
print(f"\n最优参数: fast={best_params[0]}, slow={best_params[1]}, 年化收益={best_fitness:.2f}%")

# 绘制交易信号
plot_trades(data1, best_signals, best_params[0], best_params[1])

# 绘制资金曲线
cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.addstrategy(DualMAStrategy, fast=best_params[0], slow=best_params[1])
cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl')
results = cerebro.run()
strat = results[0]

plt.figure(figsize=(16, 6))
plt.plot(pd.Series(strat.analyzers.pnl.get_analysis()).cumsum(), label='累计收益')
plt.title('资金曲线')
plt.legend()
plt.grid()
plt.show()