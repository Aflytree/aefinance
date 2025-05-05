import numpy as np
import pandas as pd
import backtrader as bt
import efinance as ef
from deap import base, creator, tools, algorithms

import akshare as ak



# 1. 获取数据（增加数据量检查）
def get_data(stock_code="600519", start_date="2015-01-01", end_date="2020-12-31"):
    # df = ef.stock.get_quote_history(stock_code, beg=start_date, end=end_date, klt=101)
    df = ef.stock.get_quote_history(stock_code, beg='20150101', end='20201231')
    print(df.head())  # 检查数据
    df['日期'] = pd.to_datetime(df['日期'])
    df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df.set_index('datetime', inplace=True)

    # 数据长度检查
    min_data_length = 100  # 至少需要100根K线
    if len(df) < min_data_length:
        raise ValueError(f"数据不足！当前{len(df)}条，至少需要{min_data_length}条")
    return df


# 2. 策略改进（增加数据长度校验）
class SafeDualMAStrategy(bt.Strategy):
    params = (
        ('fast', 10),
        ('slow', 30),
        ('min_bars', 50)  # 最小数据要求
    )

    def __init__(self):
        # 检查数据是否足够计算最长均线
        if len(self.data) < max(self.p.fast, self.p.slow, self.p.min_bars):
            raise bt.StrategySkipError("数据长度不足")

        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()


# 3. 适应度函数（增加异常处理）
def evaluate(individual):
    try:
        # 参数有效性检查
        fast, slow = int(individual[0]), int(individual[1])
        if fast >= slow or fast < 5 or slow > 50:  # 保证 fast < slow 且范围合理
            return (-1000,)

        cerebro = bt.Cerebro()
        cerebro.adddata(data)
        cerebro.addstrategy(SafeDualMAStrategy, fast=fast, slow=slow)
        cerebro.broker.set_cash(100000)
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

        results = cerebro.run()
        ann_return = results[0].analyzers.returns.get_analysis()['rnorm100']
        return (ann_return,)
    except Exception as e:
        print(f"参数{individual}失败: {str(e)}")
        return (-1000,)


# 主程序
if __name__ == '__main__':
    # 获取数据
    try:
        df = get_data()
        data = bt.feeds.PandasData(dataname=df)
    except Exception as e:
        print("数据获取失败:", e)
        exit()

    # 遗传算法配置
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", np.random.randint, 5, 50)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=5, up=50, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 运行优化
    population = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print("开始优化...")
    population, log = algorithms.eaSimple(
        population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10,
        stats=stats, halloffame=hof, verbose=True
    )

    # 输出结果
    best = hof[0]
    print(f"\n最优参数: fast={best[0]}, slow={best[1]}")