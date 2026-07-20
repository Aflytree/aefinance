import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
import efinance as ef  # 确保你已经导入了 ef 库
print("df")

# 获取股票历史数据
stock_data = ef.stock.get_quote_history("002119")
print("df")

# 处理数据，假设返回的数据包含 'date' 和 'close' 列
# 这里我们只选择日期和收盘价
df = stock_data[['日期', '收盘']].copy()
df.columns = ['日期', '价格']  # 重命名列
print("df")

# 将日期转换为数值
# df['日期'] = pd.to_datetime(df['日期'])
# df['日期'] = pd.to_datetime(df['日期']).map(pd.Timestamp.timestamp)
df['日期'] = df['date'] = (pd.to_datetime(df['日期']) - pd.Timestamp('1970-01-01')).dt.days

import pdb;pdb.set_trace()
print(df)
# 特征和目标变量
X = df[['日期']]
y = df['价格']

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 选取最近252个交易日的数据
# df = df.sort_values(by='日期', ascending=False).head(252)
# 创建贝叶斯回归模型
model = BayesianRidge()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
# import pdb;pdb.set_trace()
# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(df['日期'], y, color='blue', label='actual price')
plt.scatter(df['日期'].iloc[X_test.index], y_pred, color='red', label='predict price')
plt.xlabel('date')
plt.ylabel('price')
plt.title('贝叶斯线性回归预测股票价格')
plt.xticks(rotation=45)  # 旋转日期标签以便于阅读
plt.legend()
plt.show()