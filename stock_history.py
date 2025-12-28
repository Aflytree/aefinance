import baostock as bs
import pandas as pd

import baostock as bs
import pandas as pd
from datetime import datetime
import efinance as ef
import  random
import  time
import akshare as ak
############################################################################
#baostock
############################################################################
def get_quote_history_baostock(stock_code, beg=None, end=None, klt='101', fqt='1', day_or_week= "d"):
    """
    仿照 efinance 接口的 Baostock 数据获取函数
    参数:
    - stock_code: 股票代码，支持 '000875' 或 '0.000875' 格式
    - beg: 开始日期，格式 'YYYYMMDD'
    - end: 结束日期，格式 'YYYYMMDD'
    - klt: K线类型（为了兼容接口，实际使用 Baostock 的日线）
    - fqt: 复权类型 1-前复权 2-后复权 0-不复权
    """

    # 设置默认日期
    if end is None:
        end = datetime.now().strftime('%Y%m%d')
    if beg is None:
        beg = '20210101'  # 默认开始日期

    # 转换日期格式
    start_date = f"{beg[:4]}-{beg[4:6]}-{beg[6:8]}"
    end_date = f"{end[:4]}-{end[4:6]}-{end[6:8]}"

    # 转换股票代码格式
    if stock_code.startswith('0.') or stock_code.startswith('1.'):
        code = stock_code.split('.')[1]  # '0.000875' -> '000875'
    else:
        code = stock_code  # '000875' -> '000875'

    # Baostock 代码格式判断
    if code.startswith('6'):
        bs_code = f"sh.{code}"  # 沪市
    else:
        bs_code = f"sz.{code}"  # 深市

    # 转换复权类型
    adjustflag_map = {'1': '2', '2': '1', '0': '3'}  # efinance 到 baostock 的映射
    adjustflag = adjustflag_map.get(fqt, '2')  # 默认前复权

    # 登录系统
    lg = bs.login()
    if lg.error_code != '0':
        print(f'Baostock登录失败: {lg.error_msg}')
        return None

    try:
        print(f"使用 Baostock 获取 {bs_code} 的数据 ({start_date} 到 {end_date})...")

        # 获取沪深A股历史K线数据
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency=day_or_week,  # 日线数据/周线
            adjustflag=adjustflag  # 复权类型
        )

        if rs.error_code != '0':
            print(f'Baostock查询失败: {rs.error_msg}')
            return None

        # 处理数据
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            print("Baostock 返回数据为空")
            return None

        # 转换为DataFrame
        df = pd.DataFrame(data_list, columns=rs.fields)

        # 重命名列以匹配 efinance 格式
        column_mapping = {
            'date': '日期',
            'open': '开盘',
            'high': '最高',
            'low': '最低',
            'close': '收盘',
            'volume': '成交量',
            'amount': '成交额',
            'turn': '换手率',
            'pctChg': '涨跌幅'
        }
        df = df.rename(columns=column_mapping)

        # 数据类型转换
        numeric_columns = ['开盘', '最高', '最低', '收盘', '成交量', '成交额', '换手率', '涨跌幅']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 计算涨跌额（基于涨跌幅）
        df['涨跌额'] = df['收盘'] * df['涨跌幅'] / 100

        # 计算振幅
        df['振幅'] = (df['最高'] - df['最低']) / df['收盘'].shift(1) * 100
        df['振幅'] = df['振幅'].fillna(0)

        # 重新排列列顺序以匹配 efinance
        efinance_columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额',
                            '振幅', '涨跌幅', '涨跌额', '换手率']

        # 确保所有列都存在
        for col in efinance_columns:
            if col not in df.columns:
                df[col] = 0  # 添加缺失的列

        df = df[efinance_columns]

        print(f"成功获取 {len(df)} 条数据")
        return df

    except Exception as e:
        print(f"Baostock 获取数据失败: {e}")
        return None
    finally:
        # 退出系统
        bs.logout()


# 使用示例 - 支持 '000875' 格式
# df = get_quote_history_baostock('000875', beg='20240123', end='20250317')
# 使用示例
# df = get_stock_data_baostock('0.002112', '2021-03-23', '2023-12-01')

############################################################################
#akshare
############################################################################
def get_stock_data_akshare(stock_code, beg='20210323', end=None):
    """使用akshare获取股票数据"""
    if end is None:
        from datetime import datetime
        end = datetime.now().strftime('%Y%m%d')

    try:
        # 转换股票代码格式
        if stock_code.startswith('0.') or stock_code.startswith('1.'):
            # efinance格式: 0.002112 -> 002112
            code = stock_code.split('.')[1]
        else:
            code = stock_code

        # 判断市场
        if code.startswith('6'):
            symbol = f"{code}.SH"
        else:
            symbol = f"{code}.SZ"

        print(f"使用akshare获取 {symbol} 的数据...")

        # 获取数据
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=beg,
            end_date=end,
            adjust="qfq"  # 前复权
        )

        if df is not None and not df.empty:
            print(f"成功获取 {len(df)} 条数据")
            # 重命名列以保持兼容性
            df = df.rename(columns={
                '日期': '日期',
                '开盘': '开盘',
                '收盘': '收盘',
                '最高': '最高',
                '最低': '最低',
                '成交量': '成交量',
                '成交额': '成交额',
                '振幅': '振幅',
                '涨跌幅': '涨跌幅',
                '涨跌额': '涨跌额',
                '换手率': '换手率'
            })
            return df
        else:
            print("akshare返回数据为空")
            return None

    except Exception as e:
        print(f"akshare获取数据失败: {e}")
        return None

############################################################################
#efince
############################################################################


def get_weekly_data_baostock(self):
    """使用baostock获取周线数据"""
    import baostock as bs

    try:
        # 登陆系统
        lg = bs.login()

        # 获取周线数据
        rs = bs.query_history_k_data_plus(
            f"sh.{self.stock_code}" if self.stock_code.startswith('6') else f"sz.{self.stock_code}",
            "date,open,high,low,close,volume,amount",
            start_date=self.beg,
            end_date=self.end,
            frequency="w",  # 周线
            adjustflag="2"  # 前复权
        )

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        df = pd.DataFrame(data_list, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # 转换数据类型
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col])

        # 重命名列
        df.rename(columns={
            'open': '开盘', 'high': '最高', 'low': '最低',
            'close': '收盘', 'volume': '成交量', 'amount': '成交额'
        }, inplace=True)

        bs.logout()
        return df

    except Exception as e:
        print(f"baostock获取周线数据失败: {e}")
        return None

def get_stock_data_with_retry(stock_code, beg, end, max_retries=5):
    for attempt in range(max_retries):
        try:
            # 添加随机延时避免请求过快
            if attempt > 0:
                wait_time = (2 ** attempt) + random.random()
                print(f"第{attempt}次重试，等待{wait_time:.2f}秒...")
                time.sleep(wait_time)

            df = ef.stock.get_quote_history(
                stock_code,
                beg=beg,
                end=end,
                # 可以尝试设置超时，但efinance可能不支持直接设置
            )
            return df

        except Exception as e:
            print(f"第{attempt + 1}次尝试失败: {e}")
            if attempt == max_retries - 1:
                print("所有重试次数已用尽")
                return None
    return None
