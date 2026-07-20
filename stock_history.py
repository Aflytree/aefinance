import baostock as bs
import pandas as pd
from datetime import datetime
import efinance as ef
import random
import socket
import time
import akshare as ak
from urllib.parse import urlparse

_efinance_curl_session_ready = False
_eastmoney_host_ip_cache: dict[str, str] = {}


def _is_dns_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "could not resolve host" in msg
        or "name or service not known" in msg
        or "getaddrinfo failed" in msg
        or "curl: (6)" in msg
    )


def _resolve_host_ip(host: str, port: int = 443) -> str | None:
    cached = _eastmoney_host_ip_cache.get(host)
    if cached:
        return cached
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        ip = infos[0][4][0]
        _eastmoney_host_ip_cache[host] = ip
        return ip
    except Exception:
        try:
            ip = socket.gethostbyname(host)
            _eastmoney_host_ip_cache[host] = ip
            return ip
        except Exception:
            return None


def _warm_eastmoney_dns() -> None:
    for host in ("push2his.eastmoney.com", "push2.eastmoney.com"):
        _resolve_host_ip(host)


class _CurlCffiSession:
    """用 curl_cffi 模拟 Chrome TLS 指纹，绕过 push2his 对 Python urllib3 的拦截。"""

    trust_env = False
    headers = {}

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def request(self, method, url, **kwargs):
        from curl_cffi import requests as curl_requests

        headers = {**self.headers, **kwargs.pop("headers", {})}
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        last_err: Exception | None = None
        for dns_try in range(3):
            req_kwargs = {
                "headers": headers,
                "timeout": kwargs.get("timeout", 180),
                "impersonate": "chrome131",
                "verify": kwargs.get("verify", True),
                "params": kwargs.get("params"),
            }
            if dns_try > 0 and host:
                _eastmoney_host_ip_cache.pop(host, None)
                ip = _resolve_host_ip(host, port)
                if ip:
                    req_kwargs["resolve"] = [f"{host}:{port}:{ip}"]
                    print(f"DNS 重试: 使用 {host} -> {ip}")
                time.sleep(min(2 ** dns_try, 8))

            try:
                return curl_requests.request(method, url, **req_kwargs)
            except Exception as e:
                last_err = e
                if _is_dns_error(e) and dns_try < 2:
                    continue
                raise

        if last_err is not None:
            raise last_err
        raise RuntimeError(f"请求失败: {url}")


def setup_efinance_curl_session(force: bool = False) -> bool:
    """将 efinance 底层 session 替换为 curl_cffi，仅需初始化一次。"""
    global _efinance_curl_session_ready
    if _efinance_curl_session_ready and not force:
        return True
    try:
        import efinance.shared as ef_shared
        import efinance.common.getter as ef_getter

        session = _CurlCffiSession()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                "Referer": "https://quote.eastmoney.com/",
                "Accept": "application/json, text/plain, */*",
            }
        )
        ef_shared.session = session
        ef_getter.session = session
        _efinance_curl_session_ready = True
        _warm_eastmoney_dns()
        return True
    except ImportError:
        print("curl_cffi 未安装，efinance 可能无法拉取 push2his 数据；请执行: pip install curl_cffi")
        return False
    except Exception as e:
        print(f"初始化 efinance curl session 失败: {e}")
        return False

# Baostock 底层 recv 无超时，网络挂起时会永久阻塞；对 default_socket 设置读超时（秒）
DEFAULT_BAOSTOCK_RECV_TIMEOUT = 90.0


def apply_baostock_socket_timeout(seconds: float = DEFAULT_BAOSTOCK_RECV_TIMEOUT) -> None:
    """为当前 Baostock 会话的 socket 设置读超时，避免单次请求无限等待。"""
    try:
        from baostock.common import context as bs_ctx

        sock_obj = getattr(bs_ctx, "default_socket", None)
        if sock_obj is not None and hasattr(sock_obj, "settimeout"):
            sock_obj.settimeout(float(seconds))
    except Exception:
        pass


############################################################################
#baostock
############################################################################
def get_quote_history_baostock(
    stock_code,
    beg=None,
    end=None,
    klt="101",
    fqt="1",
    day_or_week="d",
    manage_login=True,
    verbose=True,
):
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

    if manage_login:
        lg = bs.login()
        if lg.error_code != "0":
            print(f"Baostock登录失败: {lg.error_msg}")
            return None

    try:
        apply_baostock_socket_timeout()
        if verbose:
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

        if rs.error_code != "0":
            if verbose:
                print(f"Baostock查询失败: {rs.error_msg}")
            return None

        # 处理数据
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            if verbose:
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

        if verbose:
            print(f"成功获取 {len(df)} 条数据")
        return df

    except Exception as e:
        if verbose:
            print(f"Baostock 获取数据失败: {e}")
        return None
    finally:
        if manage_login:
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
        apply_baostock_socket_timeout()

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

def _latest_bar_date(df):
    if df is None or df.empty or "日期" not in df.columns:
        return None
    try:
        return pd.to_datetime(df["日期"]).max().date()
    except Exception:
        return None


def _is_after_a_share_close(now: datetime | None = None) -> bool:
    """A股常规收盘后（含清理缓冲），日线才较完整。"""
    now = now or datetime.now()
    return now.hour > 15 or (now.hour == 15 and now.minute >= 10)


def _accept_daily_bar(df, stock_code: str, source: str) -> bool:
    """
    efinance：盘中通常已有当日未收盘K，可直接用。
    akshare hist / baostock：官方/实际都是日频盘后更新，盘中缺当日则不能当“今日信号”数据源。
    """
    latest = _latest_bar_date(df)
    today = datetime.now().date()
    print(f"{stock_code} {source} 最新K线: {latest}")
    if latest is None:
        return False
    if latest >= today:
        return True
    if today.weekday() >= 5:
        # 周末用最近交易日即可
        return True
    if not _is_after_a_share_close():
        print(
            f"拒绝 {source}: 当前盘中/早盘，无当日K线(最新 {latest})，"
            f"不可用于今日买卖判定"
        )
        return False
    print(
        f"警告: {source} 收盘后仍缺当日K线(最新 {latest})，"
        f"今日信号可能无法判定"
    )
    # 收盘后仍缺当日：不当作成功（避免 silent 用昨日报今日）
    return False


def get_stock_data_with_retry(stock_code, beg, end, max_retries=3):
    """
    日线获取策略（针对“要含当日交易日”的场景）：
    1) efinance：盘中即可带出当日K，作主源并多试几次
    2) akshare stock_zh_a_hist：文档写明“当日收盘价请在收盘后获取”，盘中一般无当日
    3) baostock：盘中无实时日线，仅盘后；盘中缺当日一律不用
    """
    setup_efinance_curl_session()
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = (2 ** attempt) + random.random()
                print(f"第{attempt}次重试，等待{wait_time:.2f}秒...")
                time.sleep(wait_time)
                if attempt >= 2:
                    setup_efinance_curl_session(force=True)

            time.sleep(random.uniform(0.3, 0.8))
            df = ef.stock.get_quote_history(
                stock_code,
                beg=beg,
                end=end,
            )
            if df is not None and not df.empty:
                if _accept_daily_bar(df, stock_code, "efinance"):
                    return df
                print(f"{stock_code} efinance 无合格当日K线，继续重试/换源")
            else:
                print(f"{stock_code} efinance 返回空数据")

        except Exception as e:
            print(f"第{attempt + 1}次尝试失败: {e}")
            if _is_dns_error(e):
                _eastmoney_host_ip_cache.clear()
                _warm_eastmoney_dns()

    # 收盘后才值得用 akshare/baostock 补当日；盘中直接失败更清晰
    if not _is_after_a_share_close():
        print(
            f"{stock_code} efinance 失败且当前未收盘："
            f"akshare/baostock 盘中通常无当日日线，放弃兜底"
        )
        return None

    print(f"{stock_code} efinance 失败（已收盘），尝试 akshare 日线...")
    try:
        df = get_stock_data_akshare(stock_code, beg=beg, end=end)
        if df is not None and not df.empty and _accept_daily_bar(df, stock_code, "akshare"):
            return df
    except Exception as e:
        print(f"{stock_code} akshare 失败: {e}")

    print(f"{stock_code} 尝试 baostock 日线（仅盘后）...")
    try:
        df = get_quote_history_baostock(stock_code, beg=beg, end=end, day_or_week="d")
        if df is not None and not df.empty and _accept_daily_bar(df, stock_code, "baostock"):
            return df
    except Exception as e:
        print(f"{stock_code} baostock 失败: {e}")

    print(f"{stock_code} 所有数据源均失败或缺少当日K线")
    return None
