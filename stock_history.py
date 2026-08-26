import baostock as bs
import pandas as pd
from datetime import datetime
import efinance as ef
import os
import random
import socket
import sys
import time
import akshare as ak
import requests
from pathlib import Path
from urllib.parse import urlparse

_TICKPLUS_ROOT = Path(__file__).resolve().parent / "TickPlusSkill"
_TICKPLUS_PLACEHOLDER_TOKENS = {"", "123456789", "your_token_here"}
_tickplus_token_warned = False
_tickplus_disabled = False
_baostock_logged_in = False
BULK_DATA_MODE = False
# 日线优先读本地 Excel（history_daily/），当天缺口用 overlay / 新浪分钟线拼凑，缺文件才走网络
USE_LOCAL_DAILY_FIRST = True
FETCH_TODAY_BAR = True  # 用 akshare 新浪分钟线合成当日日线并拼进去
_LOCAL_DAILY_DIR = Path(__file__).resolve().parent / "history_daily"
_LOCAL_OVERLAY_DIR = _LOCAL_DAILY_DIR / "overlay"


def _baostock_ensure_login() -> bool:
    """复用同一 Baostock 会话，避免每只股票反复 login/logout 被断开。"""
    global _baostock_logged_in
    if _baostock_logged_in:
        apply_baostock_socket_timeout()
        return True
    for attempt in range(3):
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(25)
        try:
            lg = bs.login()
        except Exception as e:
            print(f"Baostock登录异常({attempt + 1}/3): {e}")
            lg = None
        finally:
            socket.setdefaulttimeout(old_timeout)
        if lg is not None and lg.error_code == "0":
            apply_baostock_socket_timeout()
            _baostock_logged_in = True
            return True
        if lg is not None:
            print(f"Baostock登录失败({attempt + 1}/3): {lg.error_msg}")
        time.sleep(1.5)
    _baostock_logged_in = False
    return False

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
DEFAULT_BAOSTOCK_RECV_TIMEOUT = 25.0


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
    _retried=False,
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

    if manage_login and not _baostock_ensure_login():
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
        global _baostock_logged_in
        _baostock_logged_in = False
        if manage_login and not _retried and _baostock_ensure_login():
            return get_quote_history_baostock(
                stock_code,
                beg=beg,
                end=end,
                klt=klt,
                fqt=fqt,
                day_or_week=day_or_week,
                manage_login=True,
                verbose=verbose,
                _retried=True,
            )
        return None


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


def _ensure_tickplus_on_path() -> None:
    root = str(_TICKPLUS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _normalize_a_share_code(stock_code: str) -> str:
    code = str(stock_code).strip()
    if code.startswith("0.") or code.startswith("1."):
        return code.split(".", 1)[1]
    if "." in code:
        return code.split(".", 1)[0]
    return code


def _yyyymmdd_to_iso(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).replace("-", "").replace("/", "")[:8]
    if len(text) != 8 or not text.isdigit():
        return str(value)
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def _get_tickplus_token() -> str | None:
    for key in ("TICKPLUS_TOKEN", "TICKPLUS_API_TOKEN"):
        env_token = (os.environ.get(key) or "").strip()
        if env_token and env_token not in _TICKPLUS_PLACEHOLDER_TOKENS:
            return env_token
    try:
        _ensure_tickplus_on_path()
        from tickplus.scripts.Config import Config

        token = (getattr(Config, "TOKEN", "") or "").strip()
        if token and token not in _TICKPLUS_PLACEHOLDER_TOKENS:
            return token
    except Exception as e:
        print(f"读取 TickPlus token 失败: {e}")
    return None


def _tickplus_kline_to_df(data) -> pd.DataFrame | None:
    if isinstance(data, dict):
        message = str(data.get("message") or data.get("msg") or data)
        print(f"TickPlus 返回异常: {message}")
        global _tickplus_disabled
        if "权限" in message or "token" in message.lower():
            _tickplus_disabled = True
            print("TickPlus 接口不可用，后续股票跳过该数据源")
        return None
    if not isinstance(data, list) or not data:
        return None

    df = pd.DataFrame(data)
    column_mapping = {
        "t": "日期",
        "code": "股票代码",
        "o": "开盘",
        "c": "收盘",
        "h": "最高",
        "l": "最低",
        "v": "成交量",
        "a": "成交额",
        "pc": "昨收",
    }
    df = df.rename(columns=column_mapping)
    if "日期" not in df.columns:
        print("TickPlus K线缺少日期字段")
        return None

    df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
    numeric_columns = ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "昨收"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    prev_close = df["昨收"] if "昨收" in df.columns else df["收盘"].shift(1)
    prev_close = prev_close.replace(0, pd.NA)
    df["涨跌额"] = df["收盘"] - prev_close
    df["涨跌幅"] = df["涨跌额"] / prev_close * 100
    df["振幅"] = (df["最高"] - df["最低"]) / prev_close * 100
    df["换手率"] = 0
    df["涨跌额"] = df["涨跌额"].fillna(0)
    df["涨跌幅"] = df["涨跌幅"].fillna(0)
    df["振幅"] = df["振幅"].fillna(0)

    efinance_columns = [
        "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
        "振幅", "涨跌幅", "涨跌额", "换手率",
    ]
    return df[efinance_columns].sort_values("日期").reset_index(drop=True)


def get_quote_history_tickplus(
    stock_code,
    beg=None,
    end=None,
    period="1d",
    dividend="2",
    verbose=True,
):
    """
    使用工程内 TickPlusSkill 获取 K 线，列格式对齐 efinance/baostock。
    period: 1d 日线 / 1w 周线；dividend: 1不复权 2前复权 3后复权。
    token 优先读环境变量 TICKPLUS_TOKEN，否则用 TickPlusSkill/tickplus/scripts/Config.py。
    """
    global _tickplus_disabled
    if _tickplus_disabled:
        return None

    token = _get_tickplus_token()
    if not token:
        global _tickplus_token_warned
        if verbose and not _tickplus_token_warned:
            print(
                "TickPlus 未配置有效 token，跳过。"
                "请设置环境变量 TICKPLUS_TOKEN，或填写 TickPlusSkill/tickplus/scripts/Config.py"
            )
            _tickplus_token_warned = True
        return None

    if end is None:
        end = datetime.now().strftime("%Y%m%d")
    if beg is None:
        beg = "20210101"

    code = _normalize_a_share_code(stock_code)
    start_date = _yyyymmdd_to_iso(beg)
    end_date = _yyyymmdd_to_iso(end)
    if verbose:
        print(f"使用 TickPlus 获取 {code} 的数据 ({start_date} 到 {end_date})...")

    try:
        _ensure_tickplus_on_path()
        from tickplus.scripts.api import BasicApi
        from tickplus.scripts.util import DataUtil

        original_print_log = DataUtil.printLog
        original_request = DataUtil.request

        def _request_with_timeout(url, method="get", params=None):
            import requests as _requests

            if method == "post":
                return _requests.post(url, params=params, timeout=20).content.decode("utf-8")
            return _requests.get(url, params=params, timeout=20).content.decode("utf-8")

        DataUtil.printLog = lambda *args, **kwargs: None
        DataUtil.request = _request_with_timeout
        try:
            data = BasicApi.getDayKline(
                symbol="stock",
                code=code,
                period=period,
                dividend=str(dividend),
                startDate=start_date,
                endDate=end_date,
                token=token,
            )
        finally:
            DataUtil.printLog = original_print_log
            DataUtil.request = original_request

        df = _tickplus_kline_to_df(data)
        if df is None or df.empty:
            if verbose:
                print("TickPlus 返回数据为空")
            return None
        if verbose:
            print(f"成功获取 {len(df)} 条 TickPlus 数据")
        return df
    except Exception as e:
        _tickplus_disabled = True
        if verbose:
            print(f"TickPlus 获取数据失败: {e}")
            print("TickPlus 接口不可用，后续股票跳过该数据源")
        return None


_KLINE_CACHE_DIR = Path(__file__).resolve().parent / "kline_cache"
_http_session = None


def _get_http_session():
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        _http_session.trust_env = False
        _http_session.proxies = {"http": None, "https": None}
        _http_session.headers.update({"User-Agent": "Mozilla/5.0"})
    return _http_session


def _tencent_symbol(stock_code: str) -> str:
    code = _normalize_a_share_code(stock_code)
    prefix = "sh" if code.startswith("6") else "sz"
    return f"{prefix}{code}"


def _ohlcv_to_efinance_df(rows: list[dict]) -> pd.DataFrame | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for col in ["开盘", "收盘", "最高", "最低", "成交量"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    prev = df["收盘"].shift(1)
    df["涨跌额"] = (df["收盘"] - prev).fillna(0)
    df["涨跌幅"] = (df["涨跌额"] / prev.replace(0, pd.NA) * 100).fillna(0)
    df["振幅"] = ((df["最高"] - df["最低"]) / prev.replace(0, pd.NA) * 100).fillna(0)
    df["成交额"] = 0
    df["换手率"] = 0
    cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
    return df[cols].sort_values("日期").reset_index(drop=True)


def get_quote_history_tencent(stock_code, beg, end, timeout=8):
    """腾讯前复权日线，HTTP 直连，适合当前网络。"""
    symbol = _tencent_symbol(stock_code)
    start = _yyyymmdd_to_iso(beg)
    end_iso = _yyyymmdd_to_iso(end)
    url = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    param = f"{symbol},day,{start},{end_iso},800,qfq"
    last_err = None
    for attempt in range(3):
        try:
            r = _get_http_session().get(url, params={"param": param}, timeout=timeout)
            r.raise_for_status()
            data = (r.json().get("data") or {}).get(symbol) or {}
            bars = data.get("qfqday") or data.get("day") or []
            rows = []
            for bar in bars:
                rows.append({
                    "日期": bar[0],
                    "开盘": bar[1],
                    "收盘": bar[2],
                    "最高": bar[3],
                    "最低": bar[4],
                    "成交量": bar[5],
                })
            df = _ohlcv_to_efinance_df(rows)
            if df is not None and not df.empty:
                print(f"成功获取 {len(df)} 条腾讯日线")
                return df
            print("腾讯日线返回为空")
            return None
        except Exception as e:
            last_err = e
            print(f"腾讯日线第{attempt + 1}次失败: {e}")
            time.sleep(0.4)
    if last_err:
        print(f"{stock_code} 腾讯日线失败: {last_err}")
    return None


def get_quote_history_sina(stock_code, beg, end, timeout=8):
    code = _normalize_a_share_code(stock_code)
    symbol = ("sh" if code.startswith("6") else "sz") + code
    url = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    try:
        r = _get_http_session().get(
            url,
            params={"symbol": symbol, "scale": 240, "ma": "no", "datalen": 1023},
            timeout=timeout,
        )
        r.raise_for_status()
        bars = r.json() or []
        rows = []
        start = _yyyymmdd_to_iso(beg)
        end_iso = _yyyymmdd_to_iso(end)
        for bar in bars:
            day = str(bar.get("day") or "")
            if day < start or day > end_iso:
                continue
            rows.append({
                "日期": day,
                "开盘": bar.get("open"),
                "收盘": bar.get("close"),
                "最高": bar.get("high"),
                "最低": bar.get("low"),
                "成交量": bar.get("volume"),
            })
        df = _ohlcv_to_efinance_df(rows)
        if df is not None and not df.empty:
            print(f"成功获取 {len(df)} 条新浪日线")
            return df
    except Exception as e:
        print(f"{stock_code} 新浪日线失败: {e}")
    return None


def _kline_cache_path(stock_code, beg, end) -> Path:
    code = _normalize_a_share_code(stock_code)
    return _KLINE_CACHE_DIR / f"{code}_{beg}_{end}.csv"


def _load_kline_cache(stock_code, beg, end):
    path = _kline_cache_path(stock_code, beg, end)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df is not None and not df.empty:
            print(f"{stock_code} 使用本地K线缓存 {len(df)} 条")
            return df
    except Exception as e:
        print(f"读取K线缓存失败: {e}")
    return None


def _save_kline_cache(stock_code, beg, end, df):
    try:
        _KLINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_kline_cache_path(stock_code, beg, end), index=False)
    except Exception as e:
        print(f"写入K线缓存失败: {e}")


def _find_local_daily_workbook() -> Path | None:
    """取 history_daily/ 下最新的 daily_from_*.xlsx。"""
    if not _LOCAL_DAILY_DIR.exists():
        return None
    files = sorted(
        _LOCAL_DAILY_DIR.glob("daily_from_*.xlsx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame | None:
    """把任意日线表整理成与 efinance 兼容的列。"""
    if df is None or df.empty:
        return None
    d = df.copy()
    rename = {}
    for c in list(d.columns):
        cs = str(c).strip().lower()
        if cs in ("日期", "date", "time", "datetime", "交易日期"):
            rename[c] = "日期"
        elif cs in ("开盘", "open"):
            rename[c] = "开盘"
        elif cs in ("收盘", "close"):
            rename[c] = "收盘"
        elif cs in ("最高", "high"):
            rename[c] = "最高"
        elif cs in ("最低", "low"):
            rename[c] = "最低"
        elif cs in ("成交量", "volume", "vol"):
            rename[c] = "成交量"
        elif cs in ("成交额", "amount"):
            rename[c] = "成交额"
    d = d.rename(columns=rename)
    need = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
    if any(c not in d.columns for c in need):
        return None
    rows = []
    for _, r in d.iterrows():
        rows.append({
            "日期": pd.to_datetime(r["日期"]).strftime("%Y-%m-%d"),
            "开盘": r["开盘"],
            "收盘": r["收盘"],
            "最高": r["最高"],
            "最低": r["最低"],
            "成交量": r["成交量"],
        })
    return _ohlcv_to_efinance_df(rows)


def _filter_by_beg_end(df: pd.DataFrame, beg, end) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    start = _yyyymmdd_to_iso(str(beg))
    end_iso = _yyyymmdd_to_iso(str(end))
    dates = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
    mask = (dates >= start) & (dates <= end_iso)
    return df.loc[mask].reset_index(drop=True)


def load_local_daily_history(stock_code, beg=None, end=None) -> pd.DataFrame | None:
    """
    从 history_daily/daily_from_*.xlsx 读历史日线。
    优先读同名 sheet（股票代码），否则从「全部日线」按代码过滤。
    """
    path = _find_local_daily_workbook()
    if path is None:
        return None
    code = _normalize_a_share_code(stock_code)
    try:
        with pd.ExcelFile(path) as xl:
            if code in xl.sheet_names:
                raw = pd.read_excel(xl, sheet_name=code)
            elif "全部日线" in xl.sheet_names:
                raw = pd.read_excel(xl, sheet_name="全部日线")
                if "代码" in raw.columns:
                    raw["代码"] = raw["代码"].astype(str).str.zfill(6)
                    raw = raw[raw["代码"] == code]
                else:
                    return None
            else:
                return None
        df = _normalize_ohlcv_df(raw)
        if df is None or df.empty:
            return None
        if beg is not None and end is not None:
            df = _filter_by_beg_end(df, beg, end)
        print(f"{code} 本地Excel日线 {len(df)} 条 ({path.name})")
        return df
    except Exception as e:
        print(f"{code} 读取本地Excel失败: {e}")
        return None


def load_overlay_daily_bars(stock_code) -> pd.DataFrame | None:
    """
    读取当天/增量日线拼凑文件（你后续用别的接口写入即可）：
      history_daily/overlay/{code}.csv
      history_daily/overlay/{code}.xlsx
      history_daily/overlay/today.csv   （含「代码」列）
      history_daily/overlay/today.xlsx
    列至少：日期,开盘,收盘,最高,最低,成交量
    """
    code = _normalize_a_share_code(stock_code)
    _LOCAL_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [
        _LOCAL_OVERLAY_DIR / f"{code}.csv",
        _LOCAL_OVERLAY_DIR / f"{code}.xlsx",
        _LOCAL_OVERLAY_DIR / "today.csv",
        _LOCAL_OVERLAY_DIR / "today.xlsx",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".csv":
                raw = pd.read_csv(path)
            else:
                raw = pd.read_excel(path)
            if "代码" in raw.columns:
                raw["代码"] = raw["代码"].astype(str).str.zfill(6)
                raw = raw[raw["代码"] == code]
            df = _normalize_ohlcv_df(raw)
            if df is not None and not df.empty:
                print(f"{code} overlay 增量日线 {len(df)} 条 ({path.name})")
                return df
        except Exception as e:
            print(f"{code} 读取 overlay {path.name} 失败: {e}")
    return None


def merge_daily_with_overlay(base: pd.DataFrame, overlay: pd.DataFrame | None) -> pd.DataFrame:
    """按日期合并；overlay 覆盖同日，用于拼当天/修正最新一根。"""
    if overlay is None or overlay.empty:
        return base
    if base is None or base.empty:
        return overlay.copy()
    b = base.copy()
    o = overlay.copy()
    b["日期"] = pd.to_datetime(b["日期"]).dt.strftime("%Y-%m-%d")
    o["日期"] = pd.to_datetime(o["日期"]).dt.strftime("%Y-%m-%d")
    merged = pd.concat([b, o], ignore_index=True)
    merged = merged.drop_duplicates(subset=["日期"], keep="last")
    merged = merged.sort_values("日期").reset_index(drop=True)
    print(
        f"日线拼凑完成: base={len(b)} + overlay={len(o)} -> {len(merged)}，"
        f"最新={merged['日期'].iloc[-1]}"
    )
    return merged


def _sina_symbol(stock_code: str) -> str:
    code = _normalize_a_share_code(stock_code)
    return ("sh" if code.startswith("6") else "sz") + code


def fetch_today_bar_sina_minute(stock_code) -> pd.DataFrame | None:
    """
    akshare 新浪 1 分钟线 -> 合成当日日线 OHLCV。
    成交量按「股/100=手」与腾讯日线对齐。
    """
    code = _normalize_a_share_code(stock_code)
    symbol = _sina_symbol(code)
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        raw = ak.stock_zh_a_minute(symbol=symbol, period="1", adjust="")
        if raw is None or raw.empty or "day" not in raw.columns:
            print(f"{code} 新浪分钟线为空")
            return None
        d = raw[raw["day"].astype(str).str.startswith(today)].copy()
        if d.empty:
            print(f"{code} 新浪分钟线无今日({today})数据")
            return None
        for c in ["open", "high", "low", "close", "volume"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["open", "high", "low", "close"])
        if d.empty:
            return None
        vol_hands = float(d["volume"].sum()) / 100.0
        row = {
            "日期": today,
            "开盘": float(d["open"].iloc[0]),
            "收盘": float(d["close"].iloc[-1]),
            "最高": float(d["high"].max()),
            "最低": float(d["low"].min()),
            "成交量": vol_hands,
        }
        df = _ohlcv_to_efinance_df([row])
        print(
            f"{code} 新浪分钟线合成当日: O={row['开盘']} H={row['最高']} "
            f"L={row['最低']} C={row['收盘']} V={row['成交量']:.0f} ({len(d)}根分钟)"
        )
        return df
    except Exception as e:
        print(f"{code} 新浪分钟线合成失败: {e}")
        return None


def fetch_today_bar_sina_intraday(stock_code) -> pd.DataFrame | None:
    """akshare stock_intraday_sina 逐笔/分时 -> 合成当日日线（分钟线失败时兜底）。"""
    code = _normalize_a_share_code(stock_code)
    symbol = _sina_symbol(code)
    today = datetime.now().date()
    date_str = today.strftime("%Y%m%d")
    try:
        raw = ak.stock_intraday_sina(symbol=symbol, date=date_str)
        if raw is None or raw.empty:
            print(f"{code} 新浪分时(intraday)为空")
            return None
        price_col = "price" if "price" in raw.columns else None
        if price_col is None:
            return None
        prices = pd.to_numeric(raw[price_col], errors="coerce").dropna()
        if prices.empty:
            return None
        vol = 0.0
        if "volume" in raw.columns:
            # 新浪分时 volume 多为累计或增量，取最后一档累计更稳
            vols = pd.to_numeric(raw["volume"], errors="coerce").dropna()
            if not vols.empty:
                vol = float(vols.iloc[-1]) / 100.0
        row = {
            "日期": today.strftime("%Y-%m-%d"),
            "开盘": float(prices.iloc[0]),
            "收盘": float(prices.iloc[-1]),
            "最高": float(prices.max()),
            "最低": float(prices.min()),
            "成交量": vol,
        }
        df = _ohlcv_to_efinance_df([row])
        print(
            f"{code} 新浪分时合成当日: O={row['开盘']} H={row['最高']} "
            f"L={row['最低']} C={row['收盘']} V={row['成交量']:.0f}"
        )
        return df
    except Exception as e:
        print(f"{code} 新浪分时合成失败: {e}")
        return None


def fetch_today_daily_bar(stock_code) -> pd.DataFrame | None:
    """当天日线：优先新浪分钟线，其次新浪分时。"""
    df = fetch_today_bar_sina_minute(stock_code)
    if df is not None and not df.empty:
        return df
    return fetch_today_bar_sina_intraday(stock_code)


def get_stock_data_with_retry(stock_code, beg, end, max_retries=3):
    """
    日线获取策略：
    1) 本地 Excel（history_daily/daily_from_*.xlsx）历史底仓
    2) overlay 文件 + akshare 新浪分钟线/分时 合成当日，拼进日线
    3) 本地没有时再走网络：TickPlus -> 腾讯 -> efinance / akshare / baostock
    """
    if BULK_DATA_MODE:
        cached = _load_kline_cache(stock_code, beg, end)
        if cached is not None:
            return cached
        print(f"{stock_code} 批量模式：腾讯/新浪日线")
        df = get_quote_history_tencent(stock_code, beg=beg, end=end)
        if df is None or df.empty:
            df = get_quote_history_sina(stock_code, beg=beg, end=end)
        if df is not None and not df.empty:
            _save_kline_cache(stock_code, beg, end, df)
            latest = _latest_bar_date(df)
            print(f"{stock_code} 最新K线: {latest}")
            return df
        print(f"{stock_code} 批量日线获取失败")
        return None

    if USE_LOCAL_DAILY_FIRST:
        local = load_local_daily_history(stock_code, beg=beg, end=end)
        if local is not None and not local.empty:
            # 手动 overlay 优先；没有再拉新浪当天
            overlay = load_overlay_daily_bars(stock_code)
            if FETCH_TODAY_BAR:
                today_bar = fetch_today_daily_bar(stock_code)
                if today_bar is not None and not today_bar.empty:
                    # 手动 overlay 覆盖自动当天（同日 keep last：先自动后手动）
                    overlay = merge_daily_with_overlay(today_bar, overlay)
            df = merge_daily_with_overlay(local, overlay)
            df = _filter_by_beg_end(df, beg, end)
            latest = _latest_bar_date(df)
            print(f"{stock_code} 使用本地日线(+当天拼凑)，最新K线: {latest}")
            return df
        print(f"{stock_code} 本地Excel无数据，回退网络接口")

    df = get_quote_history_tickplus(stock_code, beg=beg, end=end)
    if df is not None and not df.empty:
        if _accept_daily_bar(df, stock_code, "tickplus"):
            return df
        print(f"{stock_code} tickplus 无合格当日K线，继续尝试其他数据源")

    print(f"{stock_code} 尝试腾讯日线...")
    df = get_quote_history_tencent(stock_code, beg=beg, end=end)
    if df is not None and not df.empty:
        latest = _latest_bar_date(df)
        print(f"{stock_code} 腾讯最新K线: {latest}")
        return df

    # TickPlus 失败且已收盘：先 baostock，避免 efinance 多次 180s 超时
    if _is_after_a_share_close():
        print(f"{stock_code} 尝试 baostock 日线（仅盘后）...")
        try:
            df = get_quote_history_baostock(stock_code, beg=beg, end=end, day_or_week="d")
            if df is not None and not df.empty and _accept_daily_bar(df, stock_code, "baostock"):
                return df
        except Exception as e:
            print(f"{stock_code} baostock 失败: {e}")
        try:
            df = get_stock_data_akshare(stock_code, beg=beg, end=end)
            if df is not None and not df.empty and _accept_daily_bar(df, stock_code, "akshare"):
                return df
        except Exception as e:
            print(f"{stock_code} akshare 失败: {e}")

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
