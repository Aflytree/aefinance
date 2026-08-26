"""命中后基本面二次过滤：业绩亏损、银行、高价股、低成交额、PE。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd

from .constants import (
    AVG_DAILY_AMOUNT_LOOKBACK,
    MAX_HIT_CLOSE_PRICE,
    MAX_HIT_PE,
    MAX_HIT_TURNOVER_PCT,
    MIN_AVG_DAILY_AMOUNT,
    MIN_HIT_CLOSE_PRICE,
    MIN_HIT_TURNOVER_PCT,
)

_earnings_cache: tuple[str, pd.DataFrame] | None = None
_pe_cache: Dict[str, float] | None = None


def _recent_quarter_ends(n: int = 8) -> List[str]:
    today = datetime.now().date()
    m, y = today.month, today.year
    qm = ((m - 1) // 3) * 3
    if qm == 0:
        qm, y = 12, y - 1
    ends: List[str] = []
    for _ in range(n):
        if qm == 12:
            d = datetime(y, 12, 31).date()
        elif qm == 9:
            d = datetime(y, 9, 30).date()
        elif qm == 6:
            d = datetime(y, 6, 30).date()
        else:
            d = datetime(y, 3, 31).date()
        ends.append(d.strftime("%Y%m%d"))
        qm -= 3
        if qm <= 0:
            qm, y = 12, y - 1
    return ends


def load_latest_earnings_table(min_rows: int = 2000) -> Tuple[str, pd.DataFrame]:
    """
    拉取最近一份覆盖较全的东财业绩报表（stock_yjbb_em）。
    返回 (报告期YYYYMMDD, DataFrame)，按股票代码索引友好列。
    """
    global _earnings_cache
    if _earnings_cache is not None:
        return _earnings_cache

    import akshare as ak

    best_date = ""
    best_df: pd.DataFrame | None = None
    for report_date in _recent_quarter_ends():
        try:
            df = ak.stock_yjbb_em(date=report_date)
        except Exception as e:
            print(f"业绩报表 {report_date} 获取失败: {e}", flush=True)
            continue
        if df is None or df.empty:
            continue
        print(f"业绩报表 {report_date}: {len(df)} 条", flush=True)
        if best_df is None or len(df) > len(best_df):
            best_date, best_df = report_date, df
        if len(df) >= min_rows:
            break

    if best_df is None or best_df.empty:
        raise RuntimeError("未能获取可用的业绩报表（stock_yjbb_em）")

    out = best_df.copy()
    out["股票代码"] = out["股票代码"].astype(str).str.zfill(6)
    out["净利润-净利润"] = pd.to_numeric(out["净利润-净利润"], errors="coerce")
    if "所处行业" not in out.columns:
        out["所处行业"] = ""
    out["所处行业"] = out["所处行业"].astype(str)
    _earnings_cache = (best_date, out)
    return _earnings_cache


def _http_get_json(url: str, params: dict) -> dict:
    """优先 curl_cffi（绕过东财对普通 Python TLS 的拦截），失败再回退 requests。"""
    try:
        from curl_cffi import requests as curl_requests

        resp = curl_requests.get(
            url, params=params, timeout=30, impersonate="chrome"
        )
        resp.raise_for_status()
        return resp.json() or {}
    except Exception as e1:
        import requests

        session = requests.Session()
        session.trust_env = False
        try:
            resp = session.get(
                url,
                params=params,
                timeout=30,
                proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
            return resp.json() or {}
        except Exception as e2:
            raise RuntimeError(f"HTTP 获取失败: curl_cffi={e1}; requests={e2}") from e2


def _to_eastmoney_secid(code: str) -> str:
    code = str(code).zfill(6)
    market = "1" if code.startswith("6") else "0"
    return f"{market}.{code}"


def load_pe_ttm_map(codes: List[str] | None = None) -> Dict[str, float]:
    """
    东财动态市盈率 code -> PE。
    传入 codes 时只拉这些股票（命中后过滤推荐）；未传则尝试全市场分页。
    """
    global _pe_cache
    if codes is None and _pe_cache is not None:
        return _pe_cache

    out: Dict[str, float] = {}

    if codes:
        uniq = sorted({str(c).zfill(6) for c in codes if str(c).strip()})
        need = [c for c in uniq if _pe_cache is None or c not in _pe_cache]
        if _pe_cache is not None:
            out.update(_pe_cache)
        if need:
            url = "https://push2.eastmoney.com/api/qt/ulist.np/get"
            batch = 50
            for i in range(0, len(need), batch):
                part = need[i : i + batch]
                secids = ",".join(_to_eastmoney_secid(c) for c in part)
                payload = _http_get_json(
                    url, {"fltt": 2, "fields": "f12,f9", "secids": secids}
                )
                diff = (payload.get("data") or {}).get("diff") or []
                for row in diff:
                    code = str(row.get("f12", "")).zfill(6)
                    try:
                        pe_f = float(row.get("f9"))
                    except (TypeError, ValueError):
                        continue
                    if pe_f == pe_f:
                        out[code] = pe_f
            print(f"动态市盈率加载(按需): 请求{len(need)} 得到{sum(1 for c in need if c in out)}", flush=True)
        _pe_cache = dict(out)
        return {c: out[c] for c in uniq if c in out}

    url = "https://82.push2.eastmoney.com/api/qt/clist/get"
    base_params = {
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f12",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
        "fields": "f12,f9",
        "pz": 100,
    }
    page = 1
    total = None
    while True:
        params = dict(base_params)
        params["pn"] = page
        data = (_http_get_json(url, params).get("data") or {})
        if total is None:
            total = int(data.get("total") or 0)
        diff = data.get("diff") or []
        if not diff:
            break
        for row in diff:
            code = str(row.get("f12", "")).zfill(6)
            try:
                pe_f = float(row.get("f9"))
            except (TypeError, ValueError):
                continue
            if pe_f == pe_f:
                out[code] = pe_f
        if total and len(out) >= total:
            break
        if len(diff) < int(base_params["pz"]):
            break
        page += 1
        if page > 200:
            break

    if not out:
        raise RuntimeError("未能获取动态市盈率（东财）")
    print(f"动态市盈率加载: {len(out)} 只", flush=True)
    _pe_cache = out
    return _pe_cache


def is_bank_stock(name: str, industry: str = "") -> bool:
    text = f"{name or ''}{industry or ''}"
    return "银行" in text


def _fmt_amount_wan(amount: float) -> str:
    return f"{amount / 1e4:.0f}万"


def describe_post_hit_filters(
    *,
    max_close: float | None = None,
    min_close: float | None = None,
    min_avg_amount: float | None = None,
    max_pe: float | None = None,
    min_turnover: float | None = None,
    max_turnover: float | None = None,
    earnings_period: str | None = None,
) -> str:
    price_cap = MAX_HIT_CLOSE_PRICE if max_close is None else max_close
    price_floor = MIN_HIT_CLOSE_PRICE if min_close is None else min_close
    amount_floor = MIN_AVG_DAILY_AMOUNT if min_avg_amount is None else min_avg_amount
    pe_cap = MAX_HIT_PE if max_pe is None else max_pe
    turn_floor = MIN_HIT_TURNOVER_PCT if min_turnover is None else min_turnover
    turn_cap = MAX_HIT_TURNOVER_PCT if max_turnover is None else max_turnover
    period = earnings_period or "最近完整报告期"
    return (
        f"二次过滤: 剔除名称/行业含银行; "
        f"剔除收盘价<{price_floor}或>{price_cap}; "
        f"剔除命中日换手率<{turn_floor}%或>{turn_cap}%; "
        f"剔除近{AVG_DAILY_AMOUNT_LOOKBACK}日日均成交额<{_fmt_amount_wan(amount_floor)}; "
        f"剔除动态PE<=0或PE>{pe_cap}; "
        f"剔除业绩报表({period})净利润<0"
    )


def filter_hits_post(
    hits: List[Dict[str, Any]],
    *,
    max_close: float | None = None,
    min_close: float | None = None,
    min_avg_amount: float | None = None,
    max_pe: float | None = None,
    min_turnover: float | None = None,
    max_turnover: float | None = None,
    drop_loss: bool = True,
    drop_bank: bool = True,
    drop_low_amount: bool = True,
    drop_bad_pe: bool = True,
    drop_bad_turnover: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    对量能命中结果做二次过滤。
    返回 (保留列表, 统计信息)。
    """
    price_cap = MAX_HIT_CLOSE_PRICE if max_close is None else max_close
    price_floor = MIN_HIT_CLOSE_PRICE if min_close is None else min_close
    amount_floor = MIN_AVG_DAILY_AMOUNT if min_avg_amount is None else min_avg_amount
    pe_cap = MAX_HIT_PE if max_pe is None else max_pe
    turn_floor = MIN_HIT_TURNOVER_PCT if min_turnover is None else min_turnover
    turn_cap = MAX_HIT_TURNOVER_PCT if max_turnover is None else max_turnover
    stats: Dict[str, Any] = {
        "before": len(hits),
        "drop_bank": 0,
        "drop_price_high": 0,
        "drop_price_low": 0,
        "drop_turnover_low": 0,
        "drop_turnover_high": 0,
        "drop_turnover_missing": 0,
        "drop_loss": 0,
        "drop_low_amount": 0,
        "drop_pe": 0,
        "drop_no_earnings": 0,
        "after": 0,
        "earnings_period": "",
        "min_avg_amount": amount_floor,
        "max_pe": pe_cap,
        "min_close": price_floor,
        "max_close": price_cap,
        "min_turnover": turn_floor,
        "max_turnover": turn_cap,
    }
    if not hits:
        return [], stats

    earnings_period = ""
    earnings_by_code: Dict[str, pd.Series] = {}
    if drop_loss or drop_bank:
        try:
            earnings_period, earnings_df = load_latest_earnings_table()
            stats["earnings_period"] = earnings_period
            for _, row in earnings_df.iterrows():
                earnings_by_code[str(row["股票代码"]).zfill(6)] = row
        except Exception as e:
            print(f"警告: 业绩报表加载失败，跳过亏损过滤: {e}", flush=True)
            drop_loss = False

    pe_by_code: Dict[str, float] = {}
    if drop_bad_pe:
        try:
            hit_codes = [str(h.get("股票代码", "")).zfill(6) for h in hits]
            pe_by_code = load_pe_ttm_map(hit_codes)
        except Exception as e:
            print(f"警告: 市盈率加载失败，跳过 PE 过滤: {e}", flush=True)
            drop_bad_pe = False

    kept: List[Dict[str, Any]] = []
    for hit in hits:
        code = str(hit.get("股票代码", "")).zfill(6)
        name = str(hit.get("股票名称", ""))
        close = hit.get("收盘")
        try:
            close_f = float(close)
        except (TypeError, ValueError):
            close_f = float("nan")

        industry = ""
        profit = None
        row = earnings_by_code.get(code)
        hit = dict(hit)
        if row is not None:
            industry = str(row.get("所处行业", "") or "")
            profit = row.get("净利润-净利润")
            hit["所处行业"] = industry
            hit["净利润"] = profit
            hit["业绩报告期"] = earnings_period

        pe = pe_by_code.get(code)
        if pe is not None:
            hit["PE"] = pe

        if drop_bank and is_bank_stock(name, industry):
            stats["drop_bank"] += 1
            continue
        if close_f == close_f:  # not NaN
            if close_f > price_cap:
                stats["drop_price_high"] += 1
                continue
            if close_f < price_floor:
                stats["drop_price_low"] += 1
                continue
        if drop_bad_turnover:
            try:
                turn_f = float(hit.get("换手率"))
            except (TypeError, ValueError):
                turn_f = float("nan")
            if turn_f != turn_f:  # NaN / 缺失
                stats["drop_turnover_missing"] += 1
                continue
            if turn_f < turn_floor:
                stats["drop_turnover_low"] += 1
                continue
            if turn_f > turn_cap:
                stats["drop_turnover_high"] += 1
                continue
        if drop_low_amount:
            avg_amt = hit.get("日均成交额")
            try:
                avg_amt_f = float(avg_amt)
            except (TypeError, ValueError):
                avg_amt_f = float("nan")
            if avg_amt_f == avg_amt_f and avg_amt_f < amount_floor:
                stats["drop_low_amount"] += 1
                continue
        if drop_bad_pe:
            # 无PE / PE<=0（亏损等）/ PE过高 -> 剔除
            if pe is None or pe <= 0 or pe > pe_cap:
                stats["drop_pe"] += 1
                continue
        if drop_loss and row is not None and profit is not None:
            try:
                if not pd.isna(profit) and float(profit) < 0:
                    stats["drop_loss"] += 1
                    continue
            except (TypeError, ValueError):
                pass
        kept.append(hit)

    stats["after"] = len(kept)
    return kept, stats


def format_post_filter_stats(stats: Dict[str, Any]) -> str:
    amount_floor = float(stats.get("min_avg_amount", MIN_AVG_DAILY_AMOUNT))
    pe_cap = float(stats.get("max_pe", MAX_HIT_PE))
    price_floor = float(stats.get("min_close", MIN_HIT_CLOSE_PRICE))
    price_cap = float(stats.get("max_close", MAX_HIT_CLOSE_PRICE))
    turn_floor = float(stats.get("min_turnover", MIN_HIT_TURNOVER_PCT))
    turn_cap = float(stats.get("max_turnover", MAX_HIT_TURNOVER_PCT))
    # 兼容旧字段 drop_price
    drop_high = stats.get("drop_price_high", stats.get("drop_price", 0))
    drop_low = stats.get("drop_price_low", 0)
    drop_turn = (
        int(stats.get("drop_turnover_low", 0))
        + int(stats.get("drop_turnover_high", 0))
        + int(stats.get("drop_turnover_missing", 0))
    )
    return (
        f"二次过滤: {stats.get('before', 0)} -> {stats.get('after', 0)} "
        f"(银行-{stats.get('drop_bank', 0)}, "
        f"高价>{price_cap}-{drop_high}, "
        f"低价<{price_floor}-{drop_low}, "
        f"换手率不在[{turn_floor},{turn_cap}]%-{drop_turn}, "
        f"日均额<{_fmt_amount_wan(amount_floor)}-{stats.get('drop_low_amount', 0)}, "
        f"PE<=0或>{pe_cap}-{stats.get('drop_pe', 0)}, "
        f"亏损-{stats.get('drop_loss', 0)}; "
        f"业绩期={stats.get('earnings_period') or 'N/A'})"
    )
