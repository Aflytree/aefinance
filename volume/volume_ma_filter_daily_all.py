import argparse
import os
import time
import traceback
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import akshare as ak
import baostock as bs
import pandas as pd
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_vol_dir = Path(__file__).resolve().parent
for _p in (_vol_dir, _root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import (
    DEFAULT_HISTORY_YEARS,
    DEFAULT_PRE20_MAX_PCT,
    DEFAULT_THRESHOLD,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    SIGNAL_DAY_BAOSTOCK_RETRIES,
    SIGNAL_DAY_BAOSTOCK_RETRY_SECONDS,
    build_summary_rows,
    check_hit_at_row,
    dedupe_hits_first_within_days,
    describe_signal_dedupe_rule,
    export_forward_result,
    format_filter_conditions_line,
    format_forward_summary,
    fetch_stock_df,
    filter_hits_post,
    format_post_filter_stats,
    get_signal_forward_returns,
    normalize_stock_code,
    prepare_ohlcv_df,
)

# 全池扫描：定期重连 Baostock，单股拉数失败时重试
BAOSTOCK_RELOGIN_EVERY = 200
FETCH_MAX_RETRIES = 2
SCAN_PROGRESS_EVERY = 100
_LOCK_FILE = _root / ".volume_ma_filter_daily_all.lock"


def acquire_run_lock(force: bool = False) -> bool:
    """避免多个进程同时占用 Baostock 导致第二实例秒退/无输出。"""
    if _LOCK_FILE.is_file():
        old = _LOCK_FILE.read_text(encoding="utf-8").strip()
        if not force:
            print(
                f"已有扫描任务在运行（锁文件 PID={old}）。"
                f"请先结束该进程，或加 --force 强制启动。\n"
                f"查进程: Get-Process -Id {old} -ErrorAction SilentlyContinue",
                flush=True,
            )
            return False
        print(f"警告: --force 覆盖运行锁（原 PID={old}）", flush=True)
    _LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
    return True


def release_run_lock() -> None:
    try:
        if _LOCK_FILE.is_file() and _LOCK_FILE.read_text(encoding="utf-8").strip() == str(
            os.getpid()
        ):
            _LOCK_FILE.unlink()
    except OSError:
        pass


def get_mainboard_stock_pool() -> pd.DataFrame:
    """主板股票池。优先全 A 列表；失败时回退沪/深交易所接口（避开北交所解析失败）。"""
    df: pd.DataFrame | None = None
    try:
        raw = ak.stock_info_a_code_name()
        df = raw.rename(columns={"code": "代码", "name": "名称"})
    except Exception as e:
        print(f"stock_info_a_code_name 失败，回退沪深接口: {e}", flush=True)
        sh = ak.stock_info_sh_name_code()
        sz = ak.stock_info_sz_name_code()
        sh_part = sh[["证券代码", "证券简称"]].rename(
            columns={"证券代码": "代码", "证券简称": "名称"}
        )
        sz_part = sz[["A股代码", "A股简称"]].rename(
            columns={"A股代码": "代码", "A股简称": "名称"}
        )
        df = pd.concat([sh_part, sz_part], ignore_index=True)

    assert df is not None
    df["代码"] = df["代码"].astype(str).str.zfill(6)

    # 两市主板：沪市主板(600/601/603/605)、深市主板(000/001/002/003)
    mainboard_prefix = ("600", "601", "603", "605", "000", "001", "002", "003")
    # mainboard_prefix = ("600")
    df = df[df["代码"].str.startswith(mainboard_prefix)].copy()

    # 剔除 ST、*ST、退市风险股票；创业板(300)已通过前缀排除
    df = df[~df["名称"].str.contains(r"ST|\*ST|退", case=False, regex=True, na=False)]
    # 银行股不做量能扫描（名称含「银行」）
    df = df[~df["名称"].str.contains("银行", na=False)]
    return df.reset_index(drop=True)


def _prepare_ohlcv_df_with_date(df: pd.DataFrame) -> pd.DataFrame | None:
    """全池扫描用：在统一 OHLCV 准备基础上附加日历日期列。"""
    data = prepare_ohlcv_df(df)
    if data is None:
        return None
    data = data.copy()
    data["_日期"] = pd.to_datetime(data["日期"], errors="coerce").dt.date
    return data


def _index_last_bar_date() -> date | None:
    """000001 在 Baostock 中最后一根已入库日 K 的日期。"""
    raw = fetch_stock_df(
        "000001",
        baostock_manage_login=False,
        baostock_verbose=False,
    )
    data = _prepare_ohlcv_df_with_date(raw)
    if data is None or data.empty:
        return None
    return data["_日期"].iloc[-1]


def _is_after_market_close(ref: datetime) -> bool:
    return (ref.hour, ref.minute) >= (MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)


def get_market_latest_trading_day(
    ref: datetime | None = None,
    *,
    signal_date: date | None = None,
) -> date:
    """
    确定全市场信号交易日。

    - 显式 signal_date：直接使用（用于 --signal-date）。
    - 交易日 15:00 后跑批：以「日历当日」为目标，等待 000001 日 K 更新到当日后再返回；
      避免收盘后仍落在上一交易日。
    - 盘中 / 周末 / 节假日：用 000001 最新 K 线日期（最近已收盘交易日）。
    """
    if signal_date is not None:
        return signal_date

    ref = ref or datetime.now()
    calendar_today = ref.date()
    last_bar = _index_last_bar_date()
    if last_bar is None:
        return calendar_today

    use_today_after_close = _is_after_market_close(ref) and calendar_today.weekday() < 5
    if not use_today_after_close:
        return last_bar

    target = calendar_today
    if last_bar >= target:
        return target

    print(
        f"收盘后跑批：目标信号日={target}，000001 日K 最新={last_bar}，"
        f"等待 Baostock 更新（最多 {SIGNAL_DAY_BAOSTOCK_RETRIES} 次，"
        f"间隔 {SIGNAL_DAY_BAOSTOCK_RETRY_SECONDS}s）...",
        flush=True,
    )
    for attempt in range(1, SIGNAL_DAY_BAOSTOCK_RETRIES + 1):
        time.sleep(SIGNAL_DAY_BAOSTOCK_RETRY_SECONDS)
        last_bar = _index_last_bar_date()
        if last_bar is None:
            continue
        print(
            f"  重试 {attempt}/{SIGNAL_DAY_BAOSTOCK_RETRIES}: 000001 最新日K={last_bar}",
            flush=True,
        )
        if last_bar >= target:
            print(f"信号日已就绪: {target}", flush=True)
            return target

    print(
        f"错误: 已收盘但 000001 日K 仍为 {last_bar}，未到 {target}。\n"
        f"请稍后再跑（通常 15:30–17:00 后更新），或加 --signal-date {target.strftime('%Y%m%d')} "
        f"在确认 Baostock 已有当日数据后指定。",
        flush=True,
    )
    raise SystemExit(1)


def check_hit_on_signal_day(
    data: pd.DataFrame,
    signal_day: date,
    threshold: float,
    pre20_max_pct: float,
) -> Dict[str, Any] | None:
    """指定信号交易日是否命中（该股当日无 K 线则视为未命中）。"""
    rows = data.index[data["_日期"] == signal_day]
    if len(rows) == 0:
        return None
    return check_hit_at_row(data, int(rows[-1]), threshold, pre20_max_pct)


def is_hit_today(df: pd.DataFrame, threshold: float, pre20_max_pct: float) -> Dict[str, Any] | None:
    """信号交易日是否命中（当日非交易日则用最近交易日）。"""
    data = _prepare_ohlcv_df_with_date(df)
    if data is None:
        return None
    signal_day = get_market_latest_trading_day()
    return check_hit_on_signal_day(data, signal_day, threshold, pre20_max_pct)


def _week_monday(d: date) -> date:
    """给定日期所在自然周的周一（ISO，周一=0）。"""
    return d - timedelta(days=d.weekday())


def _week_friday(d: date) -> date:
    """给定日期所在自然周的周五。"""
    return _week_monday(d) + timedelta(days=4)


def get_current_period_range(ref: datetime | None = None) -> Tuple[date, date]:
    """本周一至本周五（含）；若今日未到周五则截至今日。"""
    ref = ref or datetime.now()
    d = ref.date()
    start = _week_monday(d)
    end = min(d, _week_friday(d))
    return start, end


def get_previous_period_range(ref: datetime | None = None) -> Tuple[date, date]:
    """上周一至上周五（含）。"""
    ref = ref or datetime.now()
    this_monday = _week_monday(ref.date())
    end = this_monday - timedelta(days=3)
    start = end - timedelta(days=4)
    return start, end


def get_recent_calendar_days_range(
    days: int,
    *,
    end_day: date | None = None,
) -> Tuple[date, date]:
    """
    以 end_day（默认信号日/最近交易日）为终点，向前覆盖 days 个自然日的闭区间。
    例：end=07-31、days=3 → 07-29 ~ 07-31。
    """
    if days < 1:
        raise ValueError("days 必须 >= 1")
    end = end_day or datetime.now().date()
    start = end - timedelta(days=days - 1)
    return start, end


def get_week_before_days_ago(
    days_ago: int,
    *,
    anchor_day: date | None = None,
) -> Tuple[date, date]:
    """
    以 anchor_day（默认今天）往前推 days_ago 天为终点，再倒推 7 个自然日（含终点）。
    例：anchor=08-01、days_ago=3 → 终点 07-29 → 区间 07-23 ~ 07-29。
    """
    if days_ago < 0:
        raise ValueError("days_ago 必须 >= 0")
    end = (anchor_day or datetime.now().date()) - timedelta(days=days_ago)
    start = end - timedelta(days=6)
    return start, end


def scan_last_week_hits(
    pool_df: pd.DataFrame,
    threshold: float,
    pre20_max_pct: float,
    *,
    baostock_manage_login: bool = False,
) -> Tuple[List[Dict[str, Any]], date, date]:
    """扫描池内每只股票在上一交易周（周一至周五）内的全部命中。"""
    week_start, week_end = get_previous_period_range()
    hits: List[Dict[str, Any]] = []
    n = len(pool_df)
    for k, (_, row) in enumerate(pool_df.iterrows()):
        if k > 0 and k % 500 == 0:
            print(f"上周扫描进度: {k}/{n}", flush=True)
        code = normalize_stock_code(str(row["代码"]))
        name = str(row["名称"])
        raw = fetch_stock_df(
            code, baostock_manage_login=baostock_manage_login, baostock_verbose=False
        )
        data = _prepare_ohlcv_df_with_date(raw)
        if data is None:
            continue
        in_week = data[(data["_日期"] >= week_start) & (data["_日期"] <= week_end)]
        for idx in in_week.index:
            hit = check_hit_at_row(data, int(idx), threshold, pre20_max_pct)
            if hit is None:
                continue
            hit["股票代码"] = code
            hit["股票名称"] = name
            hits.append(hit)
    hits.sort(key=lambda x: (x["日期"], x["股票代码"]))
    return dedupe_hits_first_within_days(hits), week_start, week_end


def build_email_body(
    hits: List[Dict[str, Any]],
    threshold: float,
    pre20_max_pct: float,
    total_count: int,
    *,
    signal_day: date | None = None,
    post_filter_note: str = "",
) -> str:
    signal_day = signal_day or datetime.now().date()
    lines = [
        f"{signal_day} 日终量能筛选（信号交易日；全市场主板，剔除ST，排除创业板；逻辑同 volume_ma_filter_detail）",
        f"扫描范围: {total_count} 只",
        format_filter_conditions_line(
            threshold=threshold, pre20_max_pct=pre20_max_pct
        ),
    ]
    if post_filter_note:
        lines.append(post_filter_note)
    lines.extend(
        [
            f"命中数量: {len(hits)}",
            "",
        ]
    )

    for item in hits:
        extra = ""
        try:
            turn = item.get("换手率")
            if turn is not None and float(turn) == float(turn):
                extra += f" | 换手率:{float(turn):.2f}%"
        except (TypeError, ValueError):
            pass
        avg_amt = item.get("日均成交额")
        try:
            if avg_amt is not None and float(avg_amt) == float(avg_amt):
                extra += f" | 日均额:{float(avg_amt) / 1e4:.0f}万"
        except (TypeError, ValueError):
            pass
        pe = item.get("PE")
        try:
            if pe is not None and float(pe) == float(pe):
                extra += f" | PE:{float(pe):.1f}"
        except (TypeError, ValueError):
            pass
        if item.get("净利润") is not None and not (
            isinstance(item.get("净利润"), float) and pd.isna(item.get("净利润"))
        ):
            try:
                extra += f" | 净利润:{float(item['净利润'])/1e8:.2f}亿"
            except (TypeError, ValueError):
                pass
            if item.get("所处行业"):
                extra += f" | 行业:{item['所处行业']}"
        lines.append(
            f"{item['股票代码']} {item['股票名称']} | 日期:{item['日期']} | MA5/MA10:{item['MA5/MA10']:.3f} | "
            f"收盘/价格MA10:{item['收盘']:.2f}/{item['价格MA10']:.2f} | "
            f"当天涨跌幅:{item['当天涨跌幅%']:.2f}% | 前20日涨跌幅:{item['信号日前20日涨跌幅%']:.2f}%"
            f"{extra}"
        )
    return "\n".join(lines)


def build_week_hits_email_body(
    hits: List[Dict[str, Any]],
    threshold: float,
    pre20_max_pct: float,
    total_count: int,
    week_start: date,
    week_end: date,
    *,
    period_label: str = "上周",
    post_filter_note: str = "",
) -> str:
    lines = [
        f"{period_label}量能命中（{week_start} ~ {week_end}）",
        f"扫描范围: {total_count} 只（全市场主板，剔除ST）",
        format_filter_conditions_line(
            threshold=threshold,
            pre20_max_pct=pre20_max_pct,
            label="条件同当日筛",
        ),
    ]
    if post_filter_note:
        lines.append(post_filter_note)
    lines.extend(
        [
            f"命中数量: {len(hits)}（{describe_signal_dedupe_rule()}）",
            "",
        ]
    )
    if not hits:
        lines.append(f"{period_label}无命中记录。")
        return "\n".join(lines)
    for item in hits:
        extra = _format_ma_extra(item)
        try:
            turn = item.get("换手率")
            if turn is not None and float(turn) == float(turn):
                extra = f" | 换手率:{float(turn):.2f}%" + extra
        except (TypeError, ValueError):
            pass
        lines.append(
            f"{item['股票代码']} {item['股票名称']} | 命中日期:{item['日期']} | "
            f"MA5/MA10:{item['MA5/MA10']:.3f} | 收盘/价格MA10:{item['收盘']:.2f}/{item['价格MA10']:.2f} | "
            f"当天涨跌幅:{item['当天涨跌幅%']:.2f}% | 前20日涨跌幅:{item['信号日前20日涨跌幅%']:.2f}%"
            f"{extra}"
        )
    return "\n".join(lines)


def _ma_val(hit: Dict[str, Any], key: str) -> float:
    try:
        v = float(hit.get(key))
        return v if v == v else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _format_ma_extra(hit: Dict[str, Any]) -> str:
    ma120 = _ma_val(hit, "价格MA120")
    ma250 = _ma_val(hit, "价格MA250")
    close = _ma_val(hit, "收盘")
    if close != close:
        return ""
    parts = []
    if ma120 == ma120 and ma250 == ma250:
        parts.append(f"收盘:{close:.2f}/半年线:{ma120:.2f}/年线:{ma250:.2f}")
    tags = hit.get("均线位置标签")
    if tags:
        parts.append(str(tags))
    return (" | " + " | ".join(parts)) if parts else ""


def annotate_hit_ma_position(hit: Dict[str, Any]) -> Dict[str, Any]:
    """标注命中日收盘相对半年线/年线位置。"""
    out = dict(hit)
    close = _ma_val(out, "收盘")
    ma120 = _ma_val(out, "价格MA120")
    ma250 = _ma_val(out, "价格MA250")
    tags = []
    if ma250 == ma250 and close > ma250:
        tags.append("年线上方")
    if ma120 == ma120 and close > ma120:
        tags.append("半年线上方")
    out["均线位置标签"] = "+".join(tags) if tags else ""
    out["站上年线或半年线"] = bool(tags)
    return out


def filter_hits_above_ma120_or_ma250(
    hits: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """保留命中日收盘价 > MA120 或 > MA250 的记录。"""
    annotated = [annotate_hit_ma_position(h) for h in hits]
    kept = [h for h in annotated if h.get("站上年线或半年线")]
    stats = {
        "before": len(hits),
        "after": len(kept),
        "drop_below_ma": len(hits) - len(kept),
    }
    return kept, stats


def format_above_ma_filter_stats(stats: Dict[str, Any]) -> str:
    return (
        f"均线过滤(收盘>半年线或年线): {stats.get('before', 0)} -> {stats.get('after', 0)} "
        f"(低于均线-{stats.get('drop_below_ma', 0)})"
    )


def build_above_ma_report_body(
    hits: List[Dict[str, Any]],
    *,
    title: str,
    source_note: str,
    ma_filter_note: str,
) -> str:
    lines = [
        title,
        source_note,
        "附加条件: 命中日收盘价 > 价格MA120（半年线） 或 > 价格MA250（年线）",
        ma_filter_note,
        f"命中数量: {len(hits)}",
        "",
    ]
    if not hits:
        lines.append("无符合均线条件的命中。")
        return "\n".join(lines)
    for item in hits:
        extra = _format_ma_extra(item)
        try:
            turn = item.get("换手率")
            if turn is not None and float(turn) == float(turn):
                extra = f" | 换手率:{float(turn):.2f}%" + extra
        except (TypeError, ValueError):
            pass
        lines.append(
            f"{item['股票代码']} {item.get('股票名称', '')} | 命中日期:{item['日期']} | "
            f"MA5/MA10:{item['MA5/MA10']:.3f} | 收盘/价格MA10:{item['收盘']:.2f}/{item['价格MA10']:.2f} | "
            f"当天涨跌幅:{item['当天涨跌幅%']:.2f}% | 前20日涨跌幅:{item['信号日前20日涨跌幅%']:.2f}%"
            f"{extra}"
        )
    return "\n".join(lines)


def baostock_relogin(verbose: bool = False) -> bool:
    try:
        bs.logout()
    except Exception:
        pass
    lg = bs.login()
    if lg.error_code != "0":
        print(f"Baostock 登录失败: {lg.error_msg}", flush=True)
        return False
    if verbose:
        print("Baostock 重新登录成功", flush=True)
    return True


def fetch_stock_df_retry(
    stock_code: str,
    *,
    baostock_manage_login: bool = False,
    max_retries: int = FETCH_MAX_RETRIES,
) -> pd.DataFrame | None:
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = fetch_stock_df(
                stock_code,
                baostock_manage_login=baostock_manage_login,
                baostock_verbose=False,
            )
            if raw is not None and not raw.empty:
                return raw
        except Exception as e:
            last_err = e
        if attempt < max_retries:
            baostock_relogin(verbose=False)
    if last_err is not None:
        print(f"{stock_code} 拉数失败: {last_err}", flush=True)
    return None


def send_result_email(body: str, attachment_paths: Optional[List[Path]] = None) -> None:
    from efi_email import send as send_email

    paths = [p for p in (attachment_paths or []) if p.is_file()]
    send_email(body, attachments=paths if paths else None)


def send_reports_email(
    date_tag: Optional[str] = None,
    *,
    skip_forward_report: bool = False,
) -> None:
    """根据已生成的本地 txt 报告发送邮件（扫描跑完后可单独调用）。"""
    tag = date_tag or datetime.now().strftime("%Y%m%d")
    report_daily = _root / f"volume_ma_filter_daily_all_{tag}.txt"
    report_last_week = _root / f"volume_ma_filter_daily_all_lastweek_{tag}.txt"
    report_3y = _root / f"volume_ma_filter_daily_all_3y_{tag}.txt"
    signal_hits_reports = sorted(
        _root.glob(f"volume_ma_filter_daily_all_signal_hits_*y_{tag}.txt")
    )
    report_signal_hits_fwd = (
        signal_hits_reports[-1] if signal_hits_reports else None
    )

    if not report_daily.is_file():
        print(f"未找到当日报告，无法发信: {report_daily}", flush=True)
        return

    body = report_daily.read_text(encoding="utf-8")
    email_parts: List[str] = [body]
    attach: List[Path] = [report_daily]

    if report_last_week.is_file():
        email_parts.append("\n\n---\n" + report_last_week.read_text(encoding="utf-8"))
        attach.append(report_last_week)

    if report_signal_hits_fwd is not None and report_signal_hits_fwd.is_file():
        email_parts.append(
            "\n\n---\n"
            + report_signal_hits_fwd.read_text(encoding="utf-8")
        )
        attach.append(report_signal_hits_fwd)

    if report_3y.is_file():
        email_parts.append(
            "\n\n---\n"
            + f"报告1（当日命中）: {report_daily}\n"
            + f"报告2（全池近{DEFAULT_HISTORY_YEARS}年前瞻表格）: {report_3y}\n"
        )
        attach.append(report_3y)
    else:
        email_parts.append(
            "\n\n---\n"
            + f"报告1（当日命中）: {report_daily}\n"
            + (
                f"报告2: 未生成（本次使用了 --skip-forward-report 或导出失败）\n"
                if skip_forward_report
                else f"报告2（全池近{DEFAULT_HISTORY_YEARS}年前瞻表格）: 文件不存在\n"
            )
        )

    email_parts.append(f"\n（附件共 {len(attach)} 个 txt）\n")
    print(f"正在发送邮件（附件 {len(attach)} 个）...", flush=True)
    send_result_email("".join(email_parts), attachment_paths=attach)
    print("邮件已发送。", flush=True)


def export_signal_day_hits_forward_report(
    hits: List[Dict[str, Any]],
    threshold: float,
    pre20_max_pct: float,
    history_years: int,
    output_path: Path,
    *,
    signal_day: date | None = None,
) -> Tuple[Optional[Path], List[str]]:
    """
    对信号日命中股票做与 volume_ma_filter_detail 一致的前瞻统计（近 history_years 年），
    写入 export_forward_result 版式 txt，并返回各股文字摘要供邮件正文使用。
    """
    signal_day = signal_day or datetime.now().date()
    header = (
        f"=== 信号日（{signal_day}）命中股票 — 近{history_years}年前瞻 ===\n"
        f"命中股数: {len({normalize_stock_code(h['股票代码']) for h in hits})}\n"
    )
    if not hits:
        output_path.write_text(
            header + "\n无当日命中股票，无前瞻明细。\n",
            encoding="utf-8",
        )
        return output_path, []

    codes: List[str] = []
    seen: set[str] = set()
    for h in hits:
        code = normalize_stock_code(h["股票代码"])
        if code not in seen:
            seen.add(code)
            codes.append(code)
    codes.sort()

    lg = bs.login()
    if lg.error_code != "0":
        print(f"信号日命中股前瞻 Baostock 登录失败: {lg.error_msg}", flush=True)
        return None, []

    all_detail: List[pd.DataFrame] = []
    all_summary_rows: List[Dict[str, Any]] = []
    summary_lines: List[str] = []
    ok = 0
    fail = 0
    try:
        for k, code in enumerate(codes):
            if k > 0 and k % 50 == 0:
                baostock_relogin(verbose=False)
            print(f"信号日命中股前瞻: {k + 1}/{len(codes)} {code}", flush=True)
            detail_df = pd.DataFrame()
            for attempt in range(FETCH_MAX_RETRIES + 1):
                try:
                    detail_df = get_signal_forward_returns(
                        code,
                        threshold,
                        history_years,
                        pre20_max_pct,
                        baostock_manage_login=False,
                        baostock_verbose=False,
                    )
                    break
                except Exception as e:
                    if attempt < FETCH_MAX_RETRIES:
                        baostock_relogin(verbose=False)
                    else:
                        print(f"信号日命中股前瞻 {code} 失败: {e}", flush=True)
            if detail_df.empty:
                fail += 1
                summary_lines.append(f"{code} 最近{history_years}年无可用信号数据。")
                continue
            ok += 1
            all_detail.append(detail_df)
            all_summary_rows.extend(
                build_summary_rows(detail_df, code, threshold, history_years)
            )
            summary_lines.append(
                format_forward_summary(detail_df, code, threshold, history_years)
            )
    finally:
        bs.logout()

    print(
        f"信号日命中股前瞻统计: 有样本 {ok} 只, 无样本或失败 {fail} 只",
        flush=True,
    )

    if not all_detail:
        output_path.write_text(
            header + "\n=== 前瞻统计汇总 ===\n无汇总数据\n\n=== 前瞻统计明细 ===\n无明细数据\n",
            encoding="utf-8",
        )
        return output_path, summary_lines

    merged_df = pd.concat(all_detail, ignore_index=True)
    summary_df = pd.DataFrame(all_summary_rows)
    path = Path(
        export_forward_result(
            merged_df, summary_df, str(output_path), strategy_header=header
        )
    )
    return path, summary_lines


def export_full_pool_forward_3y_report(
    pool_df: pd.DataFrame,
    threshold: float,
    pre20_max_pct: float,
    history_years: int,
    output_path: Path,
) -> Optional[Path]:
    """
    对主板股票池内每一只做与 volume_ma_filter_detail 一致的前瞻统计（近 history_years 年），
    合并写入与 export_forward_result 相同版式的 txt（与报告1是否命中无关）。
    股票池定义同 get_mainboard_stock_pool：沪市 600/601/603/605、深市 000/001/002/003，剔除 ST。
    """
    if pool_df is None or pool_df.empty:
        output_path.write_text(
            "=== 前瞻统计汇总 ===\n无汇总数据（股票池为空）\n\n"
            "=== 前瞻统计明细 ===\n无明细数据\n",
            encoding="utf-8",
        )
        return output_path

    lg = bs.login()
    if lg.error_code != "0":
        print(f"报告2 Baostock 登录失败: {lg.error_msg}")
        return None

    all_detail: List[pd.DataFrame] = []
    all_summary_rows: List[Dict[str, Any]] = []
    n = len(pool_df)
    forward_ok = 0
    forward_fail = 0
    try:
        for k, (_, row) in enumerate(pool_df.iterrows()):
            code = normalize_stock_code(str(row["代码"]))
            if k > 0 and k % BAOSTOCK_RELOGIN_EVERY == 0:
                baostock_relogin(verbose=False)
            if k == 0 or (k + 1) % 500 == 0:
                print(f"报告2 前瞻进度: {k + 1}/{n} {code}", flush=True)
            detail_df = pd.DataFrame()
            for attempt in range(FETCH_MAX_RETRIES + 1):
                try:
                    detail_df = get_signal_forward_returns(
                        code,
                        threshold,
                        history_years,
                        pre20_max_pct,
                        baostock_manage_login=False,
                        baostock_verbose=False,
                    )
                    break
                except Exception as e:
                    if attempt < FETCH_MAX_RETRIES:
                        baostock_relogin(verbose=False)
                    else:
                        print(f"报告2 {code} 失败: {e}", flush=True)
            if detail_df.empty:
                forward_fail += 1
                continue
            forward_ok += 1
            all_detail.append(detail_df)
            all_summary_rows.extend(
                build_summary_rows(detail_df, code, threshold, history_years)
            )
    finally:
        bs.logout()
    print(
        f"报告2 拉数统计: 有信号样本 {forward_ok} 只, 无样本或失败 {forward_fail} 只",
        flush=True,
    )

    if not all_detail:
        output_path.write_text(
            "=== 前瞻统计汇总 ===\n无汇总数据（全池股票在近年内均无符合条件的信号样本）\n\n"
            "=== 前瞻统计明细 ===\n无明细数据\n",
            encoding="utf-8",
        )
        return output_path

    merged_df = pd.concat(all_detail, ignore_index=True)
    summary_df = pd.DataFrame(all_summary_rows)
    return Path(export_forward_result(merged_df, summary_df, str(output_path)))


def main() -> None:
    parser = argparse.ArgumentParser(description="日终量能筛选（全市场主板）")
    parser.add_argument(
        "--skip-forward-report",
        action="store_true",
        help="跳过全池近年前瞻报告（报告2），仅生成当日命中并发送邮件，大幅缩短耗时",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="仅扫描股票池中前 N 只（当前表格顺序），用于快速测试；默认扫描全池",
    )
    parser.add_argument(
        "--last-week-hits",
        action="store_true",
        help="额外扫描上周（周一至周五）内的命中",
    )
    parser.add_argument(
        "--this-week-hits",
        action="store_true",
        help="额外扫描本周（周一至周五，未到周五则截至今日）内的命中",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=None,
        metavar="N",
        help="额外扫描以信号日为终点的近 N 个自然日内命中（可与 --last-week-hits 同用）",
    )
    parser.add_argument(
        "--week-before-days-ago",
        type=int,
        default=None,
        metavar="N",
        help="以日历今日往前 N 天为终点，倒推 7 个自然日的命中（含终点日）",
    )
    parser.add_argument(
        "--above-ma120-or-ma250",
        action="store_true",
        help="额外输出/附带：命中日收盘价在半年线(MA120)或年线(MA250)上方的子集报告",
    )
    parser.add_argument(
        "--no-send-email",
        action="store_true",
        help="不发送邮件，仅写本地报告",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="若已有实例在跑，仍强制启动（会抢占 Baostock，不推荐）",
    )
    parser.add_argument(
        "--send-email-only",
        action="store_true",
        help="不扫描，仅根据已生成的报告 txt 发送邮件（可配合 --date）",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        metavar="YYYYMMDD",
        help="报告日期标签，默认今天；用于 --send-email-only",
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=DEFAULT_HISTORY_YEARS,
        help=f"报告2全池前瞻统计使用的历史年数，默认{DEFAULT_HISTORY_YEARS}",
    )
    parser.add_argument(
        "--signal-date",
        type=str,
        default=None,
        metavar="YYYYMMDD",
        help="强制信号交易日（默认：收盘后用日历当日，并等待 000001 日K 更新到该日）",
    )
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    if args.send_email_only:
        send_reports_email(
            args.date,
            skip_forward_report=args.skip_forward_report,
        )
        return

    if not acquire_run_lock(force=args.force):
        return

    try:
        _main_body(args)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        release_run_lock()


def _main_body(args: argparse.Namespace) -> None:
    threshold = DEFAULT_THRESHOLD
    pre20_max_pct = DEFAULT_PRE20_MAX_PCT

    pool = get_mainboard_stock_pool()
    if args.limit is not None:
        pool = pool.head(int(args.limit)).copy()
        print(f"已启用 --limit {args.limit}，实际扫描 {len(pool)} 只", flush=True)
    print(f"主板股票池数量(剔除ST/创业板/银行): {len(pool)}", flush=True)

    lg = bs.login()
    if lg.error_code != "0":
        print(f"Baostock 登录失败: {lg.error_msg}")
        return

    signal_date_override: date | None = None
    if args.signal_date:
        signal_date_override = datetime.strptime(args.signal_date.strip(), "%Y%m%d").date()

    signal_day = get_market_latest_trading_day(signal_date=signal_date_override)
    calendar_today = datetime.now().date()
    if signal_day == calendar_today:
        print(
            f"信号交易日: {signal_day}（收盘后日终，使用当日已入库日 K）",
            flush=True,
        )
    else:
        print(
            f"信号交易日: {signal_day}（日历今日 {calendar_today} 非交易日或盘中，"
            f"使用 000001 最近已收盘日 K）",
            flush=True,
        )

    hits: List[Dict[str, Any]] = []
    # 额外区间：(标签, 起始, 结束, 报告文件后缀)
    extra_periods: List[Tuple[str, date, date, str]] = []
    if args.this_week_hits and args.last_week_hits:
        print("请只指定 --this-week-hits 或 --last-week-hits 之一。", flush=True)
        return
    if args.recent_days is not None:
        if int(args.recent_days) < 1:
            print("--recent-days 必须 >= 1", flush=True)
            return
        r_start, r_end = get_recent_calendar_days_range(
            int(args.recent_days), end_day=signal_day
        )
        extra_periods.append(
            (f"近{int(args.recent_days)}日", r_start, r_end, f"recent{int(args.recent_days)}d")
        )
    if args.week_before_days_ago is not None:
        if int(args.week_before_days_ago) < 0:
            print("--week-before-days-ago 必须 >= 0", flush=True)
            return
        n = int(args.week_before_days_ago)
        w_start, w_end = get_week_before_days_ago(n, anchor_day=calendar_today)
        extra_periods.append(
            (f"{n}天前倒推一周", w_start, w_end, f"weekbefore{n}d")
        )
    if args.this_week_hits:
        w_start, w_end = get_current_period_range()
        extra_periods.append(("本周", w_start, w_end, "thisweek"))
    elif args.last_week_hits:
        w_start, w_end = get_previous_period_range()
        extra_periods.append(("上周", w_start, w_end, "lastweek"))

    period_hits: Dict[str, List[Dict[str, Any]]] = {
        label: [] for label, _, _, _ in extra_periods
    }
    for label, p_start, p_end, _ in extra_periods:
        print(f"{label}区间: {p_start} ~ {p_end}（将随全池扫描一并统计）", flush=True)

    n_pool = len(pool)
    fetch_ok = 0
    fetch_fail = 0
    print(f"开始拉取全池行情，共 {n_pool} 只（PID={os.getpid()}）...", flush=True)
    try:
        for k, (_, row) in enumerate(pool.iterrows()):
            if k > 0 and k % BAOSTOCK_RELOGIN_EVERY == 0:
                baostock_relogin(verbose=False)
            if k == 0 or (k > 0 and k % SCAN_PROGRESS_EVERY == 0):
                print(f"扫描进度: {k}/{n_pool}", flush=True)
            code = normalize_stock_code(str(row["代码"]))
            name = str(row["名称"])
            try:
                raw = fetch_stock_df_retry(code, baostock_manage_login=False)
                data = _prepare_ohlcv_df_with_date(raw)
                if data is None:
                    fetch_fail += 1
                    continue
                fetch_ok += 1
                hit = check_hit_on_signal_day(
                    data, signal_day, threshold, pre20_max_pct
                )
                if hit is not None:
                    hit["股票代码"] = code
                    hit["股票名称"] = name
                    hits.append(hit)
                for label, p_start, p_end, _ in extra_periods:
                    in_period = data[
                        (data["_日期"] >= p_start) & (data["_日期"] <= p_end)
                    ]
                    for idx in in_period.index:
                        w_hit = check_hit_at_row(
                            data, int(idx), threshold, pre20_max_pct
                        )
                        if w_hit is None:
                            continue
                        w_hit["股票代码"] = code
                        w_hit["股票名称"] = name
                        period_hits[label].append(w_hit)
            except Exception as e:
                fetch_fail += 1
                print(f"{code} 处理异常: {e}", flush=True)
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    print(
        f"扫描拉数统计: 成功 {fetch_ok} 只, 失败或数据不足 {fetch_fail} 只",
        flush=True,
    )

    hits = sorted(hits, key=lambda x: x["股票代码"])
    hits, hit_stats = filter_hits_post(hits)
    post_note = format_post_filter_stats(hit_stats)
    print(post_note, flush=True)

    date_tag = datetime.now().strftime("%Y%m%d")
    body = build_email_body(
        hits=hits,
        threshold=threshold,
        pre20_max_pct=pre20_max_pct,
        total_count=len(pool),
        signal_day=signal_day,
        post_filter_note=post_note,
    )

    report_daily = _root / f"volume_ma_filter_daily_all_{date_tag}.txt"
    report_daily.write_text(body, encoding="utf-8")
    print(body, flush=True)
    print(f"\n报告1（当日/最近交易日命中）: {report_daily}", flush=True)

    period_bodies: List[Tuple[str, str, Path]] = []
    for label, p_start, p_end, suffix in extra_periods:
        phits = dedupe_hits_first_within_days(period_hits.get(label, []))
        phits, pstats = filter_hits_post(phits)
        pnote = format_post_filter_stats(pstats)
        print(f"{label}{pnote}", flush=True)
        pbody = build_week_hits_email_body(
            phits,
            threshold,
            pre20_max_pct,
            len(pool),
            p_start,
            p_end,
            period_label=label,
            post_filter_note=pnote,
        )
        preport = _root / f"volume_ma_filter_daily_all_{suffix}_{date_tag}.txt"
        preport.write_text(pbody, encoding="utf-8")
        print("\n" + pbody, flush=True)
        print(f"\n报告（{label}命中）: {preport}", flush=True)
        period_bodies.append((label, pbody, preport))
        if args.above_ma120_or_ma250:
            ma_hits, ma_stats = filter_hits_above_ma120_or_ma250(phits)
            ma_note = format_above_ma_filter_stats(ma_stats)
            print(f"{label}{ma_note}", flush=True)
            ma_body = build_above_ma_report_body(
                ma_hits,
                title=f"{label}量能命中 · 股价在年线或半年线上方（{p_start} ~ {p_end}）",
                source_note=f"来源: {preport.name}",
                ma_filter_note=ma_note,
            )
            ma_report = (
                _root
                / f"volume_ma_filter_daily_all_{suffix}_above_ma120or250_{date_tag}.txt"
            )
            ma_report.write_text(ma_body, encoding="utf-8")
            print("\n" + ma_body, flush=True)
            print(f"\n报告（{label}·年线/半年线上方）: {ma_report}", flush=True)
            period_bodies.append((f"{label}·年线/半年线上方", ma_body, ma_report))

    if args.above_ma120_or_ma250:
        ma_daily, ma_daily_stats = filter_hits_above_ma120_or_ma250(hits)
        ma_daily_note = format_above_ma_filter_stats(ma_daily_stats)
        print(f"当日{ma_daily_note}", flush=True)
        ma_daily_body = build_above_ma_report_body(
            ma_daily,
            title=f"{signal_day} 日终量能命中 · 股价在年线或半年线上方",
            source_note=f"来源: {report_daily.name}",
            ma_filter_note=ma_daily_note,
        )
        ma_daily_report = (
            _root / f"volume_ma_filter_daily_all_above_ma120or250_{date_tag}.txt"
        )
        ma_daily_report.write_text(ma_daily_body, encoding="utf-8")
        print("\n" + ma_daily_body, flush=True)
        print(f"\n报告（当日·年线/半年线上方）: {ma_daily_report}", flush=True)
        period_bodies.append(("当日·年线/半年线上方", ma_daily_body, ma_daily_report))

    history_years = int(args.history_years)
    report_signal_hits_fwd = (
        _root / f"volume_ma_filter_daily_all_signal_hits_{history_years}y_{date_tag}.txt"
    )
    exported_signal_hits_fwd: Path | None = None
    signal_hits_fwd_summaries: List[str] = []
    if hits:
        print(
            f"\n开始报告（信号日命中股）: 近 {history_years} 年前瞻（共 {len(hits)} 条命中）...",
            flush=True,
        )
        exported_signal_hits_fwd, signal_hits_fwd_summaries = (
            export_signal_day_hits_forward_report(
                hits,
                threshold,
                pre20_max_pct,
                history_years,
                report_signal_hits_fwd,
                signal_day=signal_day,
            )
        )
        if exported_signal_hits_fwd:
            print(
                f"报告（信号日命中股近{history_years}年前瞻）: {exported_signal_hits_fwd}",
                flush=True,
            )

    report_forward = _root / f"volume_ma_filter_daily_all_{history_years}y_{date_tag}.txt"
    exported_forward: Path | None = None
    if args.skip_forward_report:
        print("已跳过报告2（--skip-forward-report）", flush=True)
    else:
        print(
            f"\n开始报告2：全池近 {history_years} 年前瞻（约 {len(pool)} 只，耗时较长）...",
            flush=True,
        )
        exported_forward = export_full_pool_forward_3y_report(
            pool, threshold, pre20_max_pct, history_years, report_forward
        )
        if exported_forward:
            print(
                f"报告2（全市场主板 {len(pool)} 只近{history_years}年前瞻）: {exported_forward}",
                flush=True,
            )

    # 仅「本周」且无其他区间时，邮件以本周为主；否则以当日命中开头，再附各区间
    only_this_week = (
        len(period_bodies) == 1 and period_bodies[0][0] == "本周"
    )
    if only_this_week:
        email_parts = [period_bodies[0][1]]
        attach = [period_bodies[0][2]]
    else:
        email_parts = [body]
        attach = [report_daily]
        for _, pbody, preport in period_bodies:
            email_parts.append("\n\n---\n" + pbody)
            attach.append(preport)
    if signal_hits_fwd_summaries:
        email_parts.append("\n\n---\n" + "\n\n".join(signal_hits_fwd_summaries))
    if exported_signal_hits_fwd and exported_signal_hits_fwd.is_file():
        attach.append(exported_signal_hits_fwd)
        email_parts.append(
            f"\n信号日命中股近{history_years}年前瞻报告: {exported_signal_hits_fwd}\n"
        )
    if exported_forward:
        email_parts.append(
            "\n\n---\n"
            + f"报告1（当日命中）: {report_daily}\n"
            + f"报告2（全池近{history_years}年前瞻表格）: {report_forward}\n"
        )
        attach.append(report_forward)
    else:
        email_parts.append(
            "\n\n---\n"
            + f"报告1（当日命中）: {report_daily}\n"
            + (
                f"报告2: 未生成（本次使用了 --skip-forward-report 或导出失败）\n"
                if args.skip_forward_report or not report_forward.is_file()
                else f"报告2（全池近{history_years}年前瞻表格）: {report_forward}\n"
            )
        )
    email_parts.append(f"\n（附件共 {len(attach)} 个 txt）\n")
    email_body = "".join(email_parts)
    if not args.no_send_email:
        send_result_email(email_body, attachment_paths=attach)
    else:
        print("已跳过邮件（--no-send-email）", flush=True)


if __name__ == "__main__":
    main()
