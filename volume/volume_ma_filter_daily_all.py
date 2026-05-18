import argparse
import os
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
    HITS_PERIOD_DAYS,
    HITS_RANGE_END,
    HITS_RANGE_START,
    build_summary_rows,
    check_hit_at_row,
    dedupe_hits_first_within_days,
    describe_signal_dedupe_rule,
    export_forward_result,
    format_filter_conditions_line,
    format_forward_summary,
    fetch_stock_df,
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
    df = ak.stock_info_a_code_name()
    df = df.rename(columns={"code": "代码", "name": "名称"})

    # 两市主板：沪市主板(600/601/603/605)、深市主板(000/001/002/003)
    mainboard_prefix = ("600", "601", "603", "605", "000", "001", "002", "003")
    # mainboard_prefix = ("600")
    df = df[df["代码"].astype(str).str.startswith(mainboard_prefix)].copy()

    # 剔除 ST、*ST、退市风险股票；创业板(300)已通过前缀排除
    df = df[~df["名称"].str.contains(r"ST|\*ST|退", case=False, regex=True, na=False)]
    return df.reset_index(drop=True)


def _prepare_ohlcv_df_with_date(df: pd.DataFrame) -> pd.DataFrame | None:
    """全池扫描用：在统一 OHLCV 准备基础上附加日历日期列。"""
    data = prepare_ohlcv_df(df)
    if data is None:
        return None
    data = data.copy()
    data["_日期"] = pd.to_datetime(data["日期"], errors="coerce").dt.date
    return data


def get_market_latest_trading_day(ref: datetime | None = None) -> date:
    """以沪指样本股最新 K 线日期作为全市场信号交易日（非交易日则自然落到最近交易日）。"""
    ref = ref or datetime.now()
    raw = fetch_stock_df(
        "000001",
        baostock_manage_login=False,
        baostock_verbose=False,
    )
    data = _prepare_ohlcv_df_with_date(raw)
    if data is None or data.empty:
        return ref.date()
    return data["_日期"].iloc[-1]


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


def get_current_period_range(
    ref: datetime | None = None,
    *,
    period_days: int = HITS_PERIOD_DAYS,
) -> Tuple[date, date]:
    """最近 period_days 个自然日（含参考日）。"""
    ref = ref or datetime.now()
    end = ref.date()
    start = end - timedelta(days=period_days - 1)
    return start, end


def get_previous_period_range(
    ref: datetime | None = None,
    *,
    period_days: int = HITS_PERIOD_DAYS,
) -> Tuple[date, date]:
    """紧邻其前的 period_days 个自然日。"""
    ref = ref or datetime.now()
    cur_start, _ = get_current_period_range(ref, period_days=period_days)
    end = cur_start - timedelta(days=1)
    start = end - timedelta(days=period_days - 1)
    return start, end


def scan_last_week_hits(
    pool_df: pd.DataFrame,
    threshold: float,
    pre20_max_pct: float,
    *,
    baostock_manage_login: bool = False,
) -> Tuple[List[Dict[str, Any]], date, date]:
    """扫描池内每只股票在前一统计周期（7 自然日）内的全部命中。"""
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
) -> str:
    signal_day = signal_day or datetime.now().date()
    lines = [
        f"{signal_day} 日终量能筛选（信号交易日；全市场主板，剔除ST，排除创业板；逻辑同 volume_ma_filter_detail）",
        f"扫描范围: {total_count} 只",
        format_filter_conditions_line(
            threshold=threshold, pre20_max_pct=pre20_max_pct
        ),
        f"命中数量: {len(hits)}",
        "",
    ]

    for item in hits:
        lines.append(
            f"{item['股票代码']} {item['股票名称']} | 日期:{item['日期']} | MA5/MA10:{item['MA5/MA10']:.3f} | "
            f"收盘/价格MA10:{item['收盘']:.2f}/{item['价格MA10']:.2f} | "
            f"当天涨跌幅:{item['当天涨跌幅%']:.2f}% | 前20日涨跌幅:{item['信号日前20日涨跌幅%']:.2f}%"
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
) -> str:
    lines = [
        f"{period_label}量能命中（{week_start} ~ {week_end}，最近{HITS_PERIOD_DAYS}个自然日内的交易日）",
        f"扫描范围: {total_count} 只（全市场主板，剔除ST）",
        format_filter_conditions_line(
            threshold=threshold,
            pre20_max_pct=pre20_max_pct,
            label="条件同当日筛",
        ),
        f"命中数量: {len(hits)}（{describe_signal_dedupe_rule()}）",
        "",
    ]
    if not hits:
        lines.append(f"{period_label}无命中记录。")
        return "\n".join(lines)
    for item in hits:
        lines.append(
            f"{item['股票代码']} {item['股票名称']} | 命中日期:{item['日期']} | "
            f"MA5/MA10:{item['MA5/MA10']:.3f} | 收盘/价格MA10:{item['收盘']:.2f}/{item['价格MA10']:.2f} | "
            f"当天涨跌幅:{item['当天涨跌幅%']:.2f}% | 前20日涨跌幅:{item['信号日前20日涨跌幅%']:.2f}%"
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
        help="额外扫描上一统计周期内的命中（周期见 common.constants.HITS_PERIOD_DAYS）",
    )
    parser.add_argument(
        "--this-week-hits",
        action="store_true",
        help="额外扫描最近一个统计周期内的命中（周期见 common.constants.HITS_PERIOD_DAYS）",
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
    print(f"主板股票池数量(剔除ST/创业板): {len(pool)}", flush=True)

    lg = bs.login()
    if lg.error_code != "0":
        print(f"Baostock 登录失败: {lg.error_msg}")
        return

    signal_day = get_market_latest_trading_day()
    calendar_today = datetime.now().date()
    if signal_day != calendar_today:
        print(
            f"今日({calendar_today})无行情或非交易日，信号日取最近交易日: {signal_day}",
            flush=True,
        )
    else:
        print(f"信号交易日: {signal_day}", flush=True)

    hits: List[Dict[str, Any]] = []
    week_hits: List[Dict[str, Any]] = []
    week_start: date | None = None
    week_end: date | None = None
    week_period_label = ""
    if args.this_week_hits and args.last_week_hits:
        print("请只指定 --this-week-hits 或 --last-week-hits 之一。", flush=True)
        return
    if args.this_week_hits:
        if HITS_RANGE_START is not None and HITS_RANGE_END is not None:
            week_start, week_end = HITS_RANGE_START, HITS_RANGE_END
            week_period_label = f"{week_start}~{week_end}"
        else:
            week_start, week_end = get_current_period_range()
            week_period_label = f"近{HITS_PERIOD_DAYS}日"
    elif args.last_week_hits:
        week_start, week_end = get_previous_period_range()
        week_period_label = f"前{HITS_PERIOD_DAYS}日"
    if week_start is not None and week_end is not None:
        print(
            f"{week_period_label}区间: {week_start} ~ {week_end}（将随全池扫描一并统计）",
            flush=True,
        )

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
                if week_start is not None and week_end is not None:
                    in_week = data[
                        (data["_日期"] >= week_start) & (data["_日期"] <= week_end)
                    ]
                    for idx in in_week.index:
                        w_hit = check_hit_at_row(
                            data, int(idx), threshold, pre20_max_pct
                        )
                        if w_hit is None:
                            continue
                        w_hit["股票代码"] = code
                        w_hit["股票名称"] = name
                        week_hits.append(w_hit)
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
    week_hits = dedupe_hits_first_within_days(week_hits)
    date_tag = datetime.now().strftime("%Y%m%d")
    body = build_email_body(
        hits=hits,
        threshold=threshold,
        pre20_max_pct=pre20_max_pct,
        total_count=len(pool),
        signal_day=signal_day,
    )

    report_daily = _root / f"volume_ma_filter_daily_all_{date_tag}.txt"
    report_daily.write_text(body, encoding="utf-8")
    print(body, flush=True)
    print(f"\n报告1（当日/最近交易日命中）: {report_daily}", flush=True)

    week_body = ""
    report_week: Path | None = None
    if week_start is not None and week_end is not None:
        week_body = build_week_hits_email_body(
            week_hits,
            threshold,
            pre20_max_pct,
            len(pool),
            week_start,
            week_end,
            period_label=week_period_label,
        )
        week_suffix = "thisweek" if args.this_week_hits else "lastweek"
        report_week = _root / f"volume_ma_filter_daily_all_{week_suffix}_{date_tag}.txt"
        report_week.write_text(week_body, encoding="utf-8")
        print("\n" + week_body, flush=True)
        print(f"\n报告（{week_period_label}命中）: {report_week}", flush=True)

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

    if week_body and args.this_week_hits:
        email_parts = [week_body]
        attach = [report_week] if report_week is not None else []
    else:
        email_parts = [body]
        attach = [report_daily]
        if week_body:
            email_parts.append("\n\n---\n" + week_body)
            if report_week is not None:
                attach.append(report_week)
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
