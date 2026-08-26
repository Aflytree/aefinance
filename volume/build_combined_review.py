"""扫描完成后：按邮件结构合并报告，并为年线/半年线上方段附加负面与减持判断。"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REDUCE_PLAN_KEYS = ("减持计划", "拟减持", "减持股份计划", "股份减持计划")
REDUCE_DONE_KEYS = (
    "减持结果",
    "股份结果",
    "实施情况",
    "期限届满",
    "时间届满",
    "届满暨",
    "减持完毕",
    "完成减持",
    "已减持",
)
REDUCE_OTHER_KEYS = ("减持", "权益变动", "持股变动")
NEG_NOTICE_KEYS = (
    "诉讼",
    "仲裁",
    "问询",
    "处罚",
    "立案",
    "警示",
    "监管函",
    "关注函",
    "谴责",
    "质押",
    "冻结",
    "预亏",
    "亏损",
    "违规",
    "风险提示",
    "异常波动",
)
IGNORE_TITLE_KEYS = (
    "资金占用及其他关联资金往来",
    "非经营性资金占用及其他关联资金往来情况汇总表",
    "公司债券",
    "科技创新公司债券",
    "发行结果公告",
)


def parse_hits(path: Path) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\d{6})\s+(.+?)\s+\|\s+(.+)$", line.strip())
        if not m:
            continue
        rows.append((m.group(1), m.group(2).strip(), m.group(3).strip()))
    return rows


def parse_dates_in_text(text: str) -> List[date]:
    dates: List[date] = []
    pats = [
        r"(20\d{2})年(\d{1,2})月(\d{1,2})日",
        r"(20\d{2})-(\d{1,2})-(\d{1,2})",
        r"(20\d{2})/(\d{1,2})/(\d{1,2})",
    ]
    for pat in pats:
        for y, mo, d in re.findall(pat, text):
            try:
                dates.append(date(int(y), int(mo), int(d)))
            except ValueError:
                pass
    return dates


def classify_reduce(
    title: str,
    ntype: str,
    notice_date: Optional[date],
    *,
    today: date,
    window_end: date,
) -> Tuple[str, str]:
    blob = f"{title} {ntype}"
    if any(k in title for k in IGNORE_TITLE_KEYS):
        return "none", ""
    if "增持" in blob and "减持" not in blob:
        return "none", ""
    if any(k in blob for k in REDUCE_DONE_KEYS) or (
        "届满" in blob and ("减持" in blob or "计划" in blob)
    ):
        return "past_or_done", f"{notice_date} {title}"
    if any(k in blob for k in REDUCE_PLAN_KEYS) or ("减持" in blob and "计划" in blob):
        ds = parse_dates_in_text(title)
        if len(ds) >= 2:
            start, end = min(ds), max(ds)
            if start <= window_end and end >= today:
                return (
                    "future_plan",
                    f"计划窗口约{start}~{end}；公告:{notice_date} {title}",
                )
            if end < today:
                return (
                    "past_or_done",
                    f"计划窗口已结束({start}~{end})；公告:{notice_date} {title}",
                )
        if notice_date and notice_date >= today - timedelta(days=180):
            return (
                "future_plan",
                f"近期减持计划公告(未解析到明确起止日，保守视为未来可能实施)；公告:{notice_date} {title}",
            )
        return "other_change", f"{notice_date} {title}"
    if any(k in blob for k in REDUCE_OTHER_KEYS):
        return "other_change", f"{notice_date} {title}"
    return "none", ""


def fetch_notices(code: str) -> pd.DataFrame:
    try:
        df = ak.stock_individual_notice_report(security=code, symbol="全部")
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        print(f"{code} notices fail: {e}", flush=True)
        return pd.DataFrame()


def analyze_stock(code: str, name: str, *, today: date) -> Dict[str, Any]:
    window_end = today + timedelta(days=90)
    df = fetch_notices(code)
    future_plans: List[str] = []
    past_reduce: List[str] = []
    neg_flags: List[str] = []
    neg_details: List[str] = []

    if not df.empty and "公告标题" in df.columns:
        since = today - timedelta(days=180)
        for _, row in df.iterrows():
            title = str(row.get("公告标题", "") or "")
            ntype = str(row.get("公告类型", "") or "")
            d: Optional[date] = None
            try:
                d = pd.to_datetime(row.get("公告日期")).date()
            except Exception:
                pass
            if d is not None and d < since:
                continue
            if any(k in title for k in IGNORE_TITLE_KEYS[:2]):
                continue

            tag, detail = classify_reduce(
                title, ntype, d, today=today, window_end=window_end
            )
            if tag == "future_plan":
                future_plans.append(detail)
            elif tag == "past_or_done":
                past_reduce.append(detail)

            meaningful = [k for k in NEG_NOTICE_KEYS if k in f"{title} {ntype}"]
            if meaningful:
                for k in meaningful:
                    if k not in neg_flags:
                        neg_flags.append(k)
                neg_details.append(f"{d} [{ntype}] {title}")

    if past_reduce and future_plans:
        if any("结果" in x or "届满" in x for x in past_reduce):
            future_plans = []

    if future_plans:
        reduce_3m = "是（公告显示未来可能/计划减持）"
    else:
        reduce_3m = "否（近窗未见明确未来3个月减持计划）"

    return {
        "代码": code,
        "名称": name,
        "未来3个月是否开始减持": reduce_3m,
        "负面指标": neg_flags,
        "负面明细": neg_details[:3],
    }


def annotate_above_section(src: Path, cache: Dict[str, Dict[str, Any]]) -> str:
    out: List[str] = []
    for ln in src.read_text(encoding="utf-8").splitlines():
        out.append(ln)
        m = re.match(r"^(\d{6})\s+", ln)
        if not m:
            continue
        info = cache.get(m.group(1))
        if not info:
            out.append("  负面指标: 未评估 | 未来3个月是否开始减持: 未评估")
            continue
        negs = "、".join(info["负面指标"]) if info["负面指标"] else "无（近窗未见诉讼/问询/质押/处罚等）"
        out.append(
            f"  负面指标: {negs} | 未来3个月是否开始减持: {info['未来3个月是否开始减持']}"
        )
        for d in info["负面明细"]:
            out.append(f"    - {d}")
    return "\n".join(out)


def build_for_tag(date_tag: str, *, today: date | None = None) -> Path:
    today = today or datetime.now().date()
    daily = ROOT / f"volume_ma_filter_daily_all_{date_tag}.txt"
    week = ROOT / f"volume_ma_filter_daily_all_weekbefore3d_{date_tag}.txt"
    week_above = ROOT / f"volume_ma_filter_daily_all_weekbefore3d_above_ma120or250_{date_tag}.txt"
    daily_above = ROOT / f"volume_ma_filter_daily_all_above_ma120or250_{date_tag}.txt"
    out = ROOT / f"volume_ma_filter_combined_review_{date_tag}.txt"

    missing = [p.name for p in (daily, week, week_above, daily_above) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少报告文件: {missing}")

    codes: List[Tuple[str, str]] = []
    seen = set()
    for path in (daily_above, week_above, week, daily):
        for c, n, _ in parse_hits(path):
            if c not in seen:
                seen.add(c)
                codes.append((c, n))

    cache: Dict[str, Dict[str, Any]] = {}
    # 优先分析均线上方标的
    priority = {c for c, _, _ in parse_hits(week_above) + parse_hits(daily_above)}
    ordered = [x for x in codes if x[0] in priority] + [
        x for x in codes if x[0] not in priority
    ]
    for i, (code, name) in enumerate(ordered, 1):
        print(f"analyze {i}/{len(ordered)} {code} {name}", flush=True)
        cache[code] = analyze_stock(code, name, today=today)
        time.sleep(0.25)

    parts = [
        daily.read_text(encoding="utf-8").rstrip(),
        "",
        "---",
        "",
        week.read_text(encoding="utf-8").rstrip(),
        "",
        "---",
        "",
        annotate_above_section(week_above, cache),
        "",
        "---",
        "",
        annotate_above_section(daily_above, cache),
        "",
        "---",
        "",
        "说明: 「年线/半年线上方」段落中，股票下一行附加负面指标与未来3个月减持判断"
        "（东财公告关键词启发式，非尽调）。",
    ]
    out.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return out


def send_combined(out: Path, date_tag: str) -> None:
    from efi_email import create_message, send_email

    body = out.read_text(encoding="utf-8")
    receiver = "19282286879@163.com"
    sender = "zhangaifei.2008@163.com"
    subject = f"量能合并复盘 {date_tag}（当日/周命中/均线上方+负面标注）"
    msg = create_message(sender, receiver, subject, body, attachments=[out])
    send_email("smtp.163.com", 465, sender, "FHhPc9WARnuqsG2e", receiver, msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-tag", required=True, help="如 20260821")
    parser.add_argument("--send-email", action="store_true")
    args = parser.parse_args()
    out = build_for_tag(args.date_tag)
    if args.send_email:
        send_combined(out, args.date_tag)
        print("邮件已发送。", flush=True)


if __name__ == "__main__":
    main()
