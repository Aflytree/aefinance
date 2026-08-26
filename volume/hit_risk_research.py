"""对量能命中股搜集近期减持/违规/负面相关公告与新闻。"""
from __future__ import annotations

import argparse
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import akshare as ak
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent

NEG_TITLE_KEYS = (
    "减持",
    "违规",
    "处罚",
    "立案",
    "警示",
    "问询",
    "监管",
    "谴责",
    "诉讼",
    "仲裁",
    "冻结",
    "质押",
    "违约",
    "造假",
    "调查",
    "预亏",
    "亏损",
    "暴雷",
    "商誉减值",
    "退市",
    "ST",
    "风险提示",
    "关注函",
    "监管函",
    "终止上市",
    "破产",
    "被执行",
    "失信",
    "占用",
    "处罚决定",
    "警示函",
    "公开谴责",
)
REDUCE_KEYS = ("减持", "股份转让", "权益变动", "持股变动")
VIOLATION_KEYS = ("违规", "处罚", "立案", "警示函", "监管函", "问询", "谴责", "关注函")


def _parse_codes_from_report(path: Path) -> List[Tuple[str, str]]:
    """从命中报告解析 (代码, 名称)。"""
    rows: List[Tuple[str, str]] = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\d{6})\s+(\S+)", line.strip())
        if not m:
            continue
        code, name = m.group(1), m.group(2)
        if code in seen:
            continue
        seen.add(code)
        rows.append((code, name))
    return rows


def _to_date(v: Any) -> Optional[date]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return pd.to_datetime(v).date()
    except Exception:
        return None


def _match_any(text: str, keys: tuple[str, ...]) -> List[str]:
    hit = [k for k in keys if k in text]
    return hit


def fetch_notices(code: str) -> pd.DataFrame:
    try:
        df = ak.stock_individual_notice_report(security=code, symbol="全部")
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        print(f"{code} 公告拉取失败: {e}", flush=True)
        return pd.DataFrame()


def fetch_news(code: str) -> pd.DataFrame:
    try:
        df = ak.stock_news_em(symbol=code)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        print(f"{code} 新闻拉取失败: {e}", flush=True)
        return pd.DataFrame()


def load_cninfo_reduce_map(lookback_days: int = 180) -> Dict[str, List[str]]:
    """巨潮高管减持明细，按代码汇总近窗记录。"""
    out: Dict[str, List[str]] = {}
    try:
        df = ak.stock_hold_management_detail_cninfo(symbol="减持")
    except Exception as e:
        print(f"巨潮减持明细拉取失败: {e}", flush=True)
        return out
    if df is None or df.empty:
        return out
    since = datetime.now().date() - timedelta(days=lookback_days)
    code_col = "证券代码" if "证券代码" in df.columns else df.columns[0]
    for _, row in df.iterrows():
        code = str(row.get(code_col, "")).zfill(6)
        d = _to_date(row.get("公告日期")) or _to_date(row.get("截止日期"))
        if d is None or d < since:
            continue
        name = str(row.get("证券简称", "") or "")
        person = str(row.get("高管姓名") or row.get("董监高姓名") or "")
        reason = str(row.get("持股变动原因") or "")
        qty = row.get("变动数量")
        price = row.get("成交均价")
        line = f"{d} {name} {person} 变动数量:{qty} 均价:{price} 原因:{reason}".strip()
        out.setdefault(code, []).append(line)
    return out


def research_one(
    code: str,
    name: str,
    *,
    since: date,
    news_limit: int = 8,
    cninfo_reduce: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    notices = fetch_notices(code)
    news = fetch_news(code)

    reduce_items: List[str] = []
    violation_items: List[str] = []
    neg_notice_items: List[str] = []
    if not notices.empty and "公告标题" in notices.columns:
        for _, row in notices.iterrows():
            title = str(row.get("公告标题", "") or "")
            ntype = str(row.get("公告类型", "") or "")
            d = _to_date(row.get("公告日期"))
            if d is None or d < since:
                continue
            blob = f"{title} {ntype}"
            line = f"{d} [{ntype}] {title}"
            url = str(row.get("网址", "") or "")
            if url:
                line += f" | {url}"
            if _match_any(blob, REDUCE_KEYS):
                reduce_items.append(line)
            if _match_any(blob, VIOLATION_KEYS):
                violation_items.append(line)
            elif _match_any(blob, NEG_TITLE_KEYS) and not _match_any(blob, REDUCE_KEYS):
                neg_notice_items.append(line)

    if cninfo_reduce:
        for line in cninfo_reduce.get(code, [])[:12]:
            tagged = f"[巨潮减持明细] {line}"
            if tagged not in reduce_items:
                reduce_items.append(tagged)

    neg_news: List[str] = []
    if not news.empty and "新闻标题" in news.columns:
        for _, row in news.head(40).iterrows():
            title = str(row.get("新闻标题", "") or "")
            content = str(row.get("新闻内容", "") or "")
            d = _to_date(row.get("发布时间"))
            if d is not None and d < since:
                continue
            blob = f"{title} {content}"
            keys = _match_any(blob, NEG_TITLE_KEYS)
            if not keys:
                continue
            src = str(row.get("文章来源", "") or "")
            link = str(row.get("新闻链接", "") or "")
            d_s = str(d) if d else str(row.get("发布时间", ""))
            line = f"{d_s} [{','.join(keys[:3])}] {title}"
            if src:
                line += f" ({src})"
            if link:
                line += f" | {link}"
            neg_news.append(line)
            if len(neg_news) >= news_limit:
                break

    flags = []
    if reduce_items:
        flags.append("有减持/持股变动相关")
    if violation_items:
        flags.append("有违规/监管相关公告")
    if neg_notice_items or neg_news:
        flags.append("有负面关键词命中")
    if not flags:
        flags.append("近窗内未见明显减持/违规/负面关键词")

    return {
        "代码": code,
        "名称": name,
        "标记": flags,
        "减持相关公告": reduce_items[:15],
        "违规监管公告": violation_items[:12],
        "其他负面公告": neg_notice_items[:8],
        "负面新闻": neg_news[:news_limit],
    }


def research_codes(
    codes: List[Tuple[str, str]],
    *,
    lookback_days: int = 90,
    sleep_s: float = 0.35,
) -> List[Dict[str, Any]]:
    since = datetime.now().date() - timedelta(days=lookback_days)
    print("拉取巨潮减持明细全表...", flush=True)
    cninfo_reduce = load_cninfo_reduce_map(lookback_days=max(lookback_days, 180))
    print(f"巨潮减持明细近窗涉及股票数: {len(cninfo_reduce)}", flush=True)
    out: List[Dict[str, Any]] = []
    for i, (code, name) in enumerate(codes, 1):
        print(f"信息搜集 {i}/{len(codes)} {code} {name}", flush=True)
        out.append(
            research_one(
                code,
                name,
                since=since,
                cninfo_reduce=cninfo_reduce,
            )
        )
        time.sleep(sleep_s)
    return out


def _is_clean(item: Dict[str, Any]) -> bool:
    return not (
        item.get("减持相关公告")
        or item.get("违规监管公告")
        or item.get("其他负面公告")
        or item.get("负面新闻")
    )


def format_research_report(
    results: List[Dict[str, Any]],
    *,
    title: str,
    since: date,
    source_note: str,
) -> str:
    clean = [x for x in results if _is_clean(x)]
    flagged = [x for x in results if not _is_clean(x)]
    lines = [
        title,
        source_note,
        f"信息窗: {since} 至今（公告/新闻标题关键词筛：减持、违规、处罚、立案、问询、风险提示等）",
        f"标的数: {len(results)}（无负面线索 {len(clean)} / 有线索 {len(flagged)}）",
        "说明: 基于东财公告/新闻与巨潮减持明细的关键词匹配，仅供线索排查，非完整尽调。",
        "",
        "===== 近窗无负面线索标的（单独列表）=====",
    ]
    if not clean:
        lines.append("（无）")
    else:
        for item in clean:
            lines.append(f"- {item['代码']} {item['名称']}")
    lines.extend(
        [
            "",
            "===== 有减持/违规/负面线索标的（摘要）=====",
        ]
    )
    if not flagged:
        lines.append("（无）")
    else:
        for item in flagged:
            lines.append(f"- {item['代码']} {item['名称']} | {'；'.join(item['标记'])}")
    lines.extend(["", "===== 全部标的明细 =====", ""])
    for i, item in enumerate(results, 1):
        lines.append(f"[{i}] {item['代码']} {item['名称']} | {'；'.join(item['标记'])}")
        if item["减持相关公告"]:
            lines.append("  减持/持股变动:")
            for x in item["减持相关公告"]:
                lines.append(f"    - {x}")
        if item["违规监管公告"]:
            lines.append("  违规/监管公告:")
            for x in item["违规监管公告"]:
                lines.append(f"    - {x}")
        if item["其他负面公告"]:
            lines.append("  其他负面公告:")
            for x in item["其他负面公告"]:
                lines.append(f"    - {x}")
        if item["负面新闻"]:
            lines.append("  负面相关新闻:")
            for x in item["负面新闻"]:
                lines.append(f"    - {x}")
        if _is_clean(item):
            lines.append("  （近窗无匹配条目）")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="命中股减持/违规/负面信息搜集")
    parser.add_argument("--report", required=True, help="命中报告 txt 路径")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--out", default="", help="输出报告路径")
    parser.add_argument("--send-email", action="store_true")
    args = parser.parse_args()

    src = Path(args.report)
    if not src.is_file():
        raise SystemExit(f"找不到报告: {src}")
    codes = _parse_codes_from_report(src)
    if not codes:
        raise SystemExit(f"报告中未解析到股票代码: {src}")

    since = datetime.now().date() - timedelta(days=int(args.lookback_days))
    results = research_codes(codes, lookback_days=int(args.lookback_days))
    body = format_research_report(
        results,
        title=f"命中股风险信息搜集（近{int(args.lookback_days)}日）",
        since=since,
        source_note=f"来源命中报告: {src.name}",
    )
    out = Path(args.out) if args.out else _ROOT / f"hit_risk_research_{src.stem}.txt"
    out.write_text(body, encoding="utf-8")
    print(body, flush=True)
    print(f"报告: {out}", flush=True)

    if args.send_email:
        import sys

        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        from efi_email import send

        send(body, attachments=[out, src] if src.is_file() else [out])
        print("邮件已发送。", flush=True)


if __name__ == "__main__":
    main()
