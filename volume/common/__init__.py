"""量能均线筛选公共实现。"""
import sys
from pathlib import Path

_VOL_DIR = Path(__file__).resolve().parent.parent
_ROOT = _VOL_DIR.parent
for _p in (_VOL_DIR, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from .constants import (  # noqa: E402
    DEFAULT_HISTORY_YEARS,
    DEFAULT_PRE20_MAX_PCT,
    DEFAULT_THRESHOLD,
    MA20_TO_MA5_FACTOR,
    MA5_TO_MA10_MAX_RATIO,
    MA5_TO_MA10_MIN_RATIO,
    MAX_SIGNAL_DAY_PCT_CHG,
    MIN_SIGNAL_DAY_PCT_CHG,
)
from .data import fetch_stock_df, normalize_stock_code  # noqa: E402
from .export import export_forward_result  # noqa: E402
from .filter import (  # noqa: E402
    check_hit_at_row,
    method3_price_ma_bullish_expanding,
    prepare_ohlcv_df,
    screen_by_volume_ma,
)
from .forward import (  # noqa: E402
    build_summary_rows,
    format_forward_summary,
    get_signal_forward_returns,
    prettify_detail_df,
)
from .indicators import add_price_macd_columns, macd_value_at  # noqa: E402
from .notify import format_email_body, parse_stock_codes, send_result_email  # noqa: E402

# 兼容旧名
_prepare_ohlcv_df = prepare_ohlcv_df
_macd_value_at = macd_value_at

__all__ = [
    "DEFAULT_HISTORY_YEARS",
    "DEFAULT_PRE20_MAX_PCT",
    "DEFAULT_THRESHOLD",
    "MA20_TO_MA5_FACTOR",
    "MA5_TO_MA10_MAX_RATIO",
    "MA5_TO_MA10_MIN_RATIO",
    "MAX_SIGNAL_DAY_PCT_CHG",
    "MIN_SIGNAL_DAY_PCT_CHG",
    "_macd_value_at",
    "_prepare_ohlcv_df",
    "add_price_macd_columns",
    "build_summary_rows",
    "check_hit_at_row",
    "export_forward_result",
    "fetch_stock_df",
    "format_email_body",
    "format_forward_summary",
    "get_signal_forward_returns",
    "macd_value_at",
    "method3_price_ma_bullish_expanding",
    "normalize_stock_code",
    "parse_stock_codes",
    "prettify_detail_df",
    "prepare_ohlcv_df",
    "screen_by_volume_ma",
    "send_result_email",
]
