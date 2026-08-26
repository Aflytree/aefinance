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
    HITS_PERIOD_DAYS,
    MA20_TO_MA5_FACTOR,
    MA5_TO_MA10_MAX_RATIO,
    MA5_TO_MA10_MIN_RATIO,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    AVG_DAILY_AMOUNT_LOOKBACK,
    MAX_HIT_CLOSE_PRICE,
    MAX_HIT_PE,
    MAX_HIT_TURNOVER_PCT,
    MAX_SIGNAL_DAY_PCT_CHG,
    MIN_AVG_DAILY_AMOUNT,
    MIN_HIT_CLOSE_PRICE,
    MIN_HIT_TURNOVER_PCT,
    MIN_SIGNAL_DAY_PCT_CHG,
    SIGNAL_DAY_BAOSTOCK_RETRIES,
    SIGNAL_DAY_BAOSTOCK_RETRY_SECONDS,
)
from .fundamentals import (  # noqa: E402
    describe_post_hit_filters,
    filter_hits_post,
    format_post_filter_stats,
)
from .data import fetch_stock_df, normalize_stock_code  # noqa: E402
from .export import export_forward_result  # noqa: E402
from .filter import (  # noqa: E402
    check_hit_at_row,
    dedupe_hits_first_within_days,
    describe_filter_conditions_bullets,
    describe_filter_conditions_text,
    describe_method3_condition,
    describe_signal_dedupe_rule,
    format_filter_conditions_line,
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
    "HITS_PERIOD_DAYS",
    "MARKET_CLOSE_HOUR",
    "MARKET_CLOSE_MINUTE",
    "MA20_TO_MA5_FACTOR",
    "AVG_DAILY_AMOUNT_LOOKBACK",
    "MAX_HIT_CLOSE_PRICE",
    "MAX_HIT_PE",
    "MAX_HIT_TURNOVER_PCT",
    "MIN_AVG_DAILY_AMOUNT",
    "MIN_HIT_CLOSE_PRICE",
    "MIN_HIT_TURNOVER_PCT",
    "SIGNAL_DAY_BAOSTOCK_RETRIES",
    "SIGNAL_DAY_BAOSTOCK_RETRY_SECONDS",
    "MA5_TO_MA10_MAX_RATIO",
    "MA5_TO_MA10_MIN_RATIO",
    "MAX_SIGNAL_DAY_PCT_CHG",
    "MIN_SIGNAL_DAY_PCT_CHG",
    "describe_post_hit_filters",
    "filter_hits_post",
    "format_post_filter_stats",
    "_macd_value_at",
    "_prepare_ohlcv_df",
    "add_price_macd_columns",
    "build_summary_rows",
    "check_hit_at_row",
    "dedupe_hits_first_within_days",
    "describe_filter_conditions_bullets",
    "describe_filter_conditions_text",
    "describe_method3_condition",
    "describe_signal_dedupe_rule",
    "export_forward_result",
    "format_filter_conditions_line",
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
