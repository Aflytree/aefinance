"""量能均线筛选策略默认参数。"""

from datetime import date

MA20_TO_MA5_FACTOR = 0.91
MA5_TO_MA10_MIN_RATIO = 1.0
MA5_TO_MA10_MAX_RATIO = 1.4
MIN_SIGNAL_DAY_PCT_CHG = -3.0
MAX_SIGNAL_DAY_PCT_CHG = 4.7
DEFAULT_THRESHOLD = 1.25
DEFAULT_HISTORY_YEARS = 3
DEFAULT_PRE20_MAX_PCT = 20.0

# --this-week-hits / --last-week-hits：统计区间长度；同股去重窗口（自然日）
HITS_PERIOD_DAYS = 7

# 周期命中固定区间（均非 None 时覆盖「近 N 日」）；同股在 HITS_PERIOD_DAYS 内只保留首次
HITS_RANGE_START = date(2026, 5, 11)
HITS_RANGE_END = date(2026, 5, 15)
