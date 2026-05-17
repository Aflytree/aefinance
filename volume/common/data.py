from datetime import datetime, timedelta

import pandas as pd

from stock_history import get_quote_history_baostock


def normalize_stock_code(stock_code: str) -> str:
    code = stock_code.strip()
    if "." in code:
        return code.split(".")[-1]
    return code


def fetch_stock_df(
    stock_code: str,
    days: int = 40,
    *,
    baostock_manage_login: bool = True,
    baostock_verbose: bool = True,
) -> pd.DataFrame:
    end = datetime.now().strftime("%Y%m%d")
    lookback_calendar = max(days * 3, 420)
    beg = (datetime.now() - timedelta(days=lookback_calendar)).strftime("%Y%m%d")
    return get_quote_history_baostock(
        stock_code,
        beg=beg,
        end=end,
        manage_login=baostock_manage_login,
        verbose=baostock_verbose,
    )
