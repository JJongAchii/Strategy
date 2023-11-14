"""
Time relatied utility functions.
"""
import pandas as pd

DATE_FMT = r'\d{4}-(0?[1-9]|1[012])-(0?[1-9]|[12][0-9]|3[01])'

def parse_date(date: ...) -> pd.Timestamp:
    """
    Parse date string

    Args:
        date (any): any date

    Returns:
        str: date
    """
    if date is not None: return pd.Timestamp(date)
    return None

def current_time() -> pd.Timestamp:
    """
    Current time

    Returns:
        pd.Timestamp: current time
    """
    return pd.Timestamp('now')

def offset_time(
    time: pd.Timestamp = current_time(),
    **kwargs
) -> pd.Timestamp:
    """
    Relative to current time

    Returns:
        pd.Timestamp: relative to current time. Defaults to now.
    """
    if isinstance(time, (pd.Series, pd.DataFrame)):
        if isinstance(time.index, pd.DatetimeIndex):
            temp_time = time.copy()
            temp_time.index = offset_time(temp_time.index, **kwargs)
            return temp_time
    return time + pd.DateOffset(**kwargs)

def start_time(price: pd.DataFrame) -> pd.DataFrame:
    """
    Get start.

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.DataFrame: start
    """
    if isinstance(price, pd.DataFrame):
        return price.apply(start_time)
    return price.dropna().index[0]

def end_time(price: pd.DataFrame) -> pd.DataFrame:
    """
    Get end.

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.DataFrame: end
    """
    if isinstance(price, pd.DataFrame):
        return price.apply(end_time)
    return price.dropna().index[-1]

def recent_time(time_series: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Slice the time series by recent time.

    Args:
        time_series (pd.DataFrame): time series data.

    Returns:
        pd.DataFrame: recent time series
    """
    if isinstance(time_series, (pd.Series, pd.DataFrame)):
        return time_series.loc[
            offset_time(end_time(time_series).max(), **kwargs):
        ]
    raise TypeError(f'Unsupported type {type(time_series)}')
