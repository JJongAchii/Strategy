import os
import sys
import pandas as pd
import numpy as np
from sys import float_info
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../..")))


def ta_save(func):
    ta_result = {}

    def wrapper(close, length):
        result = func(close, length)
        
        params = f"{func.__name__}_{length}"
        ta_result[params] = result
        
        return ta_result
    
    return wrapper


def non_zero_range(high: pd.Series, low: pd.Series) -> pd.Series:
    """Returns the difference of two series and adds epsilon to any zero values.  This occurs commonly in crypto data when 'high' = 'low'."""
    diff = high - low
    if diff.eq(0).any().any():
        diff += float_info.epsilon
    return diff


def bbands(
    close: pd.DataFrame,
    length: int = 5,
    std: float = 2.0,
    offset: int = 0
) -> pd.DataFrame:
    """"""
    # Implement Date-Based close Sorting
    close = close.sort_index()
    
    standard_deviation = close.rolling(length, min_periods=length).std()
    deviation = std * standard_deviation

    mid = sma(close=close, length=length)
    lower = mid - deviation
    upper = mid + deviation
    
    ulr = non_zero_range(upper, lower)
    bandwidth = 100 * ulr / mid
    percent = non_zero_range(close, lower) / ulr
    
    return (
        lower.shift(offset), 
        mid.shift(offset), 
        upper.shift(offset), 
        bandwidth.shift(offset), 
        percent.shift(offset)
    )


def sma(
    close: pd.DataFrame,
    length: int = 10,
    offset: int = 0
) -> pd.DataFrame:
    """
    Indicator: Simple Moving Average (SMA)

    Args:
        close (pd.DataFrame): _description_
        length (int, optional): _description_. Defaults to 10.
        offset (int, optional): _description_. Defaults to 0.

    Returns:
        pd.DataFrame: _description_
    """

    # Implement Date-Based close Sorting
    close = close.sort_index()
    
    # Calculate Result
    sma = close.rolling(length, min_periods=length).mean()

    return sma.shift(offset)
    
    
def ema(
    close: pd.DataFrame, 
    length: int = 10, 
    offset: int = 0
) -> pd.DataFrame:
    """
    Indicator: Exponential Moving Average (EMA)

    Args:
        close (pd.DataFrame): _description_
        length (int, optional): _description_. Defaults to 10.
        offset (int, optional): _description_. Defaults to 0.

    Returns:
        pd.DataFrame: _description_
    """
    
    # Implement Date-Based close Sorting
    close = close.sort_index()
    
    # Calculate Result
    ema = close.ewm(span=length, min_periods=length).mean()

    return ema.shift(offset)


def macd(
    close: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    offset: int = 0
) -> pd.DataFrame:
    """
    Indicator: Moving Average, Convergence/Divergence (MACD)

    Args:
        close (pd.DataFrame): _description_
        fast (int, optional): _description_. Defaults to 12.
        slow (int, optional): _description_. Defaults to 26.
        signal (int, optional): _description_. Defaults to 9.
        offset (int, optional): _description_. Defaults to 0.

    Returns:
        pd.DataFrame: _description_
    """
    
    # Implement Date-Based close Sorting
    close = close.sort_index()
    
    if fast > slow:
        fast, slow = slow, fast
        
    # Calculate Result
    fastma = ema(close=close, length=fast)
    slowma = ema(close=close, length=slow)

    macd = fastma - slowma
    signalma = ema(close=macd.loc[macd.first_valid_index():,], length=signal)
    histogram = macd - signalma
    
    return macd.shift(offset), signalma.shift(offset), histogram.shift(offset)
    

def rsi(
    close: pd.DataFrame,
    length: int = 14,
    drift: int = 1,
    offset: int = 0
) -> pd.DataFrame:
    """
    Indicator: Relative Strength Index (RSI)

    Args:
        close (pd.DataFrame): _description_
        length (int, optional): _description_. Defaults to 14.
        drift (int, optional): _description_. Defaults to 1.
        offset (int, optional): _description_. Defaults to 0.

    Returns:
        pd.DataFrame: _description_
    """
    
    # Implement Date-Based close Sorting
    close = close.sort_index()
    
    # Calculate Result
    close_diff = close.diff(drift)
    
    positive = close_diff.copy()
    negative = close_diff.copy()
    positive[positive < 0] = 0  # Make negatives 0 for the postive series
    negative[negative > 0] = 0  # Make postives 0 for the negative series

    positive_avg = positive.ewm(com=length, min_periods=length).mean()
    negative_avg = negative.ewm(com=length, min_periods=length).mean()
    
    rsi = 100 * positive_avg / (positive_avg + negative_avg.abs())
    
    return rsi.shift(offset)


def stoch(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
    offset: int = 0
) -> pd.DataFrame:
    """
    Indicator: Stochastic Oscillator (STOCH)

    Args:
        close (pd.DataFrame): _description_
        high (pd.DataFrame): _description_
        low (pd.DataFrame): _description_
        k (int, optional): _description_. Defaults to 14.
        d (int, optional): _description_. Defaults to 3.

    Returns:
        pd.DataFrame: _description_
    """
    
    # Implement Date-Based close Sorting
    close = close.set_index()
    high = high.set_index()
    low = low.set_index()
    
    # Calculate Result
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    
    stoch = 100 * (close - lowest_low)
    stoch /= non_zero_range(highest_high, lowest_low)

    stoch_k = sma(close=stoch.loc[stoch.first_valid_index():,], length=smooth_k)
    stoch_d = sma(close=stoch_k.loc[stoch_k.first_valid_index():,], length=d)

    return stoch_k.shift(offset), stoch_d.shift(offset)


def vwma(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    length: int = 10,
    offset: int = 0
) -> pd.DataFrame:
    """
    Indicator: Volume Weighted Moving Average (VWMA)

    Args:
        close (pd.DataFrame): _description_
        volume (pd.DataFrame): _description_
        length (int, optional): _description_. Defaults to 10.
        offset (int, optional): _description_. Defaults to 0.

    Returns:
        pd.DataFrame: _description_
    """
    
    # Implement Date-Based close Sorting
    close = close.sort_index()
    volume = volume.sort_index()
    
    # Calculate Result
    pv = close * volume
    vwma = sma(close=pv, length=length) / sma(close=volume, length=length)

    return vwma.shift(offset)


if __name__ == "__main__":
    
    from hive import db
    
    with db.session_local() as session:
        query = (
            session.query(
                db.TbDailyBar
            )
            .filter(db.TbDailyBar.stk_id == 1)
        )
        price = db.read_sql_query(query=query)

    price = price.set_index("trd_dt")["adj_value"]

    # with db.session_local() as session:
    #     query = (
    #         session.query(
    #             db.TbMetaUpdat
    #         )
    #         .filter(db.TbMetaUpdat.stk_id == 1)
    #     )
    #     volume = db.read_sql_query(query=query)
    
    # volume = volume.set_index("trd_dt")["volume"]
    
    data = bbands(close=price)
    
    print(data)
    
    import talib
    ta = talib.BBANDS(price.sort_index())
    print("ta-lib-----------------")
    print(ta)
    
    
    