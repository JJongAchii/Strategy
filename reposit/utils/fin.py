"""
Financial Function
"""
from collections.abc import Iterable

import numpy as np
import pandas as pd

from utils import date
from utils import regr
import database as db

def benchmark(**kwargs):
    """
    Construct benchmark performance

    Returns:
        pd.Series: benchmark performance
    """


    return reindex(
        pd.concat(
            list(
                db.price(key)[key].pct_change().fillna(0) * value
                for key, value in kwargs.items()
            ), axis=1
        ).dropna().sum(axis=1)
    )


def start(price: pd.DataFrame) -> pd.DataFrame:
    """ This is a pass through function """
    return date.start_time(price)


def end(price: pd.DataFrame) -> pd.DataFrame:
    """ This is a pass through function """
    return date.end_time(price)


def compound(
    number: float,
    periods: float
) -> float:
    """
    Calculate compounded value.

    Args:
        number (float): value to be compounded.
        periods (float): compouding periods.

    Returns:
        float: compouded value
    """
    return (1 + number) ** periods - 1


def to_pri_return(
    price: pd.DataFrame,
    periods: int or Iterable = 1,
    freq : pd.DateOffset or None = None,
    forward: bool = False,
    resample_by: str = None,
    binary: bool = False,
) -> pd.DataFrame:
    """
    Calculate price return data.

    Args:
        price (pd.DataFrame): price data.
        periods (int/Iterable): number(s) of periods.
        freq (pd.DateOffset): offset frequency.
        forward (bool, optional): if calculate forward. Defaults to False.
        resample_by (str, optional): resample period of data. Defaults to None.
        binary (bool, optional): if return only binary. Defaults to None.

    Returns:
        pd.DataFrame: price return data
    """
    if isinstance(periods, Iterable):

        result = list()

        for period in periods:

            pri_return = to_pri_return(
                price=price, periods=period, freq=freq,
                resample_by=resample_by, forward=forward, binary=binary,
            )

            if isinstance(pri_return, pd.Series):
                pri_return.name = period
                result.append(pri_return)

            elif isinstance(pri_return, pd.DataFrame):

                pri_return = pd.concat(
                    objs = [pri_return],
                    keys=[period], axis=1
                ).stack()

                pri_return.index.names = ['date', 'ticker']

                result.append(pri_return)

        return pd.concat(result, axis=1)


    if isinstance(price, (pd.Series, pd.DataFrame)):

        if resample_by is not None:
            price = price.resample(rule=resample_by).last()

        price_shift = price.shift(
            periods=periods, freq=freq
        ).dropna(how='all').resample('D').last().ffill().filter(
            items=price.index, axis=0
        )

        if forward:
            price_shift = price.shift(
                periods=-periods, freq=freq
            ).dropna(how='all').resample('D').last().ffill().filter(
                items=price.index, axis=0
            )

            result = price_shift / price - 1
        else:
            price_shift = price.shift(
                periods=periods, freq=freq
            ).dropna(how='all').resample('D').last().ffill().filter(
                items=price.index, axis=0
            )

            result = price / price_shift - 1

        if binary: return result.apply(np.sign)

        return result

    raise TypeError('not supported type.')


def to_log_return(
    price: pd.DataFrame,
    periods: int or Iterable = 1,
    freq : pd.DateOffset or None = None,
    forward: bool = False,
    resample_by: str = None,
) -> pd.DataFrame:
    """
    Calculate log return data.

    Args:
        price (pd.DataFrame): price data.
        periods (int/Iterable): number(s) of periods.
        freq (pd.DateOffset): offset frequency.
        forward (bool, optional): if calculate forward. Defaults to False.
        resample_by (str, optional): resample period of data. Defaults to None.

    Returns:
        pd.DataFrame: log return data
    """
    return to_pri_return(
        price=price,
        periods=periods,
        freq=freq,
        resample_by=resample_by,
        forward=forward,
    ).apply(np.log1p)


def start_price(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate start price

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.Series: start price
    """
    if isinstance(price, pd.DataFrame):
        return price.apply(start_price)
    return price.dropna().iloc[0]


def end_price(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate end price

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.Series: end price
    """
    if isinstance(price, pd.DataFrame):
        return price.apply(end_price)
    return price.dropna().iloc[-1]


def rebase(
    price: pd.DataFrame,
    value: float = 1.
) -> pd.DataFrame:
    """
    Make price start from value.

    Args:
        price (pd.DataFrame): price data.
        value (float, optional): start value. Default to 1.

    Returns:
        pd.DataFrame: rebased price
    """
    if isinstance(price, pd.DataFrame):
        return price.apply(rebase, value=value)
    price = price.divide(start_price(price))
    if value == 0:
        return price.subtract(1)
    return price.multiply(value)


def readjust(
    price: pd.DataFrame, value: float = 1
) -> pd.DataFrame:
    """
    Make price end at value.

    Args:
        price (pd.DataFrame): price data.
        value (float, optional): start value. Default to 1.

    Returns:
        pd.DataFrame: readjusted price
    """
    if isinstance(price, pd.DataFrame):
        return price.apply(readjust, value=value)
    price = price.divide(end_price(price))
    if value == 0:
        return price.subtract(1)
    return price.multiply(value)


def reindex(
    pri_return: pd.DataFrame,
) -> pd.DataFrame:
    """
    Make price return into price index.

    Args:
        pri_return (pd.DataFrame): price return data.

    Returns:
        pd.DataFrame: price return data
    """
    return pri_return.add(1).cumprod()


def num_year(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate number of years.

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.Series: num year
    """
    if isinstance(price, pd.DataFrame):
        return price.apply(num_year)
    return (end(price) - start(price)).days / 365.


def num_data(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate number of data.

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.Series: num data
    """
    if isinstance(price, pd.DataFrame):
        return price.apply(num_data)
    return len(price.dropna())


def frequency(
    price: pd.Series
) -> float:
    """
    Calculate frequency of price (yearly)

    Args:
        price (pd.Series): price data.

    Returns:
        float: frequency
    """
    return num_data(price) / num_year(price)


def drawdown(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate drawdown.

    Args:
        price (pd.Series): price data.

    Returns:
        pd.Series: drawdown
    """
    return price.divide(price.expanding().max()) - 1


def max_drawdown(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate max drawdown.

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.Series: max drawdown
    """
    return drawdown(price).min().abs()


def cumulative_return(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate cumulative return

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.Series: cumulative return
    """
    return end_price(price) / start_price(price) - 1


def annualized_return(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate annualized return

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.Series: cumulative return
    """
    return compound(cumulative_return(price), 1/num_year(price))


def annualized_variance(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate annualized variance.

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.Series: annualized variance
    """
    return to_pri_return(price).var() * frequency(price)


def annualized_risk(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate annualized risk.

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.Series: annualized risk
    """
    return annualized_variance(price) ** 0.5


def annualized_semi_variance(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate annualized semi variance.

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.Series: annualized semi variance
    """
    pri_return = to_pri_return(price)
    return pri_return[pri_return < 0].var() * frequency(price)


def annualized_semi_risk(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate annualized semi risk.

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.Series: annualized semi risk
    """
    return annualized_semi_variance(price) ** 0.5


def moving_average(
    price: pd.DataFrame,
    window: int = 21
) -> pd.DataFrame:
    """
    Calculate moving average.

    Args:
        price (pd.Series): price data.
        window (int, optional): window size. Defaults to 21.

    Returns:
        pd.DataFrame: moving average
    """
    if isinstance(price, pd.DataFrame):
        return price.apply(moving_average, window=window)
    return price.dropna().rolling(window).mean()


def moving_average_crossover(
    price: pd.DataFrame,
    window1: int = 21,
    window2: int = 63
) -> pd.DataFrame:
    """
    Calculate moving average crossover.

    Args:
        price (pd.DataFrame): price data.
        window1 (int, optional): short length of window. Defaults to 21.
        window2 (int, optional): long length of window. Defaults to 63.

    Returns:
        pd.DataFrame: moving average crossover
    """
    ma2 = moving_average(price, window2)
    return moving_average(price, window1).subtract(ma2).divide(ma2)


def excess_return(
    pri_return: pd.DataFrame,
    pri_return_bm: pd.Series = None,
    required_return: float = 0.0
) -> pd.DataFrame:
    """
    Calculate excess return.

    Args:
        pri_return (pd.DataFrame): price return data.
        pri_return_bm (pd.Series, optional): price return benchmark.
        required_return (float, optional): required annualized return.

    Returns:
        pd.DataFrame: excess return
    """
    if isinstance(pri_return, pd.DataFrame):
        return pri_return.apply(
            excess_return,
            pri_return_bm=pri_return_bm,
            required_return=required_return
        )

    if pri_return_bm is None:
        return pri_return.subtract(
            compound(required_return, 1/frequency(pri_return))
        )

    start = max(pri_return.index[0], pri_return_bm.index[0])
    end = min(pri_return.index[-1], pri_return_bm.index[-1])

    pri_return = pri_return.loc[start:end]

    pri_return_bm = pri_return_bm.reindex(
        pd.date_range(start=start, end=end)
    ).fillna(0)

    return pri_return.subtract(pri_return_bm)


def excess_performance(
    price: pd.DataFrame,
    price_bm: pd.Series = None,
    required_return: float = 0.0
) -> pd.DataFrame:
    """
    Calculate excess performance.

    Args:
        price (pd.DataFrame): price data.
        price_bm (pd.Series, optional): price benchmark.
        required_return (float, optional): required annualized return.

    Returns:
        pd.DataFrame: excess performance
    """
    return reindex(
        excess_return(
            pri_return=to_pri_return(price),
            pri_return_bm=None if price_bm is None else to_pri_return(price_bm),
            required_return=required_return
        )
    )


def sharpe_ratio(price: pd.DataFrame, risk_free: float = 0.00) -> pd.Series:
    """
    Calculate sharpe ratio.

    The greater Sharpe ratio, the better its risk-adjusted performance.

    If the analysis results in a negative Sharpe ratio, it either means the
    risk-free rate is greater than the portfolio's return, or the portfolio's
    return is expected to be negative.

    Args:
        price (pd.DataFrame): price data.
        risk_free (float, optional): risk free rate. Defaults to 0.02.

    Returns:
        pd.Series: sharpe ratio
    """

    pri_return = to_pri_return(price, periods=1)
    er = excess_return(pri_return=pri_return, required_return=risk_free)
    return er.mean() / er.std() * np.sqrt(frequency(price))

def omega_ratio(
    price: pd.DataFrame,
    risk_free: float = 0.,
    required_return: float = 0.
) -> pd.Series:
    """
    Calculate omega ratio.

    Omega is the probability-weighted ratio of gains over losses at a given
    level of expected return (known as threshold).

    Advantages:
    1. it is designed to encapsulate all the information about the risk and
    return of a portfolio that is contained within its return distribution,
    thus redressing the shortcomings discussed above.

    2. its precise value is directly determined by investor's risk appetite.

    Args:
        price (pd.DataFrame): price data.
        risk_free (float, optional): risk free rate. Defaults to 0.02.
        required_return (float, optional): required return. Defaults to 0.

    Returns:
        pd.Series: omega ratio
    """

    pri_return = to_pri_return(price, periods=1)
    er = excess_return(pri_return=pri_return, required_return=risk_free)
    return er[er > required_return].sum() / -er[er < required_return].sum()

def sortino_ratio(
    price: pd.DataFrame,
    risk_free: float = 0.02
) -> pd.Series:
    """
    Calculate sortino ratio.

    A higher Sortino ratio is better than a lower one as it indicates that the
    portfolio is operating efficiently by not taking on unnecessary risk that
    is not being rewarded in the form of higher returns.
    A low, or negative, Sortino ratio may suggest that the
    investor is not being rewarded for taking on additional risk.

    Args:
        price (pd.DataFrame): price data.
        risk_free (float, optional): risk free rate. Defaults to 0.02.

    Returns:
        pd.Series: sortino ratio
    """
    pri_return = to_pri_return(price, periods=1)
    er = excess_return(pri_return=pri_return, required_return=risk_free)
    return er.mean() / er[er >= 0].std() * np.sqrt(frequency(price))


def calmar_ratio(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate calmar ratio.

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.Series: calmar ratio
    """
    return annualized_return(price) / max_drawdown(price)


def VaR(
    price: pd.DataFrame,
    confidence_level=0.95
) -> pd.Series:
    """
    Calculate Value at Risk (historical)

    Args:
        price (pd.DataFrame): price data.
        confidence_level (float, optional): confidence. Defaults to 0.95.

    Returns:
        pd.Series: Value at Risk
    """
    return to_pri_return(price).quantile(1 - confidence_level)


def CVaR(
    price: pd.DataFrame,
    confidence_level=0.95
) -> pd.Series:
    """
    Calculate Conditional Value at Risk (historical)

    Args:
        price (pd.DataFrame): price data.
        confidence_level (float, optional): confidence. Defaults to 0.95.

    Returns:
        pd.Series: Conditional Value at Risk
    """
    pri_return = to_pri_return(price)
    var = VaR(price=price, confidence_level=confidence_level)
    return pri_return[pri_return <= var].mean()


def skewness(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate skewness.

    If skewness is positive, the data are positively skewed or skewed right,
    meaning that the right tail of the distribution is longer than the left.
    If skewness is negative, the data are negatively skewed or skewed left,
    meaning that the left tail is longer.
    If skewness is less than -1 or greater than +1, the distribution can be
    called highly skewed.
    If skewness is between -1 and -0.5 or between +0.5 and +1, the
    distribution can be called moderately skewed.
    If skewness is between -0.5 and +0.5, the distribution can be called
    approximately symmetric.
    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.Series: skewness
    """
    return to_pri_return(price).skew()


def kurtosis(
    price: pd.DataFrame
) -> pd.Series:
    """
    Calculate kurtosis.

    higher kurtosis means more of the variance is the result of infrequent
    extreme deviations, as opposed to frequent modestly sized deviations.
    A normal distribution has kurtosis exactly 3 (excess kurtosis exactly 0).
    Any distribution with kurtosis ≈ 3 (excess ≈ 0) is called mesokurtic.
    A distribution with kurtosis <3 (excess kurtosis <0) is called platykurtic.
    Compared to a normal distribution, its tails are shorter and thinner, and
    often its central peak is lower and broader.
    A distribution with kurtosis >3 (excess kurtosis >0) is called leptokurtic.
    Compared to a normal distribution, its tails are longer and fatter, and
    often its central peak is higher and sharper.

    Args:
        price (pd.DataFrame): price data.

    Returns:
        pd.Series: kurtosis
    """
    return to_pri_return(price).kurt()


def tail_ratio(
    price: pd.DataFrame,
    alpha=0.05
) -> pd.Series:
    """
    Calculate tail ratio.

    Args:
        price (pd.DataFrame): price data.
        alpha (float, optional): alpha. Defaults to 0.05.

    Returns:
        pd.Series: tail ratio
    """
    pri_return = to_pri_return(price)
    return pri_return.quantile(q=(1-alpha)) / -pri_return.quantile(q=(alpha))


def absolute_momentum(
    price: pd.DataFrame,
    periods: int or Iterable = 1,
    freq : pd.DateOffset or None = None,
    forward: bool = False,
    resample_by: str = None,
    binary: bool = False,
) -> pd.DataFrame:
    """ This is a pass through function """
    return to_pri_return(
        price=price,
        periods=periods, freq=freq,
        forward=forward, resample_by=resample_by,
        binary=binary
    )


def excess_over_median(
    time_series: pd.DataFrame,
    axis: int = 0,
) -> pd.DataFrame:
    """
    Calculate excess over median.

    Args:
        time_series (pd.DataFrame): time series data.
        axis (int, optional): axis of scaler transfomation.

    Returns:
        pd.DataFrame: excess over median
    """
    if isinstance(time_series, pd.DataFrame):
        return time_series.apply(excess_over_median, axis=axis)
    return time_series.subtract(time_series.median())


def excess_over_mean(
    time_series: pd.DataFrame,
    axis: int = 0,
) -> pd.DataFrame:
    """
    Calculate excess over mean.

    Args:
        time_series (pd.DataFrame): time series data.
        axis (int, optional): axis of scaler transfomation.

    Returns:
        pd.DataFrame: excess over mean
    """
    if isinstance(time_series, pd.DataFrame):
        return time_series.apply(excess_over_mean, axis=axis)
    return time_series.subtract(time_series.mean())


def z_score_filter(
    time_series: pd.DataFrame,
    mean_window: int = 252,
    std_window: int = 252,
    z_score: int = 3
) -> pd.DataFrame:
    """
    Filter time series with zscore.

    Args:
        time_series (pd.DataFrame): _description_
        mean_window (int, optional): _description_. Defaults to 252.
        std_window (int, optional): _description_. Defaults to 252.
        z_score (int, optional): _description_. Defaults to 3.

    Returns:
        pd.DataFrame: _description_
    """
    if isinstance(time_series, pd.DataFrame):
        return time_series.apply(
            z_score_filter, mean_window=mean_window,
            std_window=std_window, z_score=z_score
        )
    mean = time_series.rolling(window=mean_window).mean()
    std = time_series.rolling(window=std_window).std()
    return time_series[time_series >= mean + z_score * std]


def standard_scaler(
    time_series: pd.DataFrame,
    axis: int = 0,
    assume_centered: bool = False,
) -> pd.DataFrame:
    """
    Transform a time series into standard scaler.

    Args:
        time_series (pd.DataFrame): time series data.
        axis (int, optional): axis of scaler transfomation.
        assume_centered (bool, optional): if assume data is centered at zero.
        Default is False.

    Returns:
        pd.DataFrame: standard scaler
    """
    if isinstance(time_series, pd.DataFrame):
        return time_series.apply(standard_scaler, axis=axis)
    ts = time_series.dropna()
    mean = 0. if assume_centered else ts.mean()
    scaler = (ts - mean) / ts.std()
    return scaler.clip(lower=-3, upper=3)


def min_max_scaler(
    time_series: pd.DataFrame,
    axis: int = 0,
) -> pd.DataFrame:
    """
    Transform a time series into min max scaler.

    Args:
        time_series (pd.DataFrame): time series data.
        axis (int, optional): axis of scaler transfomation.

    Returns:
        pd.DataFrame: min max scaler
    """
    if isinstance(time_series, pd.DataFrame):
        return time_series.apply(min_max_scaler, axis=axis)
    minimum, maximum = time_series.min(), time_series.max()
    scaler = (time_series - minimum) / (maximum - minimum)
    return scaler.dropna()


def auto_correlation(
    time_series: pd.DataFrame,
    axis: int = 0,
    lags: int = 50,
) -> pd.DataFrame:
    """
    Calculate auto correlation.

    Args:
        time_series (pd.DataFrame): time series data.
        axis (int, optional): axis of scaler transfomation.
        lags (int, optional): lags. Defaults to 50.

    Returns:
        pd.Series: auto autocorrelation
    """
    if isinstance(time_series, pd.DataFrame):
        return time_series.apply(
            auto_correlation, axis=axis, lags=lags
        )
    ts = time_series.dropna()
    return pd.Series([1] + [
            np.corrcoef(ts[:-i], ts[i:])[0, 1]
            for i in range(1, lags + 1)
        ]
    )


def backtest(
    price: pd.DataFrame,
    weight: pd.DataFrame,
    freq: ... = 'M',
    verbose: bool = False,
    commission: int = 0,
    start: ... = None,
    **kwargs
) -> ...:
    """
    Calculate performance based on pre-defined allocation weight.

    Args:
        price (pd.DataFrame): price data.
        weight (pd.DataFrame): allocation weight.
        freq (M, optional): rebalancing frequency. Defaults to 'M'.
        verbose (bool, optional): if verbose. Defaults to False.
        commission (int, optional): commission in bases points. Defaults to 0.
        start (None, optional): start date. Defaults to None.

    Returns:
        BaseStrategy: base strategy class.
    """
    try:
        from strategy.base_strategy import BaseStrategy
    except ImportError as exception:
        print(exception)

    weight = weight.resample('D').last().ffill()

    class BackTest(BaseStrategy):
        """ this is a pass through class """
        def rebalance(self, price_asset, **kwargs):
            """ Pass through function """
            return weight.loc[kwargs.get('date')]

    bt = BackTest(
        verbose=verbose,
        commission=commission
    ).fit(
        price_asset=price,
        freq=freq,
        start=start or weight.index[0],
        **kwargs,
    )
    return bt


def factor_exposure(
    price: pd.DataFrame,
    price_factor: pd.DataFrame,
    model: str = 'linear',
    window: int = None,
    min_fraction: float = 0.95,
    custom_weight: str = None,
    positive: bool = True,
    fit_intercept: bool = False,
    label_score: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Calculate factor exposure.

    Args:
        pri_return (pd.DataFrame): return data.
        pri_return_factor (pd.DataFrame): factor return data.
        model (str, optional): regression model used. Defaults to linear.
            allowed: linear, lasso.
        window (int, optional): rolling window. Defualts to None.
        min_fraction (float, optional): minimum percent of data points against
            the total window. Defaults to 0.90.
        custom_weight (str, optional): regression weights. Defaults to None.
        positive (bool, optional): if positve regression coefficients.
            Defaults to False.
        fit_intercept (bool, optional): fit y-intercept. Defaults to True.
        label_score (bool, optional): if label r2 score in result.
            Defaults to False.

    Kwargs:
        parameters to pass to the model.

    Returns:
        pd.DataFrame: factor exposure.
    """

    price, price_factor = regr.align_price(
        dependent=price,
        independents=price_factor,
    )

    return regr.regression(
        dependent=to_pri_return(price).iloc[1:],
        independents=to_pri_return(price_factor).iloc[1:],
        model=model,
        window=window,
        min_fraction=min_fraction,
        custom_weight=custom_weight,
        positive=positive,
        fit_intercept=fit_intercept,
        label_score=label_score,
        **kwargs
    )


def exposures_implied_performance(
    exposures: pd.DataFrame,
    price_factor: pd.DataFrame,
) -> pd.Series:
    """
    Calculate implied performance based on beta exposure and
    factor price return.

    Args:
        beta (pd.DataFrame): beta exposure.
        pri_return_factor (pd.DataFrame): factor price return.

    Returns:
        pd.Series: implied performance
    """

    pri_return_factor = price_factor.pct_change()

    if isinstance(exposures, pd.DataFrame):
        idx_date = pd.Index.intersection(
            exposures.index,
            pri_return_factor.index,
        )

        idx_factor = pd.Index.intersection(
            exposures.columns,
            pri_return_factor.columns
        )

        if len(idx_date) == 0 or len(idx_factor) == 0: return None

        exposures = exposures.loc[idx_date, idx_factor]
        pri_return_factor = pri_return_factor.loc[idx_date, idx_factor]

    implied_pri_return = pri_return_factor.multiply(
        exposures
    ).dropna(how='all').sum(axis=1)

    return reindex(implied_pri_return)


def factor_implied_performance(
    price: pd.Series,
    price_factor: pd.DataFrame,
    model: str = 'linear',
    window: int = None,
    min_fraction: float = 0.95,
    custom_weight: str = None,
    positive: bool = True,
    fit_intercept: bool = False,
    **kwargs
) -> pd.Series:
    """
    Calculate factor exposure.

    Args:
        pri_return (pd.Series): return data.
        pri_return_factor (pd.DataFrame): factor return data.
        model (str, optional): regression model used. Defaults to linear.
            allowed: linear, lasso.
        window (int, optional): rolling window. Defualts to None.
        min_fraction (float, optional): minimum percent of data points against
            the total window. Defaults to 0.90.
        custom_weight (str, optional): _description_. Defaults to None.
        positive (bool, optional): if positve regression coefficients.
            Defaults to False.
        fit_intercept (bool, optional): fit y-intercept. Defaults to True.
        label_score (bool, optional): if label r2 score in result.
            Defaults to False.

    KwArgs:
        parameters to pass to the model.

    Returns:
        pd.DataFrame: factor exposure.
    """
    exposures = factor_exposure(
        price=price,
        price_factor=price_factor,
        model=model,
        window=window,
        min_fraction=min_fraction,
        custom_weight=custom_weight,
        positive=positive,
        fit_intercept=fit_intercept,
        label_score=False,
        **kwargs
    )

    return exposures_implied_performance(
        exposures = exposures,
        price_factor=price_factor,
    )


def manager_behavior(
    price: pd.DataFrame,
    price_factor: pd.DataFrame,
    static_window: int = 252*5,
    active_window: int = 252*3,
    model: str = 'lasso',
) -> tuple:
    """
    Calculate manager performance.

    Args:
        pri_return (pd.DataFrame): _description_
        pri_return_factor (pd.DataFrame): _description_
        static_window (int, optional): _description_. Defaults to 252*5.
        active_window (int, optional): _description_. Defaults to 252*3.

    Returns:
        tuple: _description_
    """

    static_exposures = factor_exposure(
        price=price, price_factor=price_factor,
        window=static_window,
        model=model,
    ).clip(upper=1)

    static_performance = exposures_implied_performance(
        exposures = static_exposures,
        price_factor=price_factor,
    )

    active_exposures = factor_exposure(
        price=price, price_factor=price_factor,
        window=active_window,
        model=model,
    ).clip(upper=1)

    active_performance = exposures_implied_performance(
        exposures = active_exposures,
        price_factor=price_factor,
    )

    factor_timing = excess_performance(
        active_performance,
        static_performance,
    ).dropna()

    factor_timing.name = 'factor_timing'

    stock_selection = excess_performance(
        price, active_performance,
    ).dropna()

    stock_selection.name = 'stock_selection'

    return rebase(
        pd.concat(objs=[factor_timing, stock_selection], axis=1).dropna()
    )


def residualization(
    price: pd.Series,
    price_factor: pd.DataFrame,
    window: int = 252*3,
    custom_weight: str = 'exponential',
    window_smoothing: int = 5,
) -> pd.Series:
    """
    Calculate residualized performance.

    Args:
        pri_return (pd.Series): price return data.
        pri_return_factor (pd.DataFrame): price return factor data.
        window (int, optional): rolling window. Defaults to 252*3.
        custom_weight (str, optional): custom weighting method.
            Defaults to 'exponential'.
        window_smoothing (int, optional): exposures smoothing window.
            Defaults to 5.

    Returns:
        pd.Series: residualized performance
    """
    exposures = factor_exposure(
        price=price,
        price_factor=price_factor,
        model='linear',
        custom_weight=custom_weight,
        window=window,
        positive=True,
        fit_intercept=False,
    )

    if window_smoothing is not None:

        exposures = exposures.rolling(
            window=window_smoothing
        ).mean()

    implied_perfromance = exposures_implied_performance(
        exposures=exposures,
        price_factor=price_factor
    )

    return excess_performance(
        price=price,
        price_bm=implied_perfromance
    )



def fwd_monthly_pri_return(price: pd.DataFrame) -> pd.DataFrame:
    """ this is a pass through function"""
    return to_pri_return(price, forward=True, resample_by='M')


def fwd_1m_pri_return(price: pd.DataFrame) -> pd.DataFrame:
    """ this is a pass through function"""
    return to_pri_return(
        price=price, forward=True, periods=1, freq=pd.DateOffset(months=1)
    )


def fwd_monthly_excess_return(price: pd.DataFrame) -> pd.DataFrame:
    """ this is a pass through function"""
    return excess_over_median(fwd_monthly_pri_return(price=price))


def fwd_1m_excess_return(price: pd.DataFrame) -> pd.DataFrame:
    """ this is a pass through function"""
    return excess_over_median(fwd_1m_pri_return(price=price))


def rolling_risk(
    pri_return: pd.DataFrame,
    window: int = 252,
):
    if isinstance(pri_return, pd.DataFrame):
        return pri_return.apply(
            rolling_risk,
            window=window
        )

    return pri_return.dropna().rolling(window).std()


def risk_weighted_performance(
    price: pd.DataFrame,
    window: int = 252,
) -> pd.Series:

    pri_return = to_pri_return(price)
    risk = pri_return.rolling(window).var().ffill().dropna(how='all')
    weight = risk.divide(risk.sum(axis=1), axis=0)
    return reindex(pri_return.multiply(weight).dropna(how='all').sum(axis=1))
