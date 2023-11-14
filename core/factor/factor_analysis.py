import os
import sys
import numpy as np
import pandas as pd
from typing import Optional
from pydantic import BaseModel
from sklearn import linear_model

sys.path.insert(0, os.path.join(os.path.abspath(__file__), "../../.."))
from hive import db


def exponential_weight(
    length: int = 100,
    alpha: float = None,
    com: float = None,
    span: float = None,
    halflife: int = None,
    adjust: bool = False,
) -> pd.Series:
    """
    Calculate exponential weight.

    Args:
        length (int, optional): length of weights. Defaults to 100.
        alpha (float, optional): exponential alpha. Defaults to None.
        com (float, optional): center of mass. Defaults to None.
        span (float, optional): exponential span. Defaults to None.
        halflife (int, optional): exponential halflife decay. Defaults to None.
        adjust (bool, optional): if sum to one. Defaults to False.

    Returns:
        pd.Series: exponential weight.
    """
    if com is not None:
        alpha = 1 / (1 + com)
    elif halflife is not None:
        alpha = 1 - np.exp(-np.log(2) / halflife)
    elif span is not None:
        alpha = 2 / (span + 1)
    elif alpha is None:
        return None
    out = (
        pd.Series(data=list((1 - alpha) ** i for i in range(length)), name="weight")
        .iloc[::-1]
        .reset_index(drop=True)
    )
    if adjust:
        out /= out.sum()
    return out


def exposure(
    dependents: pd.DataFrame,
    independents: pd.DataFrame,
    model: str = "linear",
    window: int = None,
    positive: bool = False,
    fit_intercept: bool = False,
    max_loss: float = 0.2,
    alpha: float = None,
    com: float = None,
    span: float = None,
    halflife: float = None,
) -> pd.DataFrame:
    """
    Calculate asset exposure to factors

    Args:
        dependents (pd.Series): price of asset.
        independents (pd.DataFrame): price of factor
        model (str, optional): regression model. Defaults to 'linear'.
        window (int, optional): rolling window. Defaults to None.
        positive (bool, optional): coefficient constraint. Defaults to False.
        fit_intercept (bool, optional): if fit intercept. Defaults to False.
        max_loss (float, optional): max loss percentage of data.
            Defaults to 0.80.
        alpha (float, optional): exponential alpha. Defaults to None.
        com (float, optional): center of mass. Defaults to None.
        span (float, optional): exponential span. Defaults to None.
        halflife (int, optional): exponential halflife decay. Defaults to None.
        adjust (bool, optional): if sum to one. Defaults to False.

    Returns:
        pd.DataFrame: exposure coefficient
    """

    if isinstance(dependents, pd.DataFrame):

        if window is None:

            return pd.concat(
                objs=list(
                    exposure(
                        dependents=dependents[x],
                        independents=independents,
                        model=model,
                        window=window,
                        max_loss=max_loss,
                        positive=positive,
                        fit_intercept=fit_intercept,
                        alpha=alpha,
                        com=com,
                        span=span,
                        halflife=halflife,
                    )
                    for x in dependents
                ),
                axis=1,
            ).T

        return pd.concat(
            objs=list(
                pd.concat(
                    objs=[
                        exposure(
                            dependents=dependents[x],
                            independents=independents,
                            model=model,
                            window=window,
                            max_loss=max_loss,
                            positive=positive,
                            fit_intercept=fit_intercept,
                            alpha=alpha,
                            com=com,
                            halflife=halflife,
                            span=span,
                        )
                    ],
                    keys=[x],
                    axis=1,
                )
                for x in dependents
            ),
            axis=1,
        )

    if window is not None:

        result = dict()

        for i in range(window, len(dependents) + 1):
            exposures = exposure(
                dependents=dependents.iloc[i - window : i],
                independents=independents,
                model=model,
                max_loss=max_loss,
                positive=positive,
                fit_intercept=fit_intercept,
                window=None,
                alpha=alpha,
                com=com,
                halflife=halflife,
                span=span,
            )
            if exposures is not None:
                result[dependents.index[i - 1]] = exposures

        return pd.DataFrame(result).T

    num = len(dependents)
    itx = dependents.index.intersection(independents.index)
    dependents = dependents.loc[itx]
    independents = independents.loc[itx]

    if len(dependents) / num < (1 - max_loss):
        print("max loss exceeded.")

    if model == "linear":

        regression_model = linear_model.LinearRegression(
            positive=positive, fit_intercept=fit_intercept
        )

    elif model == "lasso":

        regression_model = linear_model.LassoCV(
            positive=positive, fit_intercept=fit_intercept
        )

    elif model == "lasso+linear":

        expo = exposure(
            dependents=dependents,
            independents=independents,
            model="lasso",
            max_loss=max_loss,
            positive=positive,
            fit_intercept=fit_intercept,
            alpha=alpha,
            com=com,
            halflife=halflife,
            span=span,
        )

        expo = expo[expo != 0]

        return exposure(
            dependents=dependents,
            independents=independents.filter(items=expo.index),
            model="linear",
            max_loss=max_loss,
            positive=positive,
            fit_intercept=fit_intercept,
            alpha=alpha,
            com=com,
            halflife=halflife,
            span=span,
        )
    else:
        raise TypeError(f"model {model} not supported.")

    sample_weight = exponential_weight(
        length=len(dependents), alpha=alpha, com=com, halflife=halflife, span=span
    )

    regression_model.fit(
        X=independents,
        y=dependents,
        sample_weight=sample_weight,
    )

    betas = pd.Series(
        data=regression_model.coef_,
        index=independents.columns,
        name=dependents.name,
    )

    betas.loc["intercept"] = regression_model.intercept_

    betas.loc["score"] = regression_model.score(
        X=independents,
        y=dependents,
    )

    return betas


def get_pri_return(
    price: pd.DataFrame, periods: int = 1, forward: bool = False, **kwargs
) -> pd.DataFrame:
    """calculate price return
    
    Args:
        price (pd.DataFrame): price of asset.
        periods (int): interval between prices (default: 1)
        forward (bool): -freq if forward is True
        
    Returns:
        out (pd.DataFrame): percent change of the price
    """
    freq = pd.DateOffset(**kwargs)
    if forward:
        freq = -freq
    _price = price.resample("D").last().ffill()
    out = _price.pct_change(periods=periods, freq=freq).loc[price.index]
    return out


def excess_performance(
    price_1: pd.DataFrame,
    price_2: pd.Series,
) -> pd.DataFrame:
    """calculate excess performance
    Args:
        price_1 (pd.DataFrame): price of non-core factors.
        price_2 (pd.Series): price of a core-factor
        
    Returns:
        (pd.DataFrame): cumulative return of non-core factor value minus core factor value
    
    """
    if not isinstance(price_1, pd.DataFrame):
        price_1 = pd.DataFrame(price_1)
    itx = price_1.index.intersection(price_2.index)
    price_1, price_2 = price_1.loc[itx], price_2.loc[itx]
    pri_return_1 = get_pri_return(price_1).fillna(0)
    pri_return_2 = get_pri_return(price_2).fillna(0)
    er = pri_return_1.subtract(pri_return_2, axis=0)
    
    return er.add(1).cumprod()

def excess_performance2(
    price_1: pd.DataFrame,
    price_2: pd.Series,
    price_factor: pd.DataFrame
) -> pd.DataFrame:
    """calculate excess performance
    
    Args:
        price_1 (pd.DataFrame): price of non-core factors.
        price_2 (pd.Series): price of a core-factor
        
    Returns:
        (pd.DataFrame): cumulative return of non-core factor value minus core factor value
    
    """
    if not isinstance(price_1, pd.DataFrame):
        price_1 = pd.DataFrame(price_1)
    
    itx = price_1.index.intersection(price_2.index)
    price_1, price_2 = price_1.loc[itx], price_2.loc[itx]
    pri_return_1 = get_pri_return(price_1).fillna(0)
    pri_return_2 = get_pri_return(price_2).fillna(0)
    er = pri_return_1.subtract(pri_return_2, axis=0)
    
    return price_factor * er.add(1).cumprod().shift(1).iloc[1:]
    

def risk_weighted_performance(price: pd.DataFrame, window: int = 252) -> pd.Series:
    """calculate risk based performance
    
    Args:
        price (pd.DataFrame): non-core asset price.
        window (int): rolling window
        
    Returns:
        out (pd.DataFrame): risk-weighted cumulative sum of non-core asset return
    
    """
    price = price.loc[price.dropna().index[0]:]
    risk = price.pct_change().rolling(window=window).std().dropna(thresh=2, axis=0)
    weight = risk.divide(risk.sum(axis=1), axis=0)
    return (
        get_pri_return(price.loc[weight.index[0] :])
        .multiply(weight)
        .sum(axis=1)
        .add(1)
        .cumprod()
    )


def exposures_implied_performance(
    exposures: pd.DataFrame, price_factor: pd.DataFrame
) -> pd.Series:
    """calculate exposure implied performance
    
    Args:
        exposures (pd.DataFrame): betas
        price_factor (pd.DataFrame): core price factor
       
    Returns:
        (pd.Series): cumulative return of core factor multiplied by the beta
    
    """
    idx_date = price_factor.index.intersection(exposures.index)
    idx_factor = price_factor.columns.intersection(exposures.columns)
    if len(idx_date) == 0 or len(idx_factor) == 0:
        return None
    exposures = exposures.loc[idx_date, idx_factor]
    price_factor = price_factor.loc[idx_date, idx_factor]
    pri_return_factor = get_pri_return(price_factor.dropna())
    return (
        pri_return_factor.multiply(exposures)
        .dropna(how="all")
        .sum(axis=1)
        .add(1)
        .cumprod()
    )