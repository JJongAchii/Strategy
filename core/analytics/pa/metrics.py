""" time-series metrics """

from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn import linear_model

from core.strategy import BaseStrategy


def backtest(allocation_df: pd.DataFrame, price_df: pd.DataFrame) -> BaseStrategy:
    """provide a backtested strategy.

    Args:
        allocation_df (pd.DataFrame): allocation dataframe.
        price_df (pd.DataFrame): asset price dataframe.

    Returns:
        BaseStrategy: backtested strategy.
    """

    class Backtest(BaseStrategy):
        """backtest class"""

        def rebalance(self, price_asset: pd.DataFrame) -> pd.Series:
            if self.date in allocation_df.index:
                return allocation_df.index[self.date]
            return pd.Series(dtype=float)

    strategy = Backtest(price_asset=price_df, frequency="D").simulate(
        start=allocation_df.index[0]
    )

    return strategy


def to_pri_return(price_df: pd.DataFrame) -> pd.DataFrame:
    """calculate price return of asset price dataframe

    Args:
        price_df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: price return of asset
    """
    return price_df.pct_change()


def to_log_return(price_df: pd.DataFrame) -> pd.DataFrame:
    """calculate logrithmic return of asset price dataframe

    Args:
        price_df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: logrithmic return of asset
    """
    return to_pri_return(price_df=price_df).apply(np.log1p)


def numofyears(price_df: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        price_df (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    
    return price_df.count() / 252


def ann_factor(price_df: pd.DataFrame) -> pd.Series:
    """calculate annualization factor for price dataframe.

    Args:
        price_df (pd.DataFrame): _description_

    Returns:
        pd.Series: annualization factor.
    """
    return price_df.count() / numofyears(price_df=price_df)


def cum_returns(price_df: pd.DataFrame) -> pd.Series:
    """calculate cumulative returns of price dataframe.

    Args:
        price_df (pd.DataFrame): _description_

    Returns:
        pd.Series: cumulative return
    """
    return to_pri_return(price_df=price_df).add(1).prod()


def ann_returns(price_df: pd.DataFrame) -> pd.Series:
    return cum_returns(price_df=price_df) ** (
        1 / numofyears(price_df=price_df)
    ) - 1


def ann_variances(price_df: pd.DataFrame) -> pd.Series:
    return to_pri_return(price_df=price_df).var() * ann_factor(price_df=price_df)


def ann_volatilities(price_df: pd.DataFrame) -> pd.Series:
    return ann_variances(price_df=price_df) ** 0.5


def ann_semi_variances(price_df: pd.DataFrame) -> pd.Series:
    pri_return_df = to_pri_return(price_df=price_df)
    return pri_return_df[pri_return_df >= 0].var() * ann_factor(price_df=price_df)


def ann_semi_volatilies(price_df: pd.DataFrame) -> pd.Series:
    return ann_semi_variances(price_df=price_df) ** 0.5


def to_drawdown(price_df: pd.DataFrame) -> pd.DataFrame:
    return price_df / price_df.expanding().max() - 1


def max_drawdowns(price_df: pd.DataFrame) -> pd.Series:
    return to_drawdown(price_df=price_df).min()


def expected_returns(price_df: pd.DataFrame, method: str = "empirical") -> pd.Series:

    if method.lower() == "empirical":
        return ann_returns(price_df=price_df)
    raise ValueError(f"method {method} is not supported.")


def covariance_matrix(
    price_df: pd.DataFrame, method: str = "empirical", **kwargs
) -> pd.DataFrame:

    if method.lower() == "empirical":
        return to_pri_return(price_df=price_df).cov() * ann_factor(price_df=price_df)
    if method.lower() == "exponential":
        return to_pri_return(price_df=price_df).ewm(**kwargs).cov().unstack().iloc[
            -1
        ].unstack() * ann_factor(price_df=price_df)
    raise ValueError(f"method {method} is not supported.")


def sharpe_ratios(price_df: pd.DataFrame, risk_free: float = 0.0) -> pd.Series:

    # return (ann_returns(price_df=price_df) - risk_free) / ann_volatilities(
    #     price_df=price_df
    # )
    return (to_pri_return(price_df=price_df).mean() / to_pri_return(price_df=price_df).std() * (252 ** 0.5))


def sortino_ratios(price_df: pd.DataFrame) -> pd.Series:

    return ann_returns(price_df=price_df) / ann_semi_volatilies(price_df=price_df)


def omega_ratios(price_df: pd.DataFrame, required_retrun: float = 0.0) -> pd.Series:

    period_rr = (1 + required_retrun) ** (1 / numofyears(price_df=price_df)) - 1
    pri_return_df = to_pri_return(price_df=price_df)
    return (
        pri_return_df[pri_return_df >= period_rr].sum()
        / pri_return_df[pri_return_df < period_rr].sum()
    )


def calmar_ratio(price_df: pd.DataFrame) -> pd.Series:

    return ann_returns(price_df=price_df) / abs(max_drawdowns(price_df=price_df))


def tail_ratio(price_df: pd.DataFrame, alpha: float = 0.05) -> pd.Series:

    pri_return_df = to_pri_return(price_df=price_df)
    return pri_return_df.quantile(q=alpha) / pri_return_df.quantile(q=1 - alpha)


def skewness(price_df: pd.DataFrame) -> pd.Series:
    return to_pri_return(price_df=price_df).skew()


def kurtosis(price_df: pd.DataFrame) -> pd.Series:
    return to_pri_return(price_df=price_df).kurt()


def value_at_risk(price_df: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    return to_pri_return(price_df=price_df).quantile(q=alpha)


def expected_shortfall(price_df: pd.DataFrame, alpha: float = 0.05) -> pd.Series:

    var = value_at_risk(price_df=price_df, alpha=alpha)
    pri_return_df = to_pri_return(price_df=price_df)
    return pri_return_df[pri_return_df <= var].mean()


def conditional_value_at_risk(price_df: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    return expected_shortfall(price_df=price_df, alpha=alpha)



def parse_views(assets: List, views: List[Dict]):

    num_view = len(views)
    num_asset = len(assets)
    P = np.empty(shape=(num_view, num_asset))
    Q = np.empty(shape=num_view)

    for idx, view in enumerate(views):

        target_asset = np.in1d(assets, view.get("assets")) * 1.0
        target_asset /= target_asset.sum()

        if ">" in view.get("sign"):
            P[idx] = target_asset
            Q[idx] = view.get("value")
        else:
            P[idx] = np.negative(target_asset)
            Q[idx] = -view.get("value")

    return P, Q


def blacklitterman(
    prior_expected_returns: pd.Series,
    prior_covariance_matrix: pd.DataFrame,
    views: List[dict],
    tau: float = 0.05,
    risk_free: float = 0.0,
) -> Tuple[pd.Series, pd.DataFrame]:

    assets = prior_expected_returns.index.tolist()
    P, Q = parse_views(assets=assets, views=views)

    prior_excess_expected_return = prior_expected_returns - risk_free

    scaled_cov = tau * prior_covariance_matrix

    prec_p = scaled_cov @ P.T

    omega = np.diag(np.diag(np.dot(np.dot(P, scaled_cov), P.T)))

    A = P @ prec_p + omega

    post_expected_returns = (
        prior_excess_expected_return
        + scaled_cov @ P.T @ np.linalg.solve(A, Q - P @ prior_excess_expected_return)
    )

    post_covariance_matrix = (
        prior_covariance_matrix
        + scaled_cov
        - np.dot(prec_p, np.linalg.solve(A, prec_p.T))
    )

    return post_expected_returns, post_covariance_matrix


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
        pd.Series(data=list((1 - alpha) **
                  i for i in range(length)), name="weight")
        .iloc[::-1]
        .reset_index(drop=True)
    )
    if adjust:
        out /= out.sum()
    return out



def regression(
    dependent_y: pd.DataFrame,
    independent_x: pd.DataFrame,
    sample_weight: Optional[pd.Series] = None,
    method: str = "linear",
    positive: bool = False,
    fit_intercept: bool = False,
) -> pd.DataFrame:

    if method.lower() == "linear":
        regression_model = linear_model.LinearRegression(
            positive=positive, fit_intercept=fit_intercept,
        )

    elif method.lower() == "lasso":
        regression_model = linear_model.LassoCV(
            positive=positive, fit_intercept=fit_intercept,
            random_state=255,
        )
    else:
        raise NotImplementedError(f"method {method} not implemnted.")

    itx = dependent_y.index.intersection(independent_x.index)
    dependent_y = dependent_y.loc[itx].pct_change().fillna(0)
    independent_x = independent_x.loc[itx].pct_change().fillna(0)

    results = []

    for y in dependent_y:

        result = pd.Series(
            data=regression_model.fit(
                X=independent_x,
                y=dependent_y[y],
                sample_weight=sample_weight,
            ).coef_,
            index=independent_x.columns,
            name=y,
        )

        result["score"] = regression_model.score(
                X=independent_x,
                y=dependent_y[y],
                sample_weight=sample_weight,
            )

        results.append(result)

    return pd.concat(results, axis=1).T


def excess_performance(
    price_1: pd.DataFrame,
    price_2: pd.Series,
) -> pd.DataFrame:
    """calculate excess performance"""
    if not isinstance(price_1, pd.DataFrame):
        price_1 = pd.DataFrame(price_1)
    itx = price_1.index.intersection(price_2.index)
    price_1, price_2 = price_1.loc[itx], price_2.loc[itx]
    pri_return_1 = to_pri_return(price_1).fillna(0)
    pri_return_2 = to_pri_return(price_2).fillna(0)
    er = pri_return_1.subtract(pri_return_2, axis=0)
    return er.add(1).cumprod()


def risk_weighted_performance(price: pd.DataFrame, window: int = 252) -> pd.Series:
    """calculate risk based performance"""
    risk = price.pct_change().rolling(window=window).std().dropna(thresh=2, axis=0)
    weight = risk.divide(risk.sum(axis=1), axis=0)
    return (
        to_pri_return(price.loc[weight.index[0]:])
        .multiply(weight)
        .sum(axis=1)
        .add(1)
        .cumprod()
    )


def expsoures_implied_performance(
    exposures: pd.DataFrame, price_factor: pd.DataFrame
) -> pd.Series:
    """calculate exposure implied performance"""
    idx_date = price_factor.index.intersection(exposures.index)
    idx_factor = price_factor.columns.intersection(exposures.columns)
    if len(idx_date) == 0 or len(idx_factor) == 0:
        return None
    exposures = exposures.loc[idx_date, idx_factor]
    price_factor = price_factor.loc[idx_date, idx_factor]
    pri_return_factor = to_pri_return(price_factor.dropna())
    return (
        pri_return_factor.multiply(exposures)
        .dropna(how="all")
        .sum(axis=1)
        .add(1)
        .cumprod()
    )


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
                dependents=dependents.iloc[i - window: i],
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
