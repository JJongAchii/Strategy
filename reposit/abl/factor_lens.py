import os
import sys
from typing import Optional
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn import linear_model



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
    """calculate price return"""
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
    """calculate excess performance"""
    if not isinstance(price_1, pd.DataFrame):
        price_1 = pd.DataFrame(price_1)
    itx = price_1.index.intersection(price_2.index)
    price_1, price_2 = price_1.loc[itx], price_2.loc[itx]
    pri_return_1 = get_pri_return(price_1).fillna(0)
    pri_return_2 = get_pri_return(price_2).fillna(0)
    er = pri_return_1.subtract(pri_return_2, axis=0)
    return er.add(1).cumprod()


def risk_weighted_performance(price: pd.DataFrame, window: int = 252) -> pd.Series:
    """calculate risk based performance"""
    risk = price.pct_change().rolling(window=window).std().dropna(thresh=2, axis=0)
    weight = risk.divide(risk.sum(axis=1), axis=0)
    return (
        get_pri_return(price.loc[weight.index[0] :])
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
    pri_return_factor = get_pri_return(price_factor.dropna())
    return (
        pri_return_factor.multiply(exposures)
        .dropna(how="all")
        .sum(axis=1)
        .add(1)
        .cumprod()
    )


class FactorLensSettings(BaseModel):

    rate: str = "H04792US"  ### <- LGY7TRUH Index replace
    equity: str = "MXCXDMHR"
    uscredit: str = "LUACTRUU"
    eucredit: str = "LP05TRUH"
    usjunk: str = "LF98TRUU"
    eujunk: str = "LP01TRUH"
    commodity: str = "BCOMTR"
    localequity: str = "SPXT"
    shortvol: str = "PUT"
    localinflation: str = "BCIT5T"
    currency: str = "DXY"
    emergingequity: str = "M1EF"
    emergingbond: str = "EMUSTRUU"
    developedequity: str = "M1WD"
    developedbond: str = "LEGATRUU"
    momentum: str = "M1WD000$"
    value: str = "M1WD000V"
    growth: str = "M1WD000G"
    lowvol: str = "M1WDMVOL"
    smallcap: str = "M1WDSC"
    quality: str = "M1WDQU"


class FactorLens:
    def __init__(self, data: pd.DataFrame) -> None:

        self.data = data
        self.rate: Optional[pd.Series(dtype=float)] = None
        self.equity: Optional[pd.Series(dtype=float)] = None
        self.credit: Optional[pd.Series(dtype=float)] = None
        self.commodity: Optional[pd.Series(dtype=float)] = None
        self.emerging: Optional[pd.Series(dtype=float)] = None
        self.currency: Optional[pd.Series(dtype=float)] = None
        self.localequity: Optional[pd.Series(dtype=float)] = None
        self.localinflation: Optional[pd.Series(dtype=float)] = None
        self.shortvol: Optional[pd.Series(dtype=float)] = None
        self.momentum: Optional[pd.Series(dtype=float)] = None
        self.value: Optional[pd.Series(dtype=float)] = None
        self.growth: Optional[pd.Series(dtype=float)] = None
        self.smallcap: Optional[pd.Series(dtype=float)] = None
        self.lowvol: Optional[pd.Series(dtype=float)] = None
        self.process()

    def process(self) -> None:

        self.rate = self.data.rate.dropna()
        self.equity = self.data.equity.dropna()
        credit_performance = risk_weighted_performance(
            self.data[["uscredit", "eucredit", "usjunk", "eujunk"]].dropna()
        )
        self.credit = self.residualization(
            price=credit_performance,
            price_factor=pd.concat([self.rate, self.equity], axis=1).dropna(),
        ).squeeze()
        self.credit.name = "credit"

        self.commodity = self.residualization(
            price=self.data.commodity,
            price_factor=pd.concat([self.rate, self.equity], axis=1).dropna(),
        )

        self.commodity.name = "commodity"

        core_macro = pd.concat(
            objs=[self.rate, self.equity, self.credit, self.commodity], axis=1
        ).dropna()

        em_equity = excess_performance(
            self.data.emergingequity.dropna(), self.data.developedequity.dropna()
        )

        em_bond = excess_performance(
            self.data.emergingbond.dropna(), self.data.developedbond.dropna()
        )

        emerging = (
            risk_weighted_performance(pd.concat([em_equity, em_bond], axis=1).dropna())
            .dropna()
            .loc[core_macro.index[0] :]
        )

        self.emerging = self.residualization(
            price=emerging, price_factor=core_macro
        ).squeeze()
        self.emerging.name = "emerging"

        self.localinflation = self.residualization(
            price=self.data.localinflation.dropna().loc[core_macro.index[0] :],
            price_factor=core_macro,
        )

        self.localinflation.name = "localinflation"

        self.localequity = self.residualization(
            price=self.data.localequity.dropna().loc[self.equity.dropna().index[0] :],
            price_factor=self.equity.to_frame(),
        )
        self.localequity.name = "localequity"

        self.shortvol = self.residualization(
            price=self.data.shortvol.dropna().loc[self.equity.index[0] :],
            price_factor=self.equity.to_frame(),
        )
        self.shortvol.name = "shortvol"

        ccy = self.data.currency.resample("D").last().loc[core_macro.index]

        self.currency = self.residualization(price=ccy, price_factor=core_macro)
        self.currency.name = "currency"

        self.momentum = excess_performance(
            self.data.momentum.dropna(), self.data.developedequity.dropna()
        )
        self.momentum.name = "momentum"

        self.value = excess_performance(
            self.data.value.dropna(), self.data.developedequity.dropna()
        )
        self.value.name = "value"

        self.growth = excess_performance(
            self.data.growth.dropna(), self.data.developedequity.dropna()
        )
        self.growth.name = "growth"

        self.smallcap = excess_performance(
            self.data.smallcap.dropna(), self.data.developedequity.dropna()
        )
        self.smallcap.name = "smallcap"

        self.lowvol = excess_performance(
            self.data.lowvol.dropna(), self.data.developedequity.dropna()
        )
        self.lowvol.name = "lowvol"

        return self

    @property
    def performance(self):
        return pd.concat(
            objs=[
                self.rate,
                self.equity,
                self.credit,
                self.commodity,
                self.localinflation,
                self.emerging,
                self.localequity,
                self.currency,
                self.shortvol,
                self.momentum,
                self.value,
                self.lowvol,
                self.smallcap,
                self.growth,
            ],
            axis=1,
        )

    @staticmethod
    def residualization(
        price: pd.Series,
        price_factor: pd.DataFrame,
        window: int = 252 * 3,
        smoothing_window: int = 5,
        **kwargs,
    ) -> pd.Series:

        itx = price.index.intersection(price_factor.index)

        price = price.loc[itx]
        price_factor = price_factor.loc[itx]

        pri_return_1 = price.pct_change().fillna(0)
        pri_return_2 = price_factor.pct_change().fillna(0)

        betas = exposure(
            dependents=pri_return_1, independents=pri_return_2, window=window, **kwargs
        )

        if smoothing_window is not None:
            betas = betas.rolling(window=smoothing_window, min_periods=0).mean()

        perf_exposure = expsoures_implied_performance(
            exposures=betas, price_factor=price_factor
        )

        return excess_performance(price, perf_exposure)

from hive import db



# # swap dict
# kwargs = {v: k for k, v in FactorLensSettings().dict().items()}
# data = db.get_price(",".join(list(kwargs.keys())))
# data = data.rename(columns=kwargs)


import pandas as pd


prices_idx = pd.read_csv("index_price.csv", index_col=["date"], parse_dates=["date"])
lens = FactorLens(data=prices_idx)

perf = lens.performance
perf.index.name = "date"
perf.to_csv("factor_price.csv")

# perf = perf.stack().reset_index()
# perf.columns = ["date", "factor", "value"]
# with db.session_local() as session:
#     session.query(db.TbFactorIndex).delete()
#     session.flush()
#     session.bulk_insert_mappings(db.TbFactorIndex, perf.to_dict("records"))
#     session.commit()