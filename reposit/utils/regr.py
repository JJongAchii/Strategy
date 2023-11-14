"""
Regression Function
"""

import pandas as pd
from sklearn import linear_model
from utils import weight

"""
Linear regression refers to a model that assumes a linear relationship between
input variables and the target variable.

A problem with linear regression is that estimated coefficients of the model
can become large, making the model sensitive to inputs and possibly unstable.

One approach to address the stability of regression models is to change the
loss function to include additional costs for a model that has large
coefficients. Linear regression models that use these modified loss functions
during training are referred to collectively as penalized linear regression.
"""


def align_price(
    dependent: pd.Series,
    independents: pd.DataFrame,
    min_fraction: float = 0.80,
) -> tuple:
    """
    Align two regression data.

    Args:
        dependent (pd.Series): dependent variable.
        independents (pd.DataFrame): independent variable.
        min_fraction (float, optional): minimum percent of data points against
            the total window. Defaults to 0.90.

    Returns:
        tuple(pd.Series, pd.DataFrame): aligned dependent and independents.
    """

    dependent = dependent.dropna()
    independents = independents.dropna()

    start = max(dependent.index[0], independents.index[0])
    end = min(dependent.index[-1], independents.index[-1])

    dependent = dependent.loc[start:end]

    independents = independents.reindex(
        pd.date_range(start=start, end=end)
    ).ffill().dropna(how='all')

    itx = pd.Index.intersection(
        dependent.index,
        independents.index,
    )

    if len(dependent) == 0 or min_fraction > len(itx) / len(dependent):
        return None, None

    independents = independents.loc[itx]
    dependent = dependent.loc[itx]

    return dependent, independents


def align_regression(
    dependent: pd.Series,
    independents: pd.DataFrame,
    min_fraction: float = 0.80,
    custom_weight: str = None,
) -> tuple:
    """
    Align two regression data.

    Args:
        dependent (pd.Series): dependent variable.
        independents (pd.DataFrame): independent variable.
        min_fraction (float, optional): minimum percent of data points against
            the total window. Defaults to 0.90.

    Returns:
        tuple(pd.Series, pd.DataFrame): aligned dependent and independents.
    """

    dependent = dependent.dropna(how='all', axis=0)
    start = min(dependent.index[0], independents.index[0])
    end = min(dependent.index[-1], independents.index[-1])

    independents = independents.reindex(
        pd.date_range(start=start, end=end)
    ).ffill().dropna(how='all')

    itx = pd.Index.intersection(
        dependent.dropna().index,
        independents.dropna().index,
    )

    if len(dependent) == 0 or min_fraction > len(itx) / len(dependent):
        return None, None

    independents = independents.loc[itx]
    dependent = dependent.loc[itx]

    if custom_weight == 'exponential':
        dependent = weight.exponential(dependent)
        independents = weight.exponential(independents)

    return dependent, independents


def regression(
    dependent: pd.DataFrame,
    independents: pd.DataFrame,
    model: str = 'linear',
    window: int = None,
    min_fraction: float = 0.90,
    custom_weight: str = None,
    positive: bool = False,
    fit_intercept: bool = True,
    label_score: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Calculate linear regression coefficient.

    Args:
        dependent (pd.DataFrame): dependent variable.
        independents (pd.DataFrame): independent variable.
        model (str, optional): regression model. Defaults to linear.
        window (int, optional): rolling window. Defualts to None.
        min_fraction (float, optional): minimum percent of data points against
            the total window. Defaults to 0.90.
        custom_weight (str, optional): _description_. Defaults to None.
        positive (bool, optional): if positve regression coefficients.
            Defaults to False.
        fit_intercept (bool, optional): fit y-intercept. Defaults to True.
        label_score (bool, optional): if label r2 score in result.
            Defaults to False.

    References:
        A tuning parameter, λ controls the strength of the L1 penalty.
        λ is basically the amount of shrinkage:
            When λ = 0, no parameters are eliminated.
            The estimate is equal to the one found with linear regression.
            As λ increases, more coefficients are set to zero and eliminated
            (theoretically, when λ = ∞, all coefficients are eliminated).
            As λ increases, bias increases.
            As λ decreases, variance increases.

    Tuning:
        First, trying to set λ to find a pre-sopecified number of important
        features isn't a good idea. Whether a feature is predictive of the
        response is a property of the data, not your model. So you want your
        model to tell you how many features are important, not the other way
        around. If you try to mess with your alpha until it finds a
        pre-specified number of features to be predictive, you run the risk
        of over / underfitting.

        On the other hand, some papers have noted that selecting alpha by
        minimizing cross-validated error does not yield consistent feature
        selection in practice.


    Returns:
        pd.DataFrame: regression coefficients.
    """

    if isinstance(dependent, pd.DataFrame):

        if window is None:

            return pd.concat(
                list(
                    regression(
                        dependent=dependent[x],
                        independents=independents,
                        model=model,
                        window=window,
                        min_fraction=min_fraction,
                        custom_weight=custom_weight,
                        positive=positive,
                        fit_intercept=fit_intercept,
                        label_score=label_score
                    ) for x in dependent
                ), axis=1
            ).T

        return pd.concat(
            objs=list(
                pd.concat(
                    objs=[
                        regression(
                            dependent=dependent[x],
                            independents=independents,
                            model=model,
                            window=window,
                            min_fraction=min_fraction,
                            custom_weight=custom_weight,
                            positive=positive,
                            fit_intercept=fit_intercept,
                            label_score=label_score
                        )
                    ], keys=[x], axis=1
                ) for x in dependent
            ), axis=1
        )

    if window is not None:

        result = dict()

        for i in range(window, len(dependent) + 1):
            exposures = regression(
                dependent=dependent.iloc[i-window:i],
                independents=independents,
                model=model,
                min_fraction=min_fraction,
                custom_weight=custom_weight,
                positive=positive,
                fit_intercept=fit_intercept,
                label_score=label_score,
                window=None,
            )
            if exposures is not None:
                result[dependent.index[i-1]] = exposures

        return pd.DataFrame(result).T

    dependent, independents = align_regression(
        dependent=dependent,
        independents=independents,
        min_fraction=min_fraction,
        custom_weight=custom_weight,
    )

    if dependent is None or independents is None: return None

    if custom_weight == 'exponential':
        dependent = weight.exponential(dependent)
        independents = weight.exponential(independents)

    if model == 'linear':
        regressor = linear_model.LinearRegression(
            positive=positive,
            fit_intercept=fit_intercept
        )

    elif model == 'lasso':

        alpha = kwargs.get('alpha', None)
        if alpha is None:
            regressor = linear_model.LassoCV(
                positive=positive,
                fit_intercept=fit_intercept
            )
        else:
            regressor = linear_model.Lasso(
                positive=positive,
                fit_intercept=fit_intercept,
                alpha=alpha,
            )

    regressor.fit(X=independents, y=dependent)

    betas = pd.Series(
        data=regressor.coef_,
        index=independents.columns,
        name=dependent.name
    )

    if label_score:
        betas['score'] = regressor.score(
            X=independents, y=dependent
        )

    return betas
