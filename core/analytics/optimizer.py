import warnings
from functools import partial
from typing import Union, Optional, Callable, List, Dict, Tuple
from pydantic import BaseModel
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import to_tree, linkage
from scipy.spatial.distance import squareform
from .pa import metrics


class PortfolioStatsModel(BaseModel):
    """format of portfolio stats"""

    expected_return: Optional[float] = None
    expected_risk: Optional[float] = None
    expected_sharpe: Optional[float] = None
    risk_contribution: Optional[Dict] = None


class OptimizerStats:
    expected_return: float
    expected_risk: float
    expected_sharpe: float
    asset_risk_contribution: dict


class CustomTotWeight(BaseModel):
    """format of weight constraint"""

    assets: list or pd.Index
    sign: str
    value: int or float


class Optimizer:
    """portfolio optimizer class"""

    def __init__(
        self,
        price_df: Optional[pd.DataFrame] = None,
        expected_returns: Union[pd.Series, str] = "empirical",
        covariance_matrix: Union[pd.DataFrame, str] = "empirical",
        risk_free: float = 0.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        tot_weight: float = 1.0,
        target_return: Optional[float] = None,
        min_return: Optional[float] = None,
        max_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        min_risk: Optional[float] = None,
        max_risk: Optional[float] = None,
        max_active_weight: Optional[float] = None,
        max_exante_tracking_error: Optional[float] = None,
        max_expost_tracking_error: Optional[float] = None,
        weights_bm: Optional[pd.Series] = None,
        price_bm: Optional[pd.Series] = None,
    ) -> None:
        # pylint: disable=multiple-statements
        """initialize"""

        self.stats = PortfolioStatsModel()
        self.constraints: Dict = {}

        if isinstance(expected_returns, pd.Series):
            self.expected_returns = expected_returns
            self.assets = self.expected_returns.index
        if isinstance(covariance_matrix, pd.DataFrame):
            self.covariance_matrix = covariance_matrix
            if not hasattr(self, "assets"):
                self.assets = self.covariance_matrix.index
            else:
                assert self.assets.equals(self.covariance_matrix.index)

        if price_df is not None:
            self.price_df: pd.DataFrame = price_df
            if not hasattr(self, "assets"):
                self.assets = self.price_df.columns
            else:
                assert self.assets.equals(self.price_df.columns)
            if isinstance(expected_returns, str):
                self.expected_returns = metrics.expected_returns(
                    price_df=self.price_df, method=expected_returns
                )
            if isinstance(covariance_matrix, str):
                self.covariance_matrix = metrics.covariance_matrix(
                    price_df=self.price_df, method=covariance_matrix
                )

        self.num_asset = len(self.assets)

        self.risk_free = risk_free
        self.weights_bm = weights_bm
        self.price_bm = price_bm

        self.set_min_weight(min_weight)
        self.set_max_weight(max_weight)
        self.set_tot_weight(tot_weight)

        if target_return is not None:
            min_return = max_return = target_return
        if min_return is not None:
            self.set_min_return(min_return)
        if max_return is not None:
            self.set_max_return(max_return)

        if target_risk is not None:
            min_risk = max_risk = target_risk
        if min_risk is not None:
            self.set_min_return(min_risk)
        if max_risk is not None:
            self.set_max_return(max_risk)

        if max_active_weight is not None:
            self.set_max_active_weight(max_active_weight)
        if max_exante_tracking_error is not None:
            self.set_max_exante_tracking_error(max_exante_tracking_error)
        if max_expost_tracking_error is not None:
            self.set_max_expost_tracking_error(max_expost_tracking_error)

    def set_min_weight(self, min_weight: float) -> None:
        """minimum weight constraint"""
        constr = dict(min_weight=dict(type="ineq", fun=lambda w: w - min_weight))
        self.constraints.update(constr)

    def set_max_weight(self, max_weight: float) -> None:
        """maximum weight constraint"""
        constr = dict(max_weight=dict(type="ineq", fun=lambda w: max_weight - w))
        self.constraints.update(constr)

    def set_tot_weight(self, tot_weight: float) -> None:
        """total weight constraint"""
        constr = dict(tot_weight=dict(type="eq", fun=lambda w: np.sum(w) - tot_weight))
        self.constraints.update(constr)

    def set_min_return(self, min_return: float) -> None:
        """minimum return constraint"""
        if self.expected_returns is None:
            warnings.warn("Unable to set min_return, due to missing expected returns.")
            return
        constr = dict(
            min_return=dict(
                type="ineq",
                fun=lambda w: self.portfolio_return(
                    weights=w, expected_returns=np.array(self.expected_returns)
                )
                - min_return,
            )
        )
        self.constraints.update(constr)

    def set_max_return(self, max_return: float) -> None:
        """maximum return constraint"""
        if self.expected_returns is None:
            warnings.warn(
                "unable to set maximum return constraint due to missing expected returns"
            )
            return
        constr = dict(
            max_return=dict(
                type="ineq",
                fun=lambda w: max_return
                - self.portfolio_return(
                    weights=w, expected_returns=np.array(self.expected_returns)
                ),
            )
        )
        self.constraints.update(constr)

    def set_min_risk(self, min_risk: float) -> None:
        """minimum risk constriant"""
        if self.covariance_matrix is None:
            warnings.warn(
                "unable to set minimum risk constraint due to missing covariance matrix"
            )
            return
        constr = dict(
            min_risk=dict(
                type="ineq",
                func=lambda w: self.portfolio_volatility(
                    weights=w,
                    covariance_matrix=np.array(self.covariance_matrix) - min_risk,
                ),
            )
        )
        self.constraints.update(constr)

    def set_max_active_weight(self, max_active_weight: float) -> None:
        """maximum active weight constraint"""
        if self.weights_bm is None:
            warnings.warn(
                "unable to set maximum active weight, due to missing benchmark weights."
            )
            return
        constr = dict(
            max_active_weight=dict(
                type="ineq",
                fun=lambda w: max_active_weight - np.sum(np.abs(w - self.weights_bm)),
            )
        )
        self.constraints.update(constr)

    def set_max_exante_tracking_error(self, max_exante_tracking_error: float) -> None:
        """maximum exante tracking error constraint"""
        if self.weights_bm is None:
            warnings.warn(
                "unable to set max_exante_tracking_error, "
                + "due to missing benchmark weights."
            )
            return
        if self.covariance_matrix is None:
            warnings.warn(
                "unable to set max_exante_tracking_error, "
                + "due to missing covariance matrix."
            )
            return
        constr = dict(
            max_exante_tracking_error=dict(
                type="ineq",
                fun=lambda w: max_exante_tracking_error
                - self.exante_tracking_error(
                    weights=w,
                    weights_bm=np.array(self.weights_bm),
                    covariance_matrix=np.array(self.weights_bm),
                ),
            )
        )
        self.constraints.update(constr)

    def set_max_expost_tracking_error(self, max_expost_tracking_error: float) -> None:
        """maximum exante tracking error constraint"""
        if self.price_bm is not None and self.price_df is not None:
            itx = self.price_df.dropna().index.intersection(
                self.price_bm.dropna().index
            )
            pri_return_df = np.array(self.price_df.loc[itx].pct_change().fillna(0))
            pri_return_bm = np.array(self.price_bm.loc[itx].pct_change().fillna(0))

            constr = dict(
                max_exante_tracking_error=dict(
                    type="ineq",
                    fun=lambda w: max_expost_tracking_error
                    - self.expost_tracking_error(
                        weights=w,
                        pri_return_df=pri_return_df,
                        pri_return_bm=pri_return_bm,
                    ),
                )
            )
            self.constraints.update(constr)

    def custom_tot_weight(self, constr: dict) -> None:
        """custom total weight"""
        ctw = CustomTotWeight(**constr)
        name = str(ctw)
        target_assets = np.in1d(ctw.assets, np.array(self.assets))
        if ctw.sign == "=":
            self.constraints.update(
                {
                    name: dict(
                        type="eq", fun=lambda w: np.dot(w, target_assets) - ctw.value
                    )
                }
            )
        elif ctw.sign == ">=":
            self.constraints.update(
                {
                    name: dict(
                        type="ineq", fun=lambda w: np.dot(w, target_assets) - ctw.value
                    )
                }
            )
        elif ctw.sign == "<=":
            self.constraints.update(
                {
                    name: dict(
                        type="ineq", fun=lambda w: ctw.value - np.dot(w, target_assets)
                    )
                }
            )
        else:
            raise ValueError("sign is .. ")

    def solve(
        self, objective: Callable, extra_constraints: Optional[List[Callable]] = None
    ) -> Optional[pd.Series]:
        """solve the objective minimization problem"""

        constraints = list(self.constraints.values())
        if extra_constraints is not None:
            constraints.append(extra_constraints)

        problem = minimize(
            fun=objective,
            method="SLSQP",
            constraints=constraints,
            x0=np.ones(shape=self.num_asset) / self.num_asset,
        )
        if problem.success:
            weights = pd.Series(data=problem.x, index=self.assets, name="weights")

            if self.expected_returns is not None:
                self.stats.expected_risk = round(
                    self.portfolio_return(
                        weights=problem.x,
                        expected_returns=np.array(self.expected_returns),
                    ),
                    4,
                )
            if self.covariance_matrix is not None:
                self.stats.expected_risk = round(
                    self.portfolio_volatility(
                        weights=problem.x,
                        covariance_matrix=np.array(self.covariance_matrix),
                    ),
                    4,
                )
                self.stats.risk_contribution = (
                    pd.Series(
                        data=self.risk_contribution(
                            weights=problem.x,
                            covariance_matrix=np.array(self.covariance_matrix),
                        ),
                        index=self.assets,
                    )
                    .round(4)
                    .to_dict()
                )

            if self.expected_returns is not None and self.covariance_matrix is not None:
                self.stats.expected_sharpe = round(
                    self.portfolio_sharpe(
                        weights=problem.x,
                        expected_returns=np.array(self.expected_returns),
                        covariance_matrix=np.array(self.covariance_matrix),
                    ),
                    4,
                )
            return weights.round(4)

        return None

    def max_return_weights(self) -> Optional[pd.Series]:
        """maximum return weights"""
        return self.solve(
            objective=partial(
                self.portfolio_return,
                expected_returns=np.array(self.expected_returns),
            )
        )

    def min_return_weights(self) -> Optional[pd.Series]:
        """maximum return weights"""
        return self.solve(
            objective=partial(
                self.portfolio_return,
                expected_returns=np.array(self.expected_returns),
            )
        )

    def min_risk_weights(self) -> Optional[pd.Series]:
        """maximum return weights"""
        return self.solve(
            objective=partial(
                self.portfolio_volatility,
                covariance_matrix=np.array(self.covariance_matrix),
            )
        )

    def risk_budget_weights(
        self, budgets: Union[np.ndarray, pd.Series, None] = None
    ) -> Optional[pd.Series]:
        """risk parity weights"""
        # pylint: disable=unnecessary-lambda-assignment

        if budgets is None:
            budgets = np.ones(len(self.assets)) / len(self.assets)

        objective = lambda w: self.l2_norm(
            np.subtract(
                self.risk_contribution(
                    w, covariance_matrix=np.array(self.covariance_matrix)
                ),
                np.multiply(
                    budgets,
                    self.portfolio_volatility(
                        w, covariance_matrix=np.array(self.covariance_matrix)
                    ),
                ),
            )
        )
        return self.solve(objective=objective)

    def hierarchical_risk_parity_weights(
        self, linkage_method: str = "single"
    ) -> Optional[pd.Series]:
        """hrp weights"""
        corr = cov_to_corr(np.array(self.covariance_matrix))
        distance_matrix = squareform(np.sqrt((1 - corr).round(5) / 2))
        clusters = linkage(distance_matrix, method=linkage_method)
        sorted_index = list(to_tree(clusters, rd=False).pre_order())
        clustered_assets = [[self.assets[x] for x in sorted_index]]
        w = pd.Series(1, index=clustered_assets)
        while len(clustered_assets) > 0:
            clustered_assets = [
                i[j:k]
                for i in clustered_assets
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]  # bi-section
            # For each pair, optimize locally.
            for i in range(0, len(clustered_assets), 2):
                left_cluster = clustered_assets[i]
                right_cluster = clustered_assets[i + 1]
                # Form the inverse variance portfolio for this pair
                left_variance = self.inv_var_cluster_variance(
                    self.covariance_matrix.loc[left_cluster, left_cluster].values
                )
                right_variance = self.inv_var_cluster_variance(
                    self.covariance_matrix.loc[right_cluster, right_cluster].values
                )
                alpha = 1 - left_variance / (left_variance + right_variance)
                w[left_cluster] *= alpha  # weight 1
                w[right_cluster] *= 1 - alpha  # weight 2
        return w

    def hrp_weights(self) -> Optional[pd.Series]:
        corr = cov_to_corr(self.covariance_matrix.values)
        dist = np.sqrt((1 - corr).round(5) / 2)
        clusters = linkage(squareform(dist), method="single")
        cov_sets = []
        for cluster in clusters:
            left_idx = add_assets(
                clusters=clusters, num=cluster[0], num_assets=self.num_asset
            )
            right_idx = add_assets(
                clusters=clusters, num=cluster[1], num_assets=self.num_asset
            )

            left_cov = self.covariance_matrix.copy()
            right_cov = self.covariance_matrix.copy()
            for i in range(self.num_asset):
                if i not in left_idx:
                    left_cov.iloc[i, :] = 0
                    left_cov.iloc[:, i] = 0
                if i not in right_idx:
                    right_cov.iloc[i, :] = 0
                    right_cov.iloc[:, i] = 0

            cov_sets.append((left_cov, right_cov))

        objective = partial(
            self.active_risk_contribution,
            cov=np.array(self.covariance_matrix),
            cov_sets=cov_sets,
        )
        return self.solve(objective=objective)

    @staticmethod
    def portfolio_return(
        weights: np.ndarray,
        expected_returns: np.ndarray,
    ) -> float:
        """
        Portfolio expected return.

        Args:
            weight (np.ndarray): weight of assets.
            expected_returns (np.ndarray): expected return of assets.

        Returns:
            float: portfolio expected return.
        """
        return np.dot(weights, expected_returns)

    @staticmethod
    def portfolio_variance(
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> float:
        """
        Portfolio expected variance.

        Args:
            weight (np.ndarray): weight of assets.
            covariance_matrix (np.ndarray): covariance matrix of assets.

        Returns:
            float: portfolio expected variance.
        """
        return np.dot(np.dot(weights, covariance_matrix), weights)

    def portfolio_volatility(
        self, weights: np.ndarray, covariance_matrix: np.ndarray
    ) -> float:
        """
        Portfolio expected volatility.

        Args:
            weight (np.ndarray): weight of assets.
            covariance_matrix (np.ndarray): covariance matrix of assets.

        Returns:
            float: portfolio expected volatility.
        """
        return np.sqrt(
            self.portfolio_variance(
                weights=weights, covariance_matrix=covariance_matrix
            )
        )

    def portfolio_sharpe(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free: float = 0.0,
    ) -> float:
        """
        Portfolio expected sharpe ratio.

        Args:
            weight (np.ndarray): weight of assets.
            expected_returns (np.ndarray): expected return of assets.
            covariance_matrix (np.ndarray): covariance matrix of assets.

        Returns:
            float: portfolio expected sharpe ratio.
        """
        ret = self.portfolio_return(weights=weights, expected_returns=expected_returns)
        std = self.portfolio_volatility(
            weights=weights, covariance_matrix=covariance_matrix
        )
        return (ret - risk_free) / std

    @staticmethod
    def l1_norm(weights: np.ndarray, gamma: float = 1) -> float:
        """
        L1 regularization.

        Args:
            weight (np.ndarray): asset weight in the portfolio.
            gamma (float, optional): L2 regularisation parameter. Defaults to 1.
                Increase if you want more non-negligible weight.

        Returns:
            float: L2 regularization.
        """
        return np.abs(weights).sum() * gamma

    @staticmethod
    def l2_norm(weights: np.ndarray, gamma: float = 1) -> float:
        """
        L2 regularization.

        Args:
            weight (np.ndarray): asset weight in the portfolio.
            gamma (float, optional): L2 regularisation parameter. Defaults to 1.
                Increase if you want more non-negligible weight.

        Returns:
            float: L2 regularization.
        """
        return np.sum(np.square(weights)) * gamma

    @staticmethod
    def exante_tracking_error(
        weights: np.ndarray, weights_bm: np.ndarray, covariance_matrix: np.ndarray
    ) -> float:
        """
        Calculate the ex-ante tracking error.

        Maths:
            formula here.

        Args:
            weight (np.ndarray): asset weight in the portfolio.
            weight_benchmark (np.ndarray): benchmarket weight of the portfolio.
            covaraince_matrix (np.ndarray): asset covariance matrix.

        Returns:
            float: ex-ante tracking error.
        """
        rel_weight = np.subtract(weights, weights_bm)
        tracking_variance = np.dot(np.dot(rel_weight, covariance_matrix), rel_weight)
        tracking_error = np.sqrt(tracking_variance)
        return tracking_error

    @staticmethod
    def expost_tracking_error(
        weights: np.ndarray,
        pri_return_df: np.ndarray,
        pri_return_bm: np.ndarray,
    ) -> float:
        """..."""
        relative_return = np.dot(pri_return_df, weights) - pri_return_bm
        mean = np.sum(relative_return) / len(relative_return)
        return np.sum(np.square(relative_return - mean))

    def risk_contribution(
        self, weights: np.ndarray, covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """
        calculate risk contribution
        """
        volatility = self.portfolio_volatility(weights, covariance_matrix)
        return np.multiply(weights, covariance_matrix.dot(weights)) / volatility

    @staticmethod
    def inverse_variance_weights(covariance_matrix: np.ndarray) -> np.ndarray:
        """calculate weights of inverse variance. (tot weights = 100%)

        Args:
            covariance_matrix (np.ndarray): _description_

        Returns:
            np.ndarray: weights of
        """
        inv_var_weights = 1 / np.diag(covariance_matrix)
        inv_var_weights /= inv_var_weights.sum()
        return inv_var_weights

    def inv_var_cluster_variance(self, covariance_matrix: np.ndarray) -> float:
        """_summary_

        Args:
            covariance_matrix (np.ndarray): _description_

        Returns:
            float: _description_
        """
        weights = self.inverse_variance_weights(covariance_matrix=covariance_matrix)
        return np.linalg.multi_dot((weights, covariance_matrix, weights))

    def active_risk_contribution(
        self,
        weights: np.ndarray,
        cov: np.ndarray,
        cov_sets: List[Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """calculate the sum of active risk contribution from the covariance sets"""
        std = self.portfolio_volatility(weights=weights, covariance_matrix=cov)
        return np.sum(
            np.abs(
                np.array(
                    [
                        np.sum(self.risk_contribution(weights, left_cov)) / std
                        - np.sum(self.risk_contribution(weights, right_cov)) / std
                        for left_cov, right_cov in cov_sets
                    ]
                )
            )
        )


def add_assets(clusters, num, num_assets):
    if num < num_assets:
        return [int(num)]
    row = clusters[int(num - num_assets)]
    left = add_assets(clusters, row[0], num_assets)
    right = add_assets(clusters, row[1], num_assets)
    if not isinstance(left, list):
        left = [int(left)]
    if isinstance(right, list):
        left.extend(right)
    else:
        left.append(int(right))
    return left


def cov_to_corr(covariance_matrix: np.ndarray) -> np.ndarray:
    """correlation matrix from covariance matrix"""
    vol = np.sqrt(np.diag(covariance_matrix))
    corr = np.divide(covariance_matrix, np.outer(vol, vol))
    corr[corr < -1], corr[corr > 1] = -1, 1
    return corr


def quasi_diagnalization(clusters, num_assets, curr_index):
    """Rearrange the assets to reorder them according to hierarchical tree clustering order"""

    if curr_index < num_assets:
        return [curr_index]

    left = int(clusters[curr_index - num_assets, 0])
    right = int(clusters[curr_index - num_assets, 1])

    return quasi_diagnalization(clusters, num_assets, left) + quasi_diagnalization(
        clusters, num_assets, right
    )
