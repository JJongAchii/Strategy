""" base strategy class """

import warnings
from typing import Any, List, Optional, Dict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections.abc import Iterable


class Account:
    """strategy account"""

    def __init__(self):
        self.date: List = []
        self.value: List = []
        self.weights: List[Dict] = []
        self.reb_weights: List[Dict] = []
        self.trade_weights: List[Dict] = []


class BaseStrategy:
    """
    BaseStrategy class is an algorithmic trading strategy that sequentially allocates capital among
    group of assets based on pre-defined allocatioin method.

    BaseStrategy shall be the parent class for all investment strategies with period-wise
    rebalancing scheme.

    Using this class requires following pre-defined methods:
    1. rebalance(self, price_asset, date, **kwargs):
        the method shall be in charge of calculating new weights based on prices
        that are avaiable.
    2. monitor (self, ...):
        the method shall be in charge of monitoring the strategy status, and if
        necessary call the rebalance method to re-calculate weights. aka irregular rebalancing.
    """

    def __init__(
        self,
        price_asset: pd.DataFrame,
        frequency: str = "M",
        min_assets: int = 2,
        min_periods: int = 2,
        investment: float = 1000.0,
        commission: int = 0,
        currency: str = "KRW",
        name: str = "strategy",
        account: Optional[Account] = None,
    ) -> None:
        """Initialization"""
        self.price_asset: pd.DataFrame = self.check_price_df(price_asset)
        self.frequency: str = frequency
        self.min_assets: int = min_assets
        self.min_periods: int = min_periods
        self.name: str = name
        self.commission = commission
        self.currency = currency

        # account information
        self.idx: int = 0
        self.date: Any = None
        self.value: float = investment
        self.weights: pd.Series = pd.Series(dtype=float)
        self.reb_weights: pd.Series = pd.Series(dtype=float)
        self.trade_weights: pd.Series = pd.Series(dtype=float)
        self.account: Account = account or Account()

    ################################################################################################
    @property
    def value_df(self):
        """values dataframe"""
        return pd.DataFrame(
            data=self.account.value, index=self.account.date, columns=["value"]
        )

    @property
    def weights_df(self):
        """weights dataframe"""
        return pd.DataFrame(data=self.account.weights, index=self.account.date)

    @property
    def reb_weights_df(self):
        """weights dataframe"""
        return pd.DataFrame(data=self.account.reb_weights, index=self.account.date)

    @property
    def trade_weights_df(self):
        """weights dataframe"""
        return pd.DataFrame(data=self.account.trade_weights, index=self.account.date)


    @staticmethod
    def calc_adjusted_portfolio_weight(weights: pd.Series, w_max: float = 0.6, w_min: float = 0.02) -> dict:
            
        max_error = {}
        min_error = {}
        
        for key in weights.keys():

            if weights[key] > w_max:

                max_error[key] = weights[key]

            elif weights[key] < w_min:

                min_error[key] = weights[key]

        if len(max_error) > 0:
            
            final_weights_keys = weights.keys()
            final_weights_keys = [a for a in final_weights_keys if a not in max_error.keys()]
            
            max_left = weights.sum()
            diff = 0
            for err_key in max_error.keys():
                diff += max_error[err_key] - w_max
                weights[err_key] = w_max
                max_left -= max_error[err_key]
            
            for key in final_weights_keys:
                weights[key] += (diff * weights[key] / max_left)

        if len(min_error) > 0:
            
            final_weights_keys = weights.keys()
            final_weights_keys = [a for a in final_weights_keys if a not in min_error.keys()]
            
            min_left = weights.sum()
            diff = 0
            for err_key in min_error.keys():
                diff += abs(w_min - min_error[err_key]) 
                weights[err_key] = w_min
                min_left -= min_error[err_key]

            for key in final_weights_keys:
                weights[key] -= (diff * weights[key] / min_left)
        
        return pd.Series(weights)
    ################################################################################################

    def update_book(self) -> None:
        """update the account value based on the current date"""

        prices = self.price_asset.loc[self.date]
        if not self.weights.empty:
            pre_prices = self.price_asset.iloc[
                self.price_asset.index.get_loc(self.date) - 1
            ]
            capitals = self.weights * self.value
            new_capitals = capitals / pre_prices * prices
            profit_loss = new_capitals - capitals
            self.value += profit_loss.sum()
            self.weights = new_capitals / self.value

        if not self.reb_weights.empty:
            # reindex to contain the same asset.
            union_assets = self.reb_weights.index.union(self.weights.index)
            self.weights = self.weights.reindex(union_assets, fill_value=0)
            self.reb_weights = self.reb_weights.reindex(union_assets, fill_value=0)
            self.trade_weights = self.reb_weights.subtract(self.weights)
            trade_capitals = self.value * self.trade_weights
            trade_costs = trade_capitals.abs() * self.commission / 10_000
            trade_cost = trade_costs.sum()
            # update the account metrics.
            self.value -= trade_cost
            self.weights = self.reb_weights

        # do nothing if no account data.
        if self.weights.empty and self.reb_weights.empty:
            return
        # loop through all variables in account history
        for name in vars(self.account).keys():
            getattr(self.account, name).append(getattr(self, name))
        # clear the rebalancing weights.
        self.reb_weights = pd.Series(dtype=float)
        self.trade_weights = pd.Series(dtype=float)

    ################################################################################################

    def simulate(self, start: ... = None, end: ... = None) -> ...:
        """simulate historical strategy perfromance"""
        start = start or self.price_asset.index[0]
        end = end or self.price_asset.index[-1]

        reb_dates = pd.date_range(start=start, end=end, freq=self.frequency)

        for self.date in pd.date_range(start=start, end=end, freq="D"):
            if self.date in self.price_asset.index: self.update_book()
            if self.weights.empty or self.monitor() or self.date in reb_dates:
                self.reb_weights = self.allocate()
            if self.date not in self.price_asset.index:
                continue
            

        return self

    def allocate(self) -> pd.Series:
        """allocate weights based on date if date not provided use latest"""
        # pylint: disable=multiple-statements
        if self.date is None: self.date = datetime.today()
        price_slice = (
            self.price_asset.loc[:self.date]
            .dropna(thresh=self.min_periods, axis=1)
            .dropna(thresh=self.min_assets, axis=0)
        ).copy()
        if price_slice.empty:
            return pd.Series(dtype=float)
        reb_weights = self.rebalance(price_asset=price_slice)
        if reb_weights is None:
            return pd.Series(dtype=float)
        return self.clean_weights(reb_weights, decimals=4)

    # Temporary Function for core view
    def view_prediction(self, today:datetime) -> pd.Series:
        slicing_date = datetime(today.year,today.month,1)-timedelta(days=1)
        price_slice = self.price_asset.loc[:slicing_date].copy()
        if price_slice.empty:
            return pd.Series(dtype=float)
        prediction_result = self.view_prediction_(price_asset=price_slice)
        return prediction_result
        
    def rebalance(self, price_asset: pd.DataFrame) -> pd.Series:
        """Default rebalancing method"""
        asset = price_asset.columns
        uniform_weight = np.ones(len(asset))
        uniform_weight /= uniform_weight.sum()
        weight = pd.Series(index=asset, data=uniform_weight)
        return weight

    def monitor(self) -> bool:
        """Default monitoring method."""
        return False

    ################################################################################################
    @staticmethod
    def check_price_df(price_df: pd.DataFrame) -> pd.DataFrame:
        """Check the price_df.

        Args:
            price_df (pd.DataFrame): _description_

        Raises:
            TypeError: if price_df is not pd.DataFrame.

        Returns:
            pd.DataFrame: price_df
        """
        if not isinstance(price_df, pd.DataFrame):
            raise TypeError("price_df must be a pd.DataFrame.")
        if not isinstance(price_df.index, pd.DatetimeIndex):
            warnings.warn("converting price_df's index to pd.DatetimeIndex.")
            price_df.index = pd.to_datetime(price_df.index)
        return price_df

    @staticmethod
    def clean_weights(weights: pd.Series, decimals: int = 4, tot_weight=None) -> pd.Series:
        """Clean weights based on the number decimals and maintain the total of weights.

        Args:
            weights (pd.Series): asset weights.
            decimals (int, optional): number of decimals to be rounded for
                weight. Defaults to 4.

        Returns:
            pd.Series: clean asset weights.
        """
        # clip weight values by minimum and maximum.
        if not tot_weight:
            tot_weight = weights.sum().round(4)
        weights = weights.round(decimals=decimals)
        # repeat round and weight calculation.
        for _ in range(10):
            weights = weights / weights.sum() * tot_weight
            weights = weights.round(decimals=decimals)
            if weights.sum() == tot_weight:
                return weights
        # if residual remains after repeated rounding.
        # allocate the the residual weight on the max weight.
        residual = tot_weight - weights.sum()
        # !!! Error may occur when there are two max weights???
        weights.iloc[np.argmax(weights)] += np.round(residual, decimals=decimals)
        return weights