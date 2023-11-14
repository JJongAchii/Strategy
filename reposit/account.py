"""Robert Han"""
from typing import Optional, Callable
import pandas as pd

# from .account import VirtualAccount



# from typing import Optional
# import pandas as pd


class AccountRecords:
    """virtual account records to store account records"""

    def __init__(self) -> None:
        self.value = {}
        self.shares = {}
        self.capitals = {}
        self.prices = {}
        self.profits = {}
        self.trades = {}
        self.allocations = {}
        self.weights = {}


class AccountMetrics:
    """virtual account metrics to store account metrics"""

    def __init__(self, initial_investment: float = 1000.0) -> None:
        self.value: float = initial_investment
        self.date: Optional[pd.Timestamp] = None
        self.shares: pd.Series = pd.Series(dtype=float)
        self.prices: pd.Series = pd.Series(dtype=float)
        self.capitals: pd.Series = pd.Series(dtype=float)
        self.weights: pd.Series = pd.Series(dtype=float)
        self.profits: pd.Series = pd.Series(dtype=float)
        self.trades: pd.Series = pd.Series(dtype=float)
        self.allocations: pd.Series = pd.Series(dtype=float)


class VirtualAccount:
    """virtual account to store account information"""

    def __init__(self, initial_investment: float = 1000.0) -> None:
        self.metrics = AccountMetrics(initial_investment)
        self.records = AccountRecords()

    ################################################################################
    @property
    def date(self) -> Optional[pd.Timestamp]:
        """account date property"""
        return self.metrics.date

    @date.setter
    def date(self, date: pd.Timestamp) -> None:
        """account date property"""
        self.metrics.date = date

    ################################################################################
    @property
    def value(self) -> float:
        """account value property"""
        return self.metrics.value

    @value.setter
    def value(self, value: float) -> None:
        """account value property"""
        self.metrics.value = value
        self.records.value.update({self.metrics.date: self.value})
        self.weights = self.capitals.divide(self.value)

    ################################################################################
    @property
    def prices(self) -> pd.Series:
        """account value property"""
        return self.metrics.prices

    @prices.setter
    def prices(self, prices: pd.Series) -> None:
        """account value property"""
        self.metrics.prices = prices
        self.records.prices.update({self.metrics.date: self.prices})
        if not self.shares.empty:
            capitals = self.shares.multiply(self.prices.fillna(0))
            self.profits = capitals.subtract(self.capitals)
            self.capitals = capitals

    ################################################################################
    @property
    def shares(self) -> pd.Series:
        """account shares property"""
        return self.metrics.shares

    @shares.setter
    def shares(self, shares: pd.Series) -> None:
        """account shares property"""
        self.metrics.shares = shares
        self.records.shares.update({self.metrics.date: self.shares})

    ################################################################################
    @property
    def capitals(self) -> pd.Series:
        """account capitals property"""
        return self.metrics.capitals

    @capitals.setter
    def capitals(self, capitals: pd.Series) -> None:
        """account capitals property"""
        self.metrics.capitals = capitals
        self.records.capitals.update({self.metrics.date: self.capitals})
        self.value = self.capitals.sum()

    ################################################################################
    @property
    def weights(self) -> pd.Series:
        """account weights property"""
        return self.metrics.weights

    @weights.setter
    def weights(self, weights: pd.Series) -> None:
        """account weights property"""
        self.metrics.weights = weights
        self.records.weights.update({self.metrics.date: self.weights})

    ################################################################################
    @property
    def allocations(self) -> pd.Series:
        """account allocations property"""
        return self.metrics.allocations

    @allocations.setter
    def allocations(self, allocations: pd.Series) -> None:
        """account allocations property"""
        self.metrics.allocations = allocations
        self.records.allocations.update({self.metrics.date: self.allocations})

    def reset_allocations(self) -> None:
        """reset allocations"""
        self.metrics.allocations = pd.Series(dtype=float)

    ################################################################################
    @property
    def trades(self) -> pd.Series:
        """account trades property"""
        return self.metrics.trades

    @trades.setter
    def trades(self, trades: pd.Series) -> None:
        """account trades property"""
        self.metrics.trades = trades
        self.records.trades.update({self.metrics.date: self.trades})

    ################################################################################
    @property
    def profits(self) -> pd.Series:
        """account profits property"""
        return self.metrics.profits

    @profits.setter
    def profits(self, profits: pd.Series) -> None:
        """account profits property"""
        self.metrics.profits = profits
        self.records.profits.update({self.metrics.date: self.profits})


class Strategy:
    """base strategy"""

    def __init__(
        self,
        prices: pd.DataFrame,
        rebalance: Callable,
        frequency: str = "M",
        initial_investment: float = 1000.0,
        min_periods: int = 2,
        min_assets: int = 2,
    ) -> None:
        self.account = VirtualAccount(initial_investment=initial_investment)
        self.prices = prices.ffill()
        self.rebalance = rebalance
        self.frequency = frequency
        self.min_periods = min_periods
        self.min_assets = min_assets

    @property
    def date(self) -> Optional[pd.Timestamp]:
        """date property"""
        return self.account.date

    @date.setter
    def date(self, date: pd.Timestamp) -> None:
        """date property"""
        self.account.date = date
        self.account.prices = self.prices.loc[self.date]

    @property
    def reb_prices(self) -> Optional[pd.DataFrame]:
        """rebalancing prices"""
        reb_prices = self.prices.loc[: self.date].copy()
        reb_prices.dropna(thresh=self.min_periods, axis=1, inplace=True)
        if len(reb_prices) < self.min_assets:
            return None
        if len(reb_prices.columns) < self.min_assets:
            return None
        return reb_prices

    ################################################################################

    @property
    def value(self) -> pd.Series:
        """strategy value time-series"""
        val = pd.Series(
            data=self.account.records.value, name="value", dtype=float
        ).sort_index()
        return val

    @property
    def weights(self) -> pd.DataFrame:
        """strategy weights time-series"""
        return pd.DataFrame(
            data=self.account.records.weights, dtype=float
        ).T.sort_index()

    @property
    def capitals(self) -> pd.DataFrame:
        """strategy capitals time-series"""
        return pd.DataFrame(
            data=self.account.records.capitals, dtype=float
        ).T.sort_index()

    @property
    def profits(self) -> pd.DataFrame:
        """strategy profits time-series"""
        return pd.DataFrame(
            data=self.account.records.profits, dtype=float
        ).T.sort_index()

    @property
    def allocations(self) -> pd.DataFrame:
        """strategy allocations time-series"""
        return pd.DataFrame(
            data=self.account.records.allocations, dtype=float
        ).T.sort_index()

    ################################################################################

    def simulate(self, *args, **kwargs) -> "Strategy":
        """"""
        reb_dates = [
            self.prices.loc[date:].index[0]
            for date in pd.date_range(
                start=str(self.prices.index[0]),
                end=str(self.prices.index[-1]),
                freq=self.frequency,
            )
        ]
        for idx, self.date in enumerate(self.prices.index, 1):
            if not self.account.allocations.empty:
                self.account.trades = self.account.allocations.subtract(
                    self.account.weights, fill_value=0
                ).abs()
                self.account.weights = self.account.allocations
                self.account.capitals = self.account.value * self.account.weights
                self.account.shares = self.account.capitals.divide(
                    self.prices.loc[self.date].dropna()
                )
                self.account.reset_allocations()
            if self.date in reb_dates or self.account.shares.empty:
                reb_prices = self.reb_prices
                if reb_prices is not None:
                    try:
                        self.account.allocations = self.rebalance(prices=reb_prices, *args, **kwargs)
                    except:
                        self.account.reset_allocations()

        return self


import numpy as np
import yfinance as yf


def ddd(prices: pd.DataFrame):
    """Default rebalancing method"""
    asset = prices.iloc[-1].dropna().index
    uniform_weight = np.ones(len(asset))
    uniform_weight /= uniform_weight.sum()
    weight = pd.Series(index=asset, data=uniform_weight)
    return weight


prices = yf.download("SPY, AGG")["Adj Close"].dropna()

strategy = Strategy(prices=prices, rebalance=ddd).simulate()
strategy.value.plot()
print(strategy.value)