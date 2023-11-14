"""abl strategy"""
import yfinance as yf
from datetime import date, timedelta
from dateutil import parser
from pydantic import BaseModel
import pandas as pd
import riskfolio as rf
from core.analytics import USLEIHP
from core.analytics.pa import metrics
from core.strategy import BaseStrategy
from hive import db


class AlloSettings(BaseModel):
    """allocation settings"""

    model: str = "HRP"
    codependence: str = "pearson"
    covariance: str = "hist"
    rm: str = "MV"
    rf: float = 0.0
    linkage: str = "ward"
    max_k: int = 10
    leaf_order: bool = True


class RegrSettings(BaseModel):
    """regression settings"""

    method: str = "lasso"
    positive: bool = False
    fit_intercept: bool = False


class AssetClassNumAsset(BaseModel):
    """asset class num asset"""

    equity: int = 4
    fixedincome: int = 2
    alternative: int = 2
    liquidity: int = 2


class AssetClassSumWeight(BaseModel):
    """total of asset weight for each asset class"""

    equity: float
    fixedincome: float
    alternative: float
    liquidity: float

    @classmethod
    def from_level(cls, level: int = 5) -> "AssetClassSumWeight":
        """get asset class weight based on the risk level"""
        if level == 1:
            return cls(
                equity=0.05,
                fixedincome=0.10,
                alternative=0.05,
                liquidity=0.80,
            )
        if level == 2:
            return cls(
                equity=0.10,
                fixedincome=0.10,
                alternative=0.10,
                liquidity=0.70,
            )
        if level == 3:
            return cls(
                equity=0.30,
                fixedincome=0.15,
                alternative=0.05,
                liquidity=0.50,
            )
        if level == 4:
            return cls(
                equity=0.55,
                fixedincome=0.20,
                alternative=0.05,
                liquidity=0.20,
            )
        if level == 5:
            return cls(
                equity=0.80,
                fixedincome=0.10,
                alternative=0.05,
                liquidity=0.05,
            )
        raise NotImplementedError(
            "level only takes integers from 1 to 5. " + f"but {level} was given."
        )


class AblStrategy(BaseStrategy):
    frequency: str = "M"
    min_assets: int = 15
    min_periods: int = 252
    commission: int = 10

    @classmethod
    def load(
        cls,
        market: str = "us",
        level: int = 5,
        asofdate: str = date.today().strftime("%Y-%m-%d"),
        halflife: int = 21,
        regime_window: int = 252 * 5,
        regr_window: int = 21 * 3,
        positive: bool = True,
        clip: bool = True,
    ) -> "AblStrategy":
        """load predefined settings"""
        # strategy = f"abl_{market}"
        cls.asset_classes = ["equity", "fixedincome", "alternative", "liquidity"]
        cls.acna = AssetClassNumAsset().dict()
        cls.acsw = AssetClassSumWeight.from_level(level=level).dict()
        cls.price_factor = pd.read_csv(
            "factor_price.csv", index_col="date", parse_dates=True
        ).dropna()

        # cls.universe = db.load_universe("mlp_us")
        cls.universe = pd.read_csv("universe.csv", index_col="ticker")
        price_asset = yf.download(cls.universe.index.tolist())["Adj Close"]
        cls.regime = USLEIHP()
        cls.allo_settings = AlloSettings().dict()
        cls.regr_settings = RegrSettings(positive=positive).dict()
        cls.halflife = halflife
        cls.regime_window = regime_window
        cls.regr_window = regr_window
        cls.clip = clip
        return cls(
            price_asset=price_asset,
            frequency=cls.frequency,
            min_assets=cls.min_assets,
            min_periods=cls.min_periods,
            commission=cls.commission,
            name=f"abl_{market}_{level}",
        )

    @staticmethod
    def allocate_weights(pri_returns: pd.DataFrame, **kwargs) -> pd.Series:
        """calculate weights"""
        port = rf.HCPortfolio(returns=pri_returns)
        return port.optimization(**kwargs)["weights"]

    @staticmethod
    def make_views(views: pd.Series) -> list:
        """make list of views on expected returns"""
        return [
            {"assets": factor, "sign": ">", "value": expected_return}
            for factor, expected_return in views.items()
        ]

    def rebalance(self, price_asset: pd.DataFrame) -> pd.Series:
        """rebalance function"""
        state = self.regime.get_state(self.date.strftime("%Y-%m-%d"))
        exp_ret_states = self.regime.expected_returns_by_states(
            price_df=self.price_factor.loc[: self.date].iloc[-self.regime_window :]
        )
        views = self.make_views(exp_ret_states.loc[state])
        prior_mu = metrics.expected_returns(
            self.price_factor.loc[: self.date], method="empirical"
        )
        prior_cov = metrics.covariance_matrix(
            self.price_factor.loc[: self.date],
            method="exponential",
            halflife=self.halflife,
        )
        post_mu, _ = metrics.blacklitterman(prior_mu, prior_cov, views)

        betas = metrics.regression(
            dependent_y=price_asset.iloc[-self.regr_window :],
            independent_x=self.price_factor.loc[: self.date],
            **self.regr_settings,
        )
        betas.to_csv("betas.csv")
        if self.clip:
            betas = betas.clip(lower=-1, upper=1)
        betas = betas.drop("score", axis=1)
        expected_returns = post_mu @ betas.T
        expected_returns.to_csv("expected_returns.csv")
        final_weights = {}
        for ac in self.asset_classes:
            ac_tickers = self.universe[self.universe.strg_asset_class == ac].index
            ac_num_select = self.acna[ac]
            ac_weight = self.acsw[ac]
            if len(ac_tickers) == 1:
                final_weights.update(pd.Series(index=ac_tickers, data=ac_weight))
                continue
            ac_tickers_rank = expected_returns.filter(
                items=ac_tickers, axis=0
            ).sort_values(ascending=False)
            asset_selection = ac_tickers_rank.nlargest(ac_num_select).index

            weights = self.allocate_weights(
                price_asset[asset_selection].pct_change().iloc[-252:],
                **self.allo_settings,
            )
            weights = weights * ac_weight
            weights = self.clean_weights(weights=weights, decimals=4)
            final_weights.update(pd.Series(weights))

        print(final_weights)
        return pd.Series(final_weights).fillna(0)


strategy = AblStrategy.load(
    market="US",
    level=4,
    halflife=21,
    regime_window=5 * 252,
    regr_window=21 * 6,
    clip=True,
    positive=False,
)
strategy.simulate(start="2023-03-01")

strategy.value_df.plot()
