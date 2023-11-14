"""abl strategy"""
import sys
import os
import logging
from datetime import date, timedelta
from pydantic import BaseModel
import numpy as np
import pandas as pd
import riskfolio as rf
from dateutil import parser
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler


sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from core.analytics.pa import metrics
from core.strategy import BaseStrategy
from config import get_args, ALLOC_FOLDER
from hive import db

logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)
YESTERDAY = TODAY - timedelta(days=1)
REGIME = args.regime


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
                equity=0.50,
                fixedincome=0.20,
                alternative=0.05,
                liquidity=0.25,
            )
        if level == 5:
            return cls(
                equity=0.80,
                fixedincome=0.10,
                alternative=0.05,
                liquidity=0.05,
            )
        raise NotImplementedError(
            "level only takes integers from 1 to 5. " +
            f"but {level} was given."
        )


class AblStrategy(BaseStrategy):

    frequency: str = "M"
    min_assets: int = 15
    min_periods: int = 252
    commission: int = 0

    @classmethod
    def load(
        cls,
        universe: pd.DataFrame,
        price_asset: pd.DataFrame,
        price_factor: pd.DataFrame,
        regime: str = "lei",
        asofdate: ... = date.today(),
        market: str = "us",
        level: int = 5,
        halflife: int = 21 * 3,
        regime_window: int = 252 * 5,
        regr_window: int = 21 * 6,
        positive: bool = True,
        clip: bool = False,
    ) -> "AblStrategy":
        """load predefined settings"""

        obj = cls(
            price_asset=price_asset,
            frequency=cls.frequency,
            min_assets=cls.min_assets,
            min_periods=cls.min_periods,
            commission=cls.commission,
            name=f"abl_{market}_{level}",
        )
        
        obj.asset_classes = ["equity", "fixedincome",
                             "alternative", "liquidity"]
        obj.acna = AssetClassNumAsset().dict()
        obj.acsw = AssetClassSumWeight.from_level(level=level).dict()
        obj.price_factor = price_factor
        obj.universe = universe
        
        if regime=='lei':
            from core.model.regime.lei import USLEIHP
            obj.regime = USLEIHP(asofdate=TODAY)
        elif regime=='GMM':
            from core.model.regime.gmm import GMM
            obj.regime = GMM(end=TODAY)
        else:
            from core.model.regime.iml import IML
            obj.regime = IML(end=TODAY)

        obj.allo_settings = AlloSettings().dict()
        obj.regr_settings = RegrSettings(positive=positive).dict()
        obj.halflife = halflife
        obj.regime_window = regime_window
        obj.regr_window = regr_window
        obj.clip = clip
        obj.date = asofdate
        return obj

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

    def filter_corr_score(self, corr, threshold, minimum=2):
        """filter correlation score"""
        filtered_corr = corr[corr < threshold]
        if len(filtered_corr) < minimum:
            return self.filter_corr_score(corr, threshold + 0.01, minimum)
        return filtered_corr

    def rebalance(self, price_asset: pd.DataFrame) -> pd.Series:
        """rebalance function"""
        price_factor = self.price_factor.loc[: self.date].copy()
        state = self.regime.get_state(self.date.strftime("%Y-%m-%d"))
        exp_ret_states = self.regime.expected_returns_by_states(
            price_df=price_factor.iloc[-self.regime_window:]
        )
        self.regime.states.to_csv("states.txt")
        views = self.make_views(exp_ret_states.loc[state])
        prior_mu = metrics.expected_returns(
            price_factor, method="empirical"
        )
        prior_cov = metrics.covariance_matrix(
            price_factor, method="exponential", halflife=self.halflife,
        )
        post_mu, _ = metrics.blacklitterman(prior_mu, prior_cov, views)

        betas = metrics.regression(
            dependent_y=price_asset.iloc[-self.regr_window:],
            independent_x=price_factor,
            **self.regr_settings,
        )
        print("post_mu: \n", post_mu)

        if self.clip: betas = betas.clip(lower=-1, upper=1)
        score = betas.score.copy()
        betas = betas.drop("score", axis=1)

        expected_returns = post_mu.dot(betas.T)
        expected_returns = expected_returns.subtract(expected_returns.multiply(score).abs())
        final_weights = {}
        for ac in self.asset_classes:
            ac_tickers = self.universe[self.universe.strg_asset_class == ac].index
            ac_num_select = self.acna[ac]
            ac_weight = self.acsw[ac]

            if len(ac_tickers) == 1:
                final_weights.update(
                    pd.Series(index=ac_tickers, data=ac_weight))
                continue
            ac_tickers_rank = expected_returns.filter(items=ac_tickers, axis=0).sort_values(
                ascending=False
            )
            correlation_ranked_asset = (
                price_asset.iloc[-252:].pct_change()[ac_tickers_rank.index].corr()
            )
            correlation_ranked_asset = pd.Series(
                index=correlation_ranked_asset.index,
                data=np.max(np.tril(correlation_ranked_asset, k=-1), axis=1),
            )
            correlation_ranked_asset = self.filter_corr_score(
                correlation_ranked_asset, threshold=0.90, minimum=ac_num_select
            ).index

            ac_tickers_rank = ac_tickers_rank.loc[correlation_ranked_asset]

            asset_selection = ac_tickers_rank.nlargest(ac_num_select).index

            weights = self.allocate_weights(
                price_asset[asset_selection].pct_change().iloc[-252:],
                **self.allo_settings,
            )
            weights = weights * ac_weight
            weights = self.clean_weights(weights=weights, decimals=4)
            final_weights.update(pd.Series(weights))

        self.weights = pd.Series(final_weights).fillna(0)

        return self.weights


    def view_prediction_(self, price_asset: pd.DataFrame) -> pd.Series:
        """rebalance function"""
        price_factor = self.price_factor.loc[: self.date].copy()
        state = self.regime.get_state(self.date.strftime("%Y-%m-%d"))
        exp_ret_states = self.regime.expected_returns_by_states(
            price_df=price_factor.iloc[-self.regime_window:]
        )
        self.regime.states.to_csv("states.txt")
        views = self.make_views(exp_ret_states.loc[state])
        prior_mu = metrics.expected_returns(
            price_factor, method="empirical"
        )
        prior_cov = metrics.covariance_matrix(
            price_factor, method="exponential", halflife=self.halflife,
        )
        post_mu, _ = metrics.blacklitterman(prior_mu, prior_cov, views)

        betas = metrics.regression(
            dependent_y=price_asset.loc[self.date-timedelta(days=self.regr_window):self.date],
            independent_x=price_factor,
            **self.regr_settings,
        )

        if self.clip: betas = betas.clip(lower=-1, upper=1)
        score = betas.score.copy()
        betas = betas.drop("score", axis=1)

        expected_returns = post_mu.dot(betas.T)
        expected_returns = expected_returns.subtract(expected_returns.multiply(score).abs())

        return expected_returns


    def allocate_kr(self, universe_kr: pd.DataFrame, price_asset_kr: pd.DataFrame) -> pd.Series:

        assert hasattr(self, "weights")

        price_asset_kr = price_asset_kr.loc[:self.date]
        price_asset_us = self.price_asset.loc[:self.date]
        universe_us = self.universe.loc[self.weights.index]
        final_weights = {}
        for ac in self.asset_classes:
            ac_weight = self.acsw[ac]
            ac_tickers_us = universe_us[universe_us.strg_asset_class == ac].index
            ac_tickers_kr = universe_kr[universe_kr.strg_asset_class == ac].index

            asset_selection = []

            for ticker_us in ac_tickers_us:

                price_us = price_asset_us[ticker_us]

                distances = {}

                for ticker_kr in ac_tickers_kr:
                    price_kr = price_asset_kr[ticker_kr]
                    itx = price_kr.dropna().index.intersection(price_us.dropna().index)
                    if len(itx) < 252:
                        continue
                    p_us = price_us.loc[itx]
                    p_kr = price_kr.loc[itx]
                    norm_price_us = (p_us - p_us.mean()) / p_us.std()
                    norm_price_kr = (p_kr - p_kr.mean()) / p_kr.std()
                    distance = euclidean_distances(norm_price_kr.values.reshape(
                        1, -1), norm_price_us.values.reshape(1, -1))
                    distances[ticker_kr] = distance[0][0]
                distances = pd.Series(distances).sort_values()
                asset_selection.append(distances.index[0])
                ac_tickers_kr = ac_tickers_kr.drop(distances.index[0])
            weights = self.allocate_weights(
                price_asset_kr[asset_selection].iloc[-252:].pct_change().fillna(0),
                **self.allo_settings,
            )
            weights = weights * ac_weight
            weights = self.clean_weights(weights=weights, decimals=4)
            final_weights.update(weights)

        return pd.Series(final_weights).fillna(0)


def run_abl_allocation() -> None:
    """
    run mlp allocation at the month start trading date
    i.e. first trading day each month.
    """
    
    extra = dict(user=args.user, activity="abl_allocation", category="script")

    if TODAY.date() != db.get_start_trading_date(market="KR", asofdate=TODAY):
        logger.info(msg=f"[SKIP] ABL allocation. {TODAY:%Y-%m-%d}", extra=extra)
        return

    logger.info(msg=f"[PASS] Start ABL allocation. {TODAY:%Y-%m-%d}", extra=extra)
    
    ticker_mapper = db.get_meta_mapper()
    OUTPUT_COLS = ["isin", "ticker_bloomberg", "asset_class", "risk_score", "name"]
    allocation_weights = []

    universe = db.load_universe("abl_us")
    price_asset = db.get_price(tickers=", ".join(list(universe.index))).loc[:YESTERDAY]
    universe_kr = db.load_universe("abl_kr")
    price_asset_kr = db.get_price(tickers=", ".join(list(universe_kr.index))).loc[:YESTERDAY]
    price_factor = db.get_lens(TODAY)

    for level in [3, 4, 5]:
        strategy = AblStrategy.load(
            universe=universe,
            price_asset=price_asset,
            price_factor=price_factor,
            regime=REGIME,
            asofdate=TODAY,
            level=level,
        )

        us_weights = strategy.allocate()
        kr_weights = strategy.allocate_kr(
            universe_kr=universe_kr, price_asset_kr=price_asset_kr
        )

        us_risk_score = 0.0
        for asset, weight in us_weights.items():
            us_risk_score += weight * strategy.universe.loc[str(asset)].risk_score
        
        msg = f"\n[PASS] ABL MP"
        msg += f"\n{TODAY.date()} | US level {level}"
        msg += f"\nrisk score {us_risk_score:.4f}\n"
        logger.info(msg, extra=extra)
        print(us_weights.to_markdown())

        kr_risk_score = 0.0
        for asset, weight in kr_weights.items():
            kr_risk_score += weight * universe_kr.loc[str(asset)].risk_score

        msg = f"\n[PASS] ABL MP"
        msg += f"\n{TODAY.date()} | KR level {level}"
        msg += f"\nrisk score {kr_risk_score:.4f}\n"
        logger.info(msg, extra=extra)
        print(kr_weights.to_markdown())

        if args.database == "true":
            us_uni = strategy.universe[OUTPUT_COLS]
            us_uni[f"{TODAY:%Y-%m-%d} weight"] = us_uni.index.map(us_weights.to_dict())
            us_uni = us_uni.dropna()
            allocation_weights.append((f"ABL_US_{level}_allocation.csv", us_uni))
            alloc_path = os.path.join(ALLOC_FOLDER, "abl")
            if not os.path.exists(alloc_path):
                os.makedirs(alloc_path)
            csv_file_path = os.path.join(alloc_path, f"{TODAY:%Y%m%d}_ABL_US_{level}_allocation.csv")
            us_uni.to_csv(csv_file_path, index=False, encoding="utf-8-sig")

            us_weights = us_weights.to_frame().reset_index()
            us_weights.columns = ["ticker", "weights"]
            us_weights["rebal_dt"] = TODAY
            portfolio_id = db.get_portfolio_id(portfolio=f"abl_us_{level}")
            us_weights["port_id"] = portfolio_id
            us_weights["stk_id"] = us_weights.ticker.map(ticker_mapper)
            
            try:
                db.TbPortAlloc.insert(us_weights)
            except:
                try:
                    db.TbPortAlloc.update(us_weights)
                except:
                    db_alloc = db.get_alloc_weight_for_shares(strategy="ABL", market="US", level=level)
                    db_alloc = db_alloc[db_alloc.rebal_dt == TODAY]

                    merge_df = us_weights.merge(db_alloc, on=["rebal_dt", "port_id", "stk_id"], how="outer")
                    delete_asset = merge_df[merge_df.weights_x.isnull()].stk_id.tolist()
                    update_asset = merge_df.dropna()
                    update_asset['weights'] = update_asset['weights_x']
                    insert_asset = merge_df[merge_df.weights_y.isnull()]
                    insert_asset['weights'] = insert_asset['weights_x']

                    db.delete_asset_port_alloc(rebal_dt=TODAY, port_id=portfolio_id, stk_id=delete_asset)
                    db.TbPortAlloc.update(update_asset)
                    db.TbPortAlloc.insert(insert_asset)

            kr_uni = universe_kr[OUTPUT_COLS]
            kr_uni[f"{TODAY:%Y-%m-%d} weight"] = kr_uni.index.map(kr_weights.to_dict())
            kr_uni = kr_uni.dropna()
            allocation_weights.append((f"ABL_KR_{level}_allocation.csv", kr_uni))
            alloc_path = os.path.join(ALLOC_FOLDER, "abl")
            if not os.path.exists(alloc_path):
                os.makedirs(alloc_path)
            csv_file_path = os.path.join(alloc_path, f"{TODAY:%Y%m%d}_ABL_KR_{level}_allocation.csv")
            kr_uni.to_csv(csv_file_path, index=False, encoding="utf-8-sig")

            kr_weights = kr_weights.to_frame().reset_index()
            kr_weights.columns = ["ticker", "weights"]
            kr_weights["rebal_dt"] = TODAY
            portfolio_id = db.get_portfolio_id(portfolio=f"abl_kr_{level}")
            kr_weights["port_id"] = portfolio_id
            kr_weights["stk_id"] = kr_weights.ticker.map(ticker_mapper)
            
            try:
                db.TbPortAlloc.insert(kr_weights)
            except:
                try:
                    db.TbPortAlloc.update(kr_weights)
                except:
                    db_alloc = db.get_alloc_weight_for_shares(strategy="ABL", market="KR", level=level)
                    db_alloc = db_alloc[db_alloc.rebal_dt == TODAY]

                    merge_df = kr_weights.merge(db_alloc, on=["rebal_dt", "port_id", "stk_id"], how="outer")
                    delete_asset = merge_df[merge_df.weights_x.isnull()].stk_id.tolist()
                    update_asset = merge_df.dropna()
                    update_asset['weights'] = update_asset['weights_x']
                    insert_asset = merge_df[merge_df.weights_y.isnull()]
                    insert_asset['weights'] = insert_asset['weights_x']

                    db.delete_asset_port_alloc(rebal_dt=TODAY, port_id=portfolio_id, stk_id=delete_asset)
                    db.TbPortAlloc.update(update_asset)
                    db.TbPortAlloc.insert(insert_asset)
                    
    logger.info(msg=f"[PASS] End ABL allocation. {TODAY:%Y-%m-%d}", extra=extra)