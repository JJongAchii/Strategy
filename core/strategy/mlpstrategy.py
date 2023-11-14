"""mlp strategy"""
import os
import sys
import argparse
import logging
from typing import Optional
from calendar import monthrange
from pydantic.main import BaseModel
from dateutil import parser
import numpy as np
import pandas as pd
import riskfolio as rf
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import get_args, ML_MODELS_FOLDER, ALLOC_FOLDER, OUTPUT_FOLDER
from core.strategy import utils
from core.strategy.base import BaseStrategy
from core.model.ML.mlp_prediction import BaggedMlpClassifier
from core.strategy.utils import mean_variance_optimizer, data_resampler, optimal_portfolio
from hive import db


logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)
YESTERDAY = TODAY - timedelta(days=1)

OUTPUT_COLS = ["isin", "asset_class_name", "risk_score", "name"]


class PredSettings(BaseModel):
    """prediction settings"""

    train: bool = False
    save: bool = False
    model_path: str = ML_MODELS_FOLDER
    lags: int = 21
    lookback_window: int = 50
    hidden_layer_sizes: tuple = (256,)
    random_state: int = 100
    max_iter: int = 1000
    early_stopping: bool = True
    validation_fraction: float = 0.15
    shuffle: bool = False
    n_estimators: int = 100
    max_samples: float = 0.50
    max_features: float = 0.50
    bootstrap: bool = False
    bootstrap_features: bool = False
    n_jobs: int = 4


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


class AssetClsNumAsset(BaseModel):
    """number of asset in each asset class"""

    equity: int
    fixedincome: int
    alternative: int
    liquidity: int
        
    @classmethod
    def from_level(cls, level: int) -> "AssetClsNumAsset":
        """_summary_

        Args:
            level (int): _description_

        Returns:
            AssetClsNumAsset: _description_
        """
         
        if level == 1:
            return cls (
                equity = 0,
                fixedincome = 0,
                alternative = 0,
                liquidity = 10,
            )
        if level == 2:
            return cls (
                equity = 2,
                fixedincome = 2,
                alternative = 2,
                liquidity = 4,
            )
        if level == 3:
            return cls (
                equity = 4,
                fixedincome = 2,
                alternative = 2,
                liquidity = 2,
            )
        if level == 4:
            return cls (
                equity = 4,
                fixedincome = 2,
                alternative = 2,
                liquidity = 2,
            )
        if level == 5:
            return cls (
                equity = 4,
                fixedincome = 2,
                alternative = 2,
                liquidity = 2,
            )


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
                equity=0.0,
                fixedincome=0.0,
                alternative=0.0,
                liquidity=1.0,
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
                fixedincome=0.05,
                alternative=0.05,
                liquidity=0.60,
            )
        if level == 4:
            return cls(
                equity=0.40,
                fixedincome=0.10,
                alternative=0.20,
                liquidity=0.30,
            )
        if level == 5:
            return cls(
                equity=0.70,
                fixedincome=0.05,
                alternative=0.20,
                liquidity=0.05,
            )
        raise NotImplementedError(
            "level only takes integers from 1 to 5. " + f"but {level} was given."
        )


def get_ml_model_path(asofdate: str) -> str:
    """get the model path"""
    parsed_date = parser.parse(asofdate).date()
    year = int(parsed_date.year)
    month = int(parsed_date.month)
    if month >= 7:
        _, day = monthrange(year=year, month=6)
        return os.path.join(OUTPUT_FOLDER, "ml_models", f"{year}-06-{day}")
    _, day = monthrange(year=year - 1, month=12)
    return os.path.join(OUTPUT_FOLDER, "ml_models", f"{year-1}-{12}-{day}")
 
    
class MlpStrategy(BaseStrategy):
    """kbtestbed strategy"""

    __assetclass__ = ["equity", "fixedincome", "alternative", "liquidity"]

    @classmethod
    def load(
        cls,
        universe: pd.DataFrame,
        prices: pd.DataFrame,
        model_path: str,
        level: int = 5,
        max_samples: float = 0.5
    ) -> "MlpStrategy":
        """load the strategy"""

        cls.asset_cls = ["equity", "fixedincome", "alternative", "liquidity"]
        cls.ac_sum_weight = AssetClassSumWeight.from_level(level).dict()
        cls.ac_num_select = AssetClsNumAsset.from_level(level).dict()
        cls.allo_settings = AlloSettings().dict()
        cls.pred_settings = PredSettings(model_path=model_path, max_samples=max_samples)
        cls.pred_model = BaggedMlpClassifier(**cls.pred_settings.dict())
        cls.prediction = pd.Series
        cls.universe = universe

        if level == 1:
            
            return cls(
                price_asset=prices,
                frequency="M",
                min_periods=253,
                commission=0,
            )
        
        else:
            
            return cls(
                price_asset=prices,
                frequency="M",
                min_periods=1000,
                commission=0,
            )

    def allocate_weights(self, pri_return: pd.DataFrame, allocation_method: str) -> pd.Series:
        """allocate weight"""
        if allocation_method == 'HRP':
            port = rf.HCPortfolio(returns=pri_return)
            weights = port.optimization(**self.allo_settings)["weights"]
            return weights
            
        elif allocation_method == 'MVO':     
            data = data_resampler(pri_return.mean(), pri_return.cov())
            weights,_,_ = mean_variance_optimizer(data)
            weights.columns = ["weights"]
            return weights
            

    def filter_corr_score(self, corr, threshold, minimum=2):
        """filter correlation score"""
        filtered_corr = corr[corr < threshold]
        if len(filtered_corr) < minimum:
            return self.filter_corr_score(corr, threshold + 0.01, minimum)
        return filtered_corr
    
    def rebalance(self, price_asset: pd.DataFrame, allocation_method: str = "HRP") -> pd.Series:
        """rebalancing function"""        
        self.prediction = self.pred_model.predict(price_asset)
        final_weights = {}
        
        for asset_class in self.universe.strg_asset_class.unique():
            ac_tickers = self.universe[
                self.universe.strg_asset_class == asset_class
            ].index
            ac_num_select = self.ac_num_select[asset_class]
            ac_sum_weight = self.ac_sum_weight[asset_class]
            ac_tickers_rank = self.prediction.filter(items=ac_tickers, axis=0).sort_values(
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
                correlation_ranked_asset, threshold=0.96, minimum=ac_num_select
            ).index

            ac_tickers_rank = ac_tickers_rank.loc[correlation_ranked_asset]

            asset_selection = ac_tickers_rank.nlargest(ac_num_select).index

            weights = (
                self.allocate_weights(
                    price_asset[asset_selection].pct_change().iloc[-252:],
                    allocation_method,
                )
                * ac_sum_weight
            )
            weights = self.clean_weights(weights=weights, decimals=4)
            final_weights.update(weights.to_dict())
        return pd.Series(final_weights)
    

    def view_prediction_(self, price_asset: pd.DataFrame) -> pd.Series:
        """rebalancing function"""        
        self.prediction = self.pred_model.predict(price_asset)
        return pd.Series(self.prediction)


def liquidity_allocate(strategy: MlpStrategy, today: datetime, allocation_method: str = "HRP") -> Optional[pd.Series]:
    """allocate weights based on date if date not provided use latest"""    
    import numpy as np
    from hive import db
    #date = datetime.today()
    price_slice = (
        strategy.price_asset.loc[:today]
        .dropna(thresh=strategy.min_periods, axis=1)          
    ).copy()
    if price_slice.empty:
        return pd.Series(dtype=float)
    
    prediction = strategy.pred_model.predict(price_slice)
    prediction_updated = prediction.copy()
    if len(prediction_updated.index) < len(price_slice.columns):
        
        prediction_updated_index = prediction_updated.index.tolist()
        price_slice_index = price_slice.columns.tolist()
        missing_index = [a for a in price_slice_index if a not in prediction_updated_index]
        df_missing_data = db.get_price(tickers=", ".join(missing_index))
        df_missing_data = df_missing_data[df_missing_data.index <= today].dropna(how = 'all', axis=1)

        missing_index = df_missing_data.columns.tolist()
        price_slice_index = prediction_updated_index + missing_index

        arr = []
        for i in range(len(missing_index)):
            
            arr.append(1)
        
        missing = pd.Series(arr, index = missing_index)
        prediction_updated = prediction_updated.append(missing)
        prediction_updated = prediction_updated[prediction_updated.index.isin(price_slice_index)]
    
    final_weights = {}
    
    ac_num_select = strategy.ac_num_select["liquidity"]
    ac_sum_weight = strategy.ac_sum_weight["liquidity"]
    ac_tickers_rank = prediction_updated.filter(items=strategy.price_asset.columns, axis=0).sort_values(
        ascending=False
    )
    
    correlation_ranked_asset = (
        price_slice.iloc[-252:].pct_change()[ac_tickers_rank.index].corr()
    )
    correlation_ranked_asset = pd.Series(
        index=correlation_ranked_asset.index,
        data=np.max(np.tril(correlation_ranked_asset, k=-1), axis=1),
    )
    correlation_ranked_asset = strategy.filter_corr_score(
        correlation_ranked_asset, threshold=0.96, minimum=ac_num_select
    ).index
    
    ac_tickers_rank = ac_tickers_rank.loc[correlation_ranked_asset]
    asset_selection = ac_tickers_rank.nlargest(ac_num_select).index
 
    asset_weights = (
            strategy.allocate_weights(
                price_slice[asset_selection].pct_change().iloc[-252:],
                allocation_method,
            )
            * ac_sum_weight
        )
    
    weights = strategy.clean_weights(weights=asset_weights, decimals=4)
    final_weights.update(weights.to_dict())
    
    return pd.Series(final_weights)  


def run_mlp_allocation() -> None:
    """
    run mlp allocation at the month start trading date
    i.e. first trading day each month.
    """
    extra = dict(user=args.user, activity="mlp_allocation", category="script")

    if TODAY.date() != db.get_start_trading_date(market="KR", asofdate=TODAY):
        logger.info(msg=f"[SKIP] MLP allocation. {TODAY:%Y-%m-%d}", extra=extra)
        return

    logger.info(msg=f"[PASS] Start MLP allocation. {TODAY:%Y-%m-%d}", extra=extra)

    ticker_mapper = db.get_meta_mapper()
    OUTPUT_COLS = ["isin", "ticker_bloomberg", "asset_class", "risk_score", "name"]
    allocation_weights = []

    model_path = os.path.join(
        ML_MODELS_FOLDER, utils.get_ml_model_training_date(asofdate=TODAY).strftime("%Y-%m-%d")
    )

    for market in ["US", "KR"]:
        universe = db.load_universe(f"MLP_{market}")
        prices = db.get_price(tickers=", ".join(universe.index.tolist())).loc[:YESTERDAY]

        for level in [1,2,3,4,5]:
            
            if level == 1:
                
                from mlpstrategy import liquidity_allocate
                
                portfolio_id = db.get_portfolio_id(portfolio=f"MLP_{market}_{level}")
                weights = pd.Series()
                prob = pd.Series()
                
                ac_universe = universe[
                        universe.strg_asset_class == "liquidity"
                    ]  
                
                strategy = MlpStrategy.load(
                            universe=ac_universe, prices=prices[ac_universe.index], level=level, model_path=model_path, max_samples= 1.0
                        )
                
                ac_weights = liquidity_allocate(strategy,TODAY)
                weights = weights.append(ac_weights)
                
            else:
                
                portfolio_id = db.get_portfolio_id(portfolio=f"MLP_{market}_{level}")
                weights = pd.Series()
                prob = pd.Series()
                for asset_class in universe.strg_asset_class.unique():
                    ac_universe = universe[
                        universe.strg_asset_class == asset_class
                    ]
                    if asset_class in ["fixedincome", "liquidity"]:
                        strategy = MlpStrategy.load(
                            universe=ac_universe, prices=prices[ac_universe.index], level=level, model_path=model_path, max_samples= 1.0
                        )
                    elif asset_class in ["equity", "alternative"]:
                        strategy = MlpStrategy.load(
                            universe=ac_universe, prices=prices[ac_universe.index], level=level, model_path=model_path, max_samples= 0.5
                        )
                    ac_weights = strategy.allocate()
                    weights = weights.append(ac_weights)
                    prob = prob.append(strategy.prediction)
                
            if args.script == "dws":
                from core.strategy.dwsstrategy import run_dws_allocation
                run_dws_allocation(market, level, weights, universe, prices)

            weights_as_cl = pd.concat([weights, universe["strg_asset_class"]], axis=1, join="inner", keys=['weights', 'strg_asset_class'])
            weights = pd.Series()
            
            for asset_class in weights_as_cl.strg_asset_class.unique():
                split_weights = weights_as_cl[weights_as_cl.strg_asset_class == asset_class]["weights"]
                adjusted_weights = strategy.calc_adjusted_portfolio_weight(weights=split_weights)
                weights = weights.append(adjusted_weights)
            weights = strategy.clean_weights(weights=weights, decimals=4)
            
            risk_score = 0.0
            for asset, weight in weights.items():
                risk_score += weight * universe.loc[str(asset)].risk_score

            msg = f"\n[PASS] MLP MP"
            msg += f"\n{TODAY.date()} | {market} level {level}"
            msg += f"\nrisk score {risk_score:.4f}\n"
            logger.info(msg, extra=extra)
            print(weights.to_markdown())
            
            if args.database == "true":
                
                uni = universe[OUTPUT_COLS]
                uni[f"{TODAY:%Y-%m-%d} weight"] = uni.index.map(weights.to_dict())
                uni = uni.dropna()
                allocation_weights.append((f"{market}_{level}_allocation.csv", uni))
                alloc_path = os.path.join(ALLOC_FOLDER, "mlp")
                if not os.path.exists(alloc_path):
                    os.makedirs(alloc_path)
                csv_file_path = os.path.join(alloc_path, f"{TODAY:%Y%m%d}_MLP_{market}_{level}_allocation.csv")
                uni.to_csv(csv_file_path, index=False, encoding="utf-8-sig")
                allo = weights.to_frame().reset_index()
                allo.columns = ["ticker", "weights"]
                allo["rebal_dt"] = TODAY
                allo["port_id"] = portfolio_id
                allo["stk_id"] = allo.ticker.map(ticker_mapper)
                
                try:
                    db.TbPortAlloc.insert(allo)
                except:
                    try:
                        db.TbPortAlloc.update(allo)
                    except:
                        db_alloc = db.get_alloc_weight_for_shares(strategy="MLP", market=market, level=level)
                        db_alloc = db_alloc[db_alloc.rebal_dt == TODAY]

                        merge_df = allo.merge(db_alloc, on=["rebal_dt", "port_id", "stk_id"], how="outer")
                        delete_asset = merge_df[merge_df.weights_x.isnull()].stk_id.tolist()
                        update_asset = merge_df.dropna()
                        update_asset['weights'] = update_asset['weights_x']
                        insert_asset = merge_df[merge_df.weights_y.isnull()]
                        insert_asset['weights'] = insert_asset['weights_x']

                        db.delete_asset_port_alloc(rebal_dt=TODAY, port_id=portfolio_id, stk_id=delete_asset)
                        db.TbPortAlloc.update(update_asset)
                        db.TbPortAlloc.insert(insert_asset)

                """update db for increase probability of assets"""
                univ = universe[["isin", "strg_asset_class", "iso_code", "wrap_asset_class_code"]]
                
                univ["prob"] = univ.index.map(prob.to_dict())
                univ["rebal_dt"] = TODAY
                univ = univ.dropna()
                
                try:
                    db.TbProbIncrease.insert(univ)
                except:
                    try:
                        db.TbProbIncrease.update(univ)
                    except:
                        db_prob = db.get_probability_increase(market=market)
                        db_prob = db_prob[db_prob.rebal_dt == TODAY]
                        
                        merge_df = univ.merge(db_prob, on=["rebal_dt", "isin"], how="outer")
                        delete_isin = merge_df[merge_df.prob_x.isnull()]["isin"].tolist()
                        update_isin = merge_df.dropna()
                        update_isin['prob'] = update_isin['prob_x']
                        insert_isin = merge_df[merge_df.prob_y.isnull()]
                        insert_isin = insert_isin.assign(
                                        prob=insert_isin['prob_x'],
                                        iso_code=market,
                                        wrap_asset_class_code=insert_isin['wrap_asset_class_code_x']
                                    )

                        db.delete_isin_prob_increase(rebal_dt=TODAY, isin=delete_isin)
                        db.TbProbIncrease.update(update_isin)
                        db.TbProbIncrease.insert(insert_isin)
    
    logger.info(msg=f"[PASS] End MLP allocation. {TODAY:%Y-%m-%d}", extra=extra)