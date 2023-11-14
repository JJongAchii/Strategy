import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import sqlalchemy as sa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from hive import db
from core.strategy import utils

logger = logging.getLogger("sqlite")

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(PROJECT_FOLDER, "output")
ML_MODELS_FOLDER = os.path.join(OUTPUT_FOLDER, "irp_models_test")

'''
#### 코드 내 변경사항
1. mlpstrategy.py
	- line 226 : min_periods 삭제
	- line 272~288 는 아래 코드로 변경


    if len(correlation_ranked_asset) > 1:

        correlation_ranked_asset = self.filter_corr_score(

            correlation_ranked_asset, threshold=0.96, minimum=ac_num_select

        ).index

    

        ac_tickers_rank = ac_tickers_rank.loc[correlation_ranked_asset]

    

        asset_selection = ac_tickers_rank.nlargest(ac_num_select).index

    

        weights = (

        self.allocate_weights(

            price_asset[asset_selection].pct_change().iloc[-252:].dropna(),

            allocation_method,

        ) * ac_sum_weight

        )

        weights = self.clean_weights(weights=weights, decimals=4)

        final_weights.update(weights.to_dict())

    else:

        final_weights.update({ac_tickers_rank.index[0] : ac_sum_weight})
    
2. base.py
	- line 43 : min_assets : int = 1

'''



def get_ml_model_training_date(asofdate: datetime) -> datetime:
    """get the supposed date for model training based on the given the asofdate"""
    half = datetime(asofdate.year, 6, 30)
    if min(half, asofdate) == half:
        return half
    return datetime(asofdate.year - 1, 12, 31)


def run_prediction():

    for date in pd.date_range("2023-7-1", "2023-8-1"):

        # run mlp prediction
        YESTERDAY = date - timedelta(days=1)

        if date.month not in [1, 7]:
            print(f"skip prediction_{date}")
            continue
        else:
            if not date.day == db.get_start_trading_date(market="KR", asofdate=date).day:
                print(f"skip prediction_{date}")
                continue

        print(f"start prediction_{date}")

        from core.strategy.mlpstrategy import PredSettings
        from core.model.ML.mlp_prediction import BaggedMlpClassifier
        model_path = os.path.join(
            ML_MODELS_FOLDER, utils.get_ml_model_training_date(asofdate=date).strftime("%Y-%m-%d")
        )
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        market = "IRP"
        universe = db.load_universe(strategy=f"MLP_{market}")
        price_df = db.get_price(tickers=", ".join(universe.index.tolist())).loc[:YESTERDAY]
        for asset_class in universe.strg_asset_class.unique():
            ac_tickers = universe[
                universe.strg_asset_class == asset_class
            ].index
            if asset_class in ["fixedincome", "liquidity"]:
                pred_settings = PredSettings(
                    train=True, save=True, model_path=model_path, max_samples= 1.0
                ).dict()
            elif asset_class in ["equity", "alternative"]:
                pred_settings = PredSettings(
                    train=True, save=True, model_path=model_path, max_samples= 0.5
                ).dict()
                
            prediction_model = BaggedMlpClassifier(**pred_settings)
            prediction_model.predict(price_df=price_df[ac_tickers])


def run_allocation():

    allocations = pd.DataFrame(
        columns=["date", "strategy", "isin", "ticker_bloomberg", "asset_class", "risk_score", "name"])

    for date in pd.date_range("2023-8-1", "2023-8-30"):

        YESTERDAY = date - timedelta(days=1)

        # run mlp allocation
        if date.date() != db.get_start_trading_date(market="KR", asofdate=date):
            print(f"skip allocation_{date}")
            continue

        print(f"start allocation_{date}")

        from core.strategy.mlpstrategy import MlpStrategy

        ticker_mapper = db.get_meta_mapper()
        OUTPUT_COLS = ["strategy", "isin", "ticker_bloomberg", "asset_class", "risk_score", "name"]
        allocation_weights = []

        model_path = os.path.join(
            ML_MODELS_FOLDER, utils.get_ml_model_training_date(asofdate=date).strftime("%Y-%m-%d")
        )

        market = "IRP"
        universe = db.load_universe(strategy=f"MLP_{market}")
        prices = db.get_price(tickers=", ".join(universe.index.tolist())).loc[:YESTERDAY]

        for level in [3]:
            portfolio_id = db.get_portfolio_id(portfolio=f"MLP_{market}_{level}")
            weights = pd.Series()

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
            
            risk_score = 0.0
            for asset, weight in weights.items():
                risk_score += weight * universe.loc[str(asset)].risk_score
            max_date_price = prices.index[-1]
            msg = f"\nallocate weights for market={market}, level={level}, "
            msg += f"with latest available price at date {max_date_price:%Y-%m-%d}."
            msg += f"\nallocation weights is as below, with risk score = {risk_score:.4f}:\n"
            msg += weights.to_markdown()
            print(msg)

            uni = universe[OUTPUT_COLS]
            uni["rebal_dt"] = date
            uni["strategy"] = uni["strategy"] + "_" + str(level)
            uni["weights"] = uni.index.map(weights.to_dict())
            uni["port_id"] = portfolio_id
            uni["stk_id"] = uni.index.map(ticker_mapper)
            uni = uni.dropna()
            allocations = allocations.append(uni)
        allocations.to_csv("irp_allocation_test.csv", index=False, encoding="utf-8-sig")



if __name__ == "__main__":
    # run_prediction()
    run_allocation()