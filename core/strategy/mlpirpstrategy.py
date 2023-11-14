"""mlp strategy"""
import os
import sys
import logging
from dateutil import parser
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import get_args, ML_MODELS_FOLDER, ALLOC_FOLDER, OUTPUT_FOLDER
from core.strategy.mlpstrategy import MlpStrategy
from core.strategy import utils
from hive import db


logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)
YESTERDAY = TODAY - timedelta(days=1)

OUTPUT_COLS = ["isin", "asset_class_name", "risk_score", "name"]

    
def run_mlp_irp_allocation() -> None:
    """
    run mlp allocation at the month start trading date
    i.e. first trading day each month.
    """
    extra = dict(user=args.user, activity="mlp_irp_allocation", category="script")

    if TODAY.date() != db.get_start_trading_date(market="KR", asofdate=TODAY):
        logger.info(msg=f"[SKIP] MLP IRP allocation. {TODAY:%Y-%m-%d}", extra=extra)
        return

    logger.info(msg=f"[PASS] Start MLP IRP allocation. {TODAY:%Y-%m-%d}", extra=extra)

    ticker_mapper = db.get_meta_mapper()
    OUTPUT_COLS = ["isin", "ticker_bloomberg", "asset_class", "risk_score", "name"]
    allocation_weights = []

    model_path = os.path.join(
        ML_MODELS_FOLDER, utils.get_ml_model_training_date(asofdate=TODAY).strftime("%Y-%m-%d")
    )

    market = "KR"
    universe = db.load_universe(f"MLP_IRP")
    prices = db.get_price(tickers=", ".join(universe.index.tolist())).loc[:YESTERDAY]
    
    for level in [2,3,4]:
        portfolio_id = db.get_portfolio_id(portfolio=f"MLP_{market}_{level}_IRP")
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

        msg = f"\n[PASS] MLP IRP MP"
        msg += f"\n{TODAY.date()} | MLP IRP level {level}"
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
            csv_file_path = os.path.join(alloc_path, f"{TODAY:%Y%m%d}_MLP_{market}_{level}_IRP_allocation.csv")
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
                    db_alloc = db.get_alloc_weight_for_shares(strategy="MLP", market="KR", level=f"{level}_IRP")
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

    logger.info(msg=f"[PASS] End MLP allocation. {TODAY:%Y-%m-%d}", extra=extra)