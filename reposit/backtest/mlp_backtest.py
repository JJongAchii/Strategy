import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import sqlalchemy as sa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from hive import db
from config import ML_MODELS_FOLDER
from core.strategy import utils

logger = logging.getLogger("sqlite")


def get_ml_model_training_date(asofdate: datetime) -> datetime:
    """get the supposed date for model training based on the given the asofdate"""
    half = datetime(asofdate.year, 6, 30)
    if min(half, asofdate) == half:
        return half
    return datetime(asofdate.year - 1, 12, 31)


def run_prediction():

    for date in pd.date_range("2018-1-1", "2023-5-11"):

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
        from core.analytics.prediction import BaggedMlpClassifier
        model_path = os.path.join(
            ML_MODELS_FOLDER, utils.get_ml_model_training_date(asofdate=date).strftime("%Y-%m-%d")
        )
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        risk_date = YESTERDAY
        if YESTERDAY <= datetime(2022, 1, 1):
            risk_date = datetime(2022, 1, 1)

        for market in ["US"]:
            universe = db.load_universe(
                asofdate=risk_date.strftime("%Y-%m-%d"), strategy=f"mlp_{market}"
            )
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

    for date in pd.date_range("2020-3-1", "2023-5-11"):

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

        risk_date = YESTERDAY
        if YESTERDAY <= datetime(2022, 1, 1):
            risk_date = datetime(2022, 1, 1)

        for market in ["US"]:
            universe = db.load_universe(asofdate=risk_date.strftime("%Y-%m-%d"), strategy=f"MLP_{market}")
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
            allocations.to_csv("mlp_allocation_230821_oldversion.csv", index=False, encoding="utf-8-sig")


def update_db_backtest_book():
    csv_file_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "result"), f"mlp_{market}_{level}_allocation.csv")
    data = pd.read_csv(csv_file_path, index_col=False)
    db.TbBacktestAlloc.insert(data)


def get_portfolio_min_date(portfolio: str):
    with db.session_local() as session:
        return (
            session.query(
                sa.func.min(db.TbBacktestValue.trd_dt).label("min_date")
            )
            .join(db.TbPort,
                  sa.and_(db.TbPort.port_id == db.TbBacktestValue.port_id,
                          db.TbPort.portfolio == portfolio.upper())
                  )
        ).scalar()


def get_portfolio_first_date(portfolio: str):
    with db.session_local() as session:
        subquery = (
            session.query(
                sa.func.min(db.TbBacktestAlloc.rebal_dt).label("min_date"),
                db.TbBacktestAlloc.port_id
            )
            .join(db.TbPort,
                  sa.and_(db.TbPort.port_id == db.TbBacktestAlloc.port_id,
                          db.TbPort.portfolio == portfolio.upper())
                  )
            .group_by(db.TbBacktestAlloc.port_id)
        ).subquery()

        query = (
            session.query(
                subquery.c.min_date,
                subquery.c.port_id,
                db.TbBacktestAlloc.stk_id,
                db.TbBacktestAlloc.weights,
            )
            .join(subquery,
                  sa.and_(subquery.c.port_id == db.TbBacktestAlloc.port_id,
                          subquery.c.min_date == db.TbBacktestAlloc.rebal_dt)
                  )
        )
        return db.read_sql_query(query)


def get_portfolio_max_date(portfolio: str):
    """get the existing gross return data that has not been updated in the portfolio book"""
    with db.session_local() as session:
        return (
            session.query(
                sa.func.max(db.TbBacktestValue.trd_dt).label("max_date")
            )
            .join(db.TbPort,
                  sa.and_(db.TbPort.port_id == db.TbBacktestValue.port_id,
                          db.TbPort.portfolio == portfolio.upper())
                  )
        ).scalar()


def get_portfolio_allocation_list(portfolio):
    with db.session_local() as session:
        query = (
            session.query(db.TbBacktestAlloc.rebal_dt.label("trd_dt"), db.TbBacktestAlloc.port_id, db.TbBacktestAlloc.stk_id, db.TbBacktestAlloc.weights)
            .join(db.TbPort,
                  sa.and_(db.TbPort.port_id == db.TbBacktestAlloc.port_id,
                          db.TbPort.portfolio == portfolio.upper())
                  )
            .distinct()
        )
        return db.read_sql_query(query).sort_values("trd_dt")


def get_portfolio_book_at_max_date(portfolio: str, max_date: date):
    """
    Use the subquery in the main query to get the rows
    corresponding to the maximum date and the specified portfolio_id
    """
    with db.session_local() as session:
        query = (
            session.query(
                db.TbBacktestBook.port_id,
                db.TbBacktestBook.stk_id,
                db.TbBacktestBook.weights
            )
            .join(
                db.TbPort,
                sa.and_(
                    db.TbPort.port_id == db.TbBacktestBook.port_id,
                    db.TbPort.portfolio == portfolio.upper()
                )
            )
            .filter(db.TbBacktestBook.trd_dt == max_date)
        )
        return db.read_sql_query(query)


def get_portfolio_value_at_max_date(portfolio: str, max_date: date):
    with db.session_local() as session:
        query = (
            session.query(
                db.TbBacktestValue.port_id,
                db.TbBacktestValue.value
            )
            .join(
                db.TbPort,
                sa.and_(
                    db.TbPort.port_id == db.TbBacktestValue.port_id,
                    db.TbPort.portfolio == portfolio.upper()
                )
            )
            .filter(db.TbBacktestValue.trd_dt == max_date)
        )
        return db.read_sql_query(query)


def mdd(x):
    return (x / x.expanding().max() - 1).min()


def sharpe(x):
    re = x.pct_change().dropna().fillna(0)

    return re.mean() / re.std() * (252 ** 0.5)


def calculate_nav():

    port_list = db.get_portfolio_list()

    for portfolio in port_list.portfolio:
        port_id = int(port_list.loc[port_list['portfolio'] == portfolio, 'port_id'].values[0])
        min_date = get_portfolio_min_date(portfolio=portfolio)

        if min_date is None:
            first_date_df = get_portfolio_first_date(portfolio=portfolio).rename(columns={'min_date': 'trd_dt'})
            first_date_df['value'] = 1000
            if first_date_df.empty:
                continue
            db.TbBacktestValue.insert(first_date_df.iloc[0])
            db.TbBacktestBook.insert(first_date_df)

        max_date = get_portfolio_max_date(portfolio=portfolio)
        trade_date_df = db.trading_date_until_max_date(max_date=max_date)
        reb_date_df = get_portfolio_allocation_list(portfolio=portfolio)

        for trade_date in trade_date_df["trd_dt"].unique():
            new_max_date = get_portfolio_max_date(portfolio=portfolio)
            port_book_df = get_portfolio_book_at_max_date(portfolio=portfolio, max_date=new_max_date)
            port_val_df = get_portfolio_value_at_max_date(portfolio=portfolio, max_date=new_max_date)
            return_df = db.get_gross_return_at_trading_date(trade_date=trade_date, stk_list=port_book_df.stk_id)

            if not return_df.stk_id.equals(port_book_df.stk_id):
                continue

            update_df = port_book_df.merge(return_df, on="stk_id")
            weight_sum = sum(update_df["weights"])
            update_df["weights"] = update_df["weights"] * (1 + update_df["gross_rtn"])

            if weight_sum != 0:
                weight_sum = sum(update_df["weights"]) / weight_sum
            else:
                continue

            port_val_df["value"] = port_val_df["value"] * weight_sum
            port_val_df["trd_dt"] = trade_date

            update_df["weights"] = update_df["weights"] / sum(update_df["weights"])

            if trade_date in reb_date_df["trd_dt"].values:
                update_df = reb_date_df[reb_date_df["trd_dt"] == trade_date]

            db.TbBacktestBook.insert(update_df)
            db.TbBacktestValue.insert(port_val_df)

        port_val = db.TbBacktestValue.query_df(port_id=port_id).sort_values("trd_dt")
        port_val["mdd_1y"] = port_val.value.rolling(252).apply(mdd)
        port_val["sharp_1y"] = port_val.value.rolling(252).apply(sharpe)
        port_val["mdd"] = port_val.value.expanding(252).apply(mdd)
        port_val["sharp"] = port_val.value.expanding(252).apply(sharpe)

        port_val = port_val.replace(np.nan, None)
        db.TbBacktestValue.update(port_val)


if __name__ == "__main__":
    #run_prediction()
    run_allocation()
    #update_db_backtest_book()
    #calculate_nav()