import os
import sys
import time
import pytz
import logging
import argparse
from datetime import datetime, date, timedelta
from dateutil import parser
from calendar import monthrange
import sqlalchemy as sa
import numpy as np
import pandas as pd
import cvxpy as cp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))
from hive import db
from core.analytics import va
from config import ML_MODELS_FOLDER, ALLOC_FOLDER
logger = logging.getLogger("sqlite")

####################################################################################################
# parse arguments
parse = argparse.ArgumentParser(description="Run MlpStrategy Script.")
parse.add_argument("-s", "--script")
parse.add_argument("-d", "--date", default=date.today().strftime("%Y-%m-%d"))
parse.add_argument("-u", "--user")
parse.add_argument("-r", "--regime", default = 'lei')
args = parse.parse_args()

####################################################################################################
# global variables
TODAY = parser.parse(args.date)
YESTERDAY = TODAY - timedelta(days=1)
TENWEEKSAGO = TODAY - timedelta(weeks=10)
YEAR = TODAY.year
MONTH = TODAY.month
DB_UPDATE = False

logger.info(f"running script {TODAY:%Y-%m-%d}")


def is_start_of_half_year() -> bool:
    """check if today is the start of half year"""
    if MONTH not in [1, 7]:
        return False
    return TODAY.day == db.get_start_trading_date(market="KR", asofdate=TODAY).day


def get_ml_model_training_date(asofdate: datetime) -> datetime:
    """get the supposed date for model training based on the given the asofdate"""
    half = datetime(asofdate.year, 6, 30)
    if min(half, asofdate) == half:
        return half
    return datetime(asofdate.year - 1, 12, 31)


def run_num_allocation():
    """check number of allocation holdings"""
    extra = dict(user=args.user, activity="num_allocation_check", category="monitoring")
    with db.session_local() as session:
        query = (
            session.query(
                db.TbPort.portfolio,
                db.TbPortAlloc.rebal_dt,
                sa.func.count(db.TbPortAlloc.stk_id).label("num_asset"),
            )
            .select_from(db.TbPortAlloc)
            .join(db.TbPort)
            .filter(db.TbPortAlloc.rebal_dt >= TENWEEKSAGO)
            .group_by(db.TbPortAlloc.rebal_dt, db.TbPort.portfolio)
            .order_by(db.TbPortAlloc.rebal_dt)
        )

        data = db.read_sql_query(query)

        checks = data[data.num_asset != 10]

        if checks.empty:
            logger.info("PASS: Num Asset == 10.", extra=extra)
            return

        for _, check in checks.iterrows():
            port = check.get("portfolio")
            asofdate = check.get("rebal_dt")
            num_asset = check.get("num_asset")
            logger.warning(msg=f"FAIL: {port} - {asofdate} - {num_asset}", extra=extra)


def run_sum_allocation():
    """check number of allocation holdings"""
    extra = dict(user=args.user, activity="sum_allocation_check", category="monitoring")
    with db.session_local() as session:
        query = (
            session.query(
                db.TbPort.portfolio,
                db.TbPortAlloc.rebal_dt,
                sa.func.sum(db.TbPortAlloc.weights).label("sum_weight"),
            )
            .select_from(db.TbPortAlloc)
            .join(db.TbPort)
            .filter(db.TbPortAlloc.rebal_dt >= TENWEEKSAGO)
            .group_by(db.TbPortAlloc.rebal_dt, db.TbPort.portfolio)
            .order_by(db.TbPortAlloc.rebal_dt)
        )

        data = db.read_sql_query(query)

        checks = data[data.sum_weight != 1.0]

        if checks.empty:
            logger.info("PASS: Sum Weight == 100%", extra=extra)
            return

        for _, check in checks.iterrows():
            port = check.get("portfolio")
            asofdate = check.get("rebal_dt")
            sum_weight = check.get("sum_weight")
            logger.warning(msg=f"{port} - {asofdate} - {sum_weight}", extra=extra)


def run_mlp_prediction() -> None:
    """
    run mlp prediction at the half year start
    i.e. Jan 1st & Jul 1st of each year.
    """
    extra = dict(user=args.user, activity="mlp_prediction", category="script")
    if not is_start_of_half_year():
        logger.info(msg=f"PASS: skip mlp prediction. {TODAY:%Y-%m-%d}", extra=extra)
        return

    from core.strategy.mlpstrategy import PredSettings
    from core.analytics.prediction import BaggedMlpClassifier

    logger.info(msg=f"PASS: start mlp prediction. {TODAY:%Y-%m-%d}", extra=extra)
    model_path = os.path.join(
        ML_MODELS_FOLDER, get_ml_model_training_date(TODAY).strftime("%Y-%m-%d")
    )
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for market in ["US", "KR"]:
        universe = db.load_universe(
            asofdate=YESTERDAY.strftime("%Y-%m-%d"), strategy=f"mlp_{market}"
        )
        price_df = db.get_price(tickers=", ".join(universe.index.tolist())).loc[:YESTERDAY]
        pred_settings = PredSettings(
            train=True, save=True, model_path=model_path
        ).dict()
        prediction_model = BaggedMlpClassifier(**pred_settings)
        prediction_model.predict(price_df=price_df)
        logger.info(
            msg=f"PASS: mlp prediction for {market}. {TODAY:%Y-%m-%d}", extra=extra
        )


def run_mlp_allocation() -> None:
    """
    run mlp allocation at the month start trading date
    i.e. first trading day each month.
    """
    extra = dict(user=args.user, activity="mlp_allocation", category="script")

    if TODAY.date() != db.get_start_trading_date(market="KR", asofdate=TODAY):
        logger.info(msg=f"PASS: skip mlp allocation. {TODAY:%Y-%m-%d}", extra=extra)
        return

    logger.info(msg=f"PASS: start mlp allocation. {TODAY:%Y-%m-%d}", extra=extra)

    from core.strategy.mlpstrategy import MlpStrategy

    ticker_mapper = db.get_meta_mapper()
    OUTPUT_COLS = ["isin", "ticker_bloomberg", "asset_class", "risk_score", "name"]
    allocation_weights = []

    model_path = os.path.join(
        ML_MODELS_FOLDER, get_ml_model_training_date(TODAY).strftime("%Y-%m-%d")
    )

    for market in ["US", "KR"]:
        universe = db.load_universe(f"MLP_{market}")
        prices = db.get_price(tickers=", ".join(universe.index.tolist())).loc[:YESTERDAY]

        for level in [3, 4, 5]:
            portfolio_id = db.get_portfolio_id(portfolio=f"MLP_{market}_{level}")
            strategy = MlpStrategy.load(
                universe=universe, prices=prices, level=level, model_path=model_path
            )
            weights = strategy.allocate()
            risk_score = 0.0
            for asset, weight in weights.items():
                risk_score += weight * strategy.universe.loc[str(asset)].risk_score
            max_date_price = strategy.price_asset.index[-1]
            msg = f"\nallocate weights for market={market}, level={level}, "
            msg += f"with latest available price at date {max_date_price:%Y-%m-%d}."
            msg += f"\nallocation weights is as below, with risk score = {risk_score:.4f}:\n"
            msg += weights.to_markdown()
            logger.info(msg, extra=extra)

            if DB_UPDATE is True:
                uni = strategy.universe[OUTPUT_COLS]
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
                    db.TbPortAlloc.update(allo)
                    db.delete_portfolio_from_date(portfolio=portfolio_id, asofdate=TODAY)

                """update db for increase probability of assets"""
                univ = strategy.universe[["isin", "strg_asset_class", "iso_code", "wrap_asset_class_code"]]
                prob = strategy.prediction
                
                univ["prob"] = univ.index.map(prob.to_dict())
                univ = univ.groupby('strg_asset_class').apply(lambda x: x.nlargest(5, 'prob'))
                univ["rebal_dt"] = TODAY
                univ = univ.dropna()
                
                try:
                    db.TbProbIncrease.insert(univ)
                except:
                    db.TbProbIncrease.update(univ)


def update_shares():
    for strategy in ["MLP", "ABL"]:
        for market in ["US", "KR"]:
            for level in [3, 4, 5]:
                    account = va.VirtualAccount(strategy=strategy, market=market, level=level)
                    account.calculate_virtual_account_nav(weight=account.weights)
                    account.update_shares_db()

    kb_dam = va.VirtualAccount(strategy="KB", market="DAM", level=5, portfolio_value=3_000_000)
    kb_dam.calculate_virtual_account_nav(weight=kb_dam.weights, currency="KRW")
    kb_dam.update_shares_db()
    
    logger.info("PASS: shares update complete.")
    

def is_during_trading_session(zone="Asia/Seoul", start="9:00", end="15:30") -> bool:
    """_summary_

    Args:
        zone (str, optional): _description_. Defaults to "Asia/Seoul".

    Returns:
        bool: True if in trading session.
    """
    tz = pytz.timezone(zone=zone)
    now = datetime.now(tz=tz)
    start = parser.parse(now.strftime("%Y-%m-%d") + " " + start).astimezone(tz)
    end = parser.parse(now.strftime("%Y-%m-%d") + " " + end).astimezone(tz)
    return start <= now <= end


def is_trading_session_kr() -> bool:
    """this is passthrough function"""
    return is_during_trading_session(zone="Asia/Seoul", start="9:00", end="15:30")


def is_trading_session_us() -> bool:
    """this is passthrough function"""
    return is_during_trading_session(zone="America/New_York", start="9:30", end="16:00")


def run_update_price():
    try:
        import yfinance as yf
        import pandas_datareader as pdr
    except ImportError:
        return
    with db.session_local() as session:
        for meta in db.TbMeta.query().all():
            gross_return_date = (
                session.query(sa.func.max(db.TbDailyBar.trd_dt))
                .filter(
                    db.TbDailyBar.gross_rtn.isnot(None),
                    db.TbDailyBar.stk_id == meta.stk_id,
                )
                .scalar()
            )
            close_date = (
                session.query(sa.func.max(db.TbDailyBar.trd_dt))
                .filter(
                    db.TbDailyBar.close_prc.isnot(None),
                    db.TbDailyBar.stk_id == meta.stk_id,
                )
                .scalar()
            )

            if close_date:
                close_date = datetime.combine(close_date, datetime.min.time())

            if gross_return_date:
                gross_return_date = datetime.combine(
                    gross_return_date, datetime.min.time()
                )

            if not close_date is None and not gross_return_date is None:
                if close_date >= YESTERDAY and gross_return_date >= YESTERDAY:
                    continue

            query = (
                session.query(
                    db.TbTicker.ticker_yahoo,
                    db.TbTicker.ticker_naver
                )
                .filter(db.TbTicker.stk_id == meta.stk_id)
            )
            ticker = db.read_sql_query(query)

            if meta.source == "naver":
                data = pdr.DataReader(ticker.ticker_naver.values[0], "naver", "1900-1-1").astype(
                    float
                )
                data["Close"] = data["Close"].replace(0, np.nan).ffill()
                data["GROSS_RETURN"] = data["Close"].pct_change().fillna(0)

                if is_trading_session_kr():
                    data = data.loc[data.index < YESTERDAY]

                if close_date is not None:
                    close = data.loc[data.index > close_date][["Close"]].reset_index()
                else:
                    close = data.loc[:, ["Close"]][["Close"]].reset_index()
                if not close.empty:
                    close.columns = ["trd_dt", "close_prc"]
                    close["stk_id"] = meta.stk_id

                if gross_return_date is not None:
                    gross_return = data.loc[data.index > close_date][
                        ["GROSS_RETURN"]
                    ].reset_index()
                else:
                    gross_return = data.loc[:, ["GROSS_RETURN"]][
                        ["GROSS_RETURN"]
                    ].reset_index()
                gross_return = gross_return.dropna()
                if not gross_return.empty:
                    gross_return.columns = ["trd_dt", "gross_rtn"]
                    gross_return["stk_id"] = meta.stk_id

                records = pd.merge(close, gross_return, on=['trd_dt', 'stk_id'])

            elif meta.source == "yahoo":
                data = yf.download(ticker.ticker_yahoo.values[0], "1990-1-1")
                data["Adj Close"] = data["Adj Close"].replace(0, np.nan).ffill()
                data["GROSS_RETURN"] = data["Adj Close"].pct_change().fillna(0)
                if is_trading_session_us():
                    data = data.loc[data.index < YESTERDAY]

                if close_date is not None:
                    close = data.loc[data.index > close_date][["Close"]].reset_index()
                else:
                    close = data.loc[:, ["Adj Close"]][["Adj Close"]].reset_index()
                if not close.empty:
                    close.columns = ["trd_dt", "close_prc"]
                    close["stk_id"] = meta.stk_id
                    close = close.replace(0, np.nan).fillna(method="ffill").fillna(0)

                if gross_return_date is not None:
                    gross_return = data.loc[data.index > close_date][
                        ["GROSS_RETURN"]
                    ].reset_index()
                else:
                    gross_return = data.loc[:, ["GROSS_RETURN"]][
                        ["GROSS_RETURN"]
                    ].reset_index()
                gross_return = gross_return.dropna()
                if not gross_return.empty:
                    gross_return.columns = ["trd_dt", "gross_rtn"]
                    gross_return["stk_id"] = meta.stk_id
                if close.empty:
                    continue
                if gross_return.empty:
                    continue
                records = pd.merge(close, gross_return, on=['trd_dt', 'stk_id'])

            else:
                records = None

            if records is not None:
                if not records.empty:
                    db.TbDailyBar.insert(records)
    logger.info("PASS: price update complete.")


def save_yesterday_timeseries_to_excel(path="output/db/tb_timeseries.csv"):
    # Check if the path exists
    if not os.path.exists(os.path.dirname(path)):
        # If the path doesn't exist, create the directory and any missing parent directories
        os.makedirs(os.path.dirname(path))

    db.TbDailyBar.query_df(trd_dt=YESTERDAY)[
        ["trd_dt", "stk_id", "close_prc", "gross_rtn", "adj_value"]
    ].to_csv(path, index=False)


def update_price_data_file(path="output/db/tb_timeseries.csv"):
    # Check if the path exists
    if not os.path.exists(os.path.dirname(path)):
        # If the path doesn't exist, create the directory and any missing parent directories
        os.makedirs(os.path.dirname(path))
    records = pd.read_csv(path, parse_dates=True)
    try:
        db.TbDailyBar.insert(records=records)
    except:
        db.TbDailyBar.update(records=records)


def mdd(x):
    return (x / x.expanding().max() - 1).min()


def sharpe(x):
    re = x.pct_change().dropna().fillna(0)

    return re.mean() / re.std() * (252 ** 0.5)


def run_update_db_portfolio() -> None:
    port_list = db.get_portfolio_list()

    for portfolio in port_list.portfolio:
        port_id = int(port_list.loc[port_list['portfolio'] == portfolio, 'port_id'].values[0])
        min_date = db.get_portfolio_min_date(portfolio=portfolio)

        if min_date is None:
            first_date_df = db.get_portfolio_first_date(portfolio=portfolio).rename(columns={'min_date': 'trd_dt'})
            first_date_df['value'] = 1000
            if first_date_df.empty:
                continue
            db.TbPortValue.insert(first_date_df.iloc[0])
            db.TbPortBook.insert(first_date_df)

        max_date = db.get_portfolio_max_date(portfolio=portfolio)
        trade_date_df = db.trading_date_until_max_date(max_date=max_date)
        reb_date_df = db.get_portfolio_allocation_list(portfolio=portfolio)

        for trade_date in trade_date_df["trd_dt"].unique():
            new_max_date = db.get_portfolio_max_date(portfolio=portfolio)
            port_book_df = db.get_portfolio_book_at_max_date(portfolio=portfolio, max_date=new_max_date)
            port_val_df = db.get_portfolio_value_at_max_date(portfolio=portfolio, max_date=new_max_date)
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

            db.TbPortBook.insert(update_df)
            db.TbPortValue.insert(port_val_df)

        port_val = db.TbPortValue.query_df(port_id=port_id).sort_values("trd_dt")
        port_val["mdd_1y"] = port_val.value.rolling(252).apply(mdd)
        port_val["sharp_1y"] = port_val.value.rolling(252).apply(sharpe)
        port_val["mdd"] = port_val.value.expanding(252).apply(mdd)
        port_val["sharp"] = port_val.value.expanding(252).apply(sharpe)

        port_val = port_val.replace(np.nan, None)
        db.TbPortValue.update(port_val)


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


def run_abl_allocation(regime: str = "lei") -> None:
    """
    run mlp allocation at the month start trading date
    i.e. first trading day each month.
    """
    extra = dict(user=args.user, activity="abl_allocation", category="script")

    if TODAY.date() != db.get_start_trading_date(market="KR", asofdate=TODAY):
        logger.info(msg=f"PASS: skip abl allocation. {TODAY:%Y-%m-%d}", extra=extra)
        return

    logger.info(msg=f"PASS: start abl allocation. {TODAY:%Y-%m-%d}", extra=extra)

    from core.strategy.ablstrategy import AblStrategy
    
    ticker_mapper = db.get_meta_mapper()
    OUTPUT_COLS = ["isin", "ticker_bloomberg", "asset_class", "risk_score", "name"]
    allocation_weights = []

    universe = db.load_universe("abl_us")
    price_asset = db.get_price(tickers=", ".join(list(universe.index))).loc[:YESTERDAY]
    universe_kr = db.load_universe("abl_kr")
    price_asset_kr = db.get_price(tickers=", ".join(list(universe_kr.index))).loc[:YESTERDAY]
    price_factor = db.get_lens()

    for level in [3, 4, 5]:
        strategy = AblStrategy.load(
            universe=universe,
            price_asset=price_asset,
            price_factor=price_factor,
            regime=regime,
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

        max_date_price = strategy.price_asset.index[-1]
        msg = f"\nallocate weights for market=US, level={level}, "
        msg += f"with latest available price at date {max_date_price:%Y-%m-%d}."
        msg += f"\nallocation weights is as below, with risk score = {us_risk_score:.4f}:\n"
        msg += us_weights.to_markdown()
        logger.info(msg, extra=extra)

        kr_risk_score = 0.0
        for asset, weight in kr_weights.items():
            kr_risk_score += weight * universe_kr.loc[str(asset)].risk_score

        msg = f"\nallocate weights for market=KR, level={level}, "
        msg += f"with latest available price at date {max_date_price:%Y-%m-%d}."
        msg += f"\nallocation weights is as below, with risk score = {kr_risk_score:.4f}:\n"
        msg += kr_weights.to_markdown()
        logger.info(msg, extra=extra)

        if DB_UPDATE is True:
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
                db.TbPortAlloc.update(us_weights)
                db.delete_portfolio_from_date(portfolio=portfolio_id, asofdate=TODAY)

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
                db.TbPortAlloc.update(kr_weights)
                db.delete_portfolio_from_date(portfolio=portfolio_id, asofdate=TODAY)


def kr_price_monitoring():
    """
    KR only
    """
    extra = dict(user=args.user, activity="KR_daily_price_uploaded", category="monitoring")
    logger.info(msg=f"PASS: start KR_price monitoring. {TODAY:%Y-%m-%d}", extra=extra)

    TODAY_8 = TODAY.date().strftime("%Y%m%d")

    new_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../hive/eai/receive"))
    os.chdir(new_dir)
    sys.path.insert(0, new_dir)

    kr_daily_data = pd.read_csv(f"stk_close_{TODAY_8}.txt", sep="|", names=["trd_dt","TICKER","close_prc","gross_rtn","close_prc_1","gross_rtn_1","volume","shr_volume","aum","volume_1","shr_volume_1","aum_1"])
    kr_daily_data["TICKER"] = kr_daily_data["TICKER"].apply(lambda x: str(x).zfill(6))

    kr_adj_value_df_1d = db.query.get_last_trading_date_price(TODAY, "KR")

    with db.session_local() as session:
        for meta in db.TbMeta.query().all():
            ticker = meta.ticker

            if meta.iso_code == "KR":
                kr_ticker = ticker
                today_close = kr_daily_data[kr_daily_data["TICKER"]==kr_ticker].close_prc

                if today_close.empty:
                    logger.warning(msg=f"Warning: KR-{kr_ticker}-No price", extra=extra)
                else:
                    today_close = float(np.array2string(today_close.values, separator=',')[1:-1])
                    yesterday_close = kr_adj_value_df_1d[kr_adj_value_df_1d["stk_id"]==meta.stk_id].close_prc.values
                    yesterday_close = float(np.array2string(yesterday_close, separator=',')[1:-1])
                    rtn = (today_close-yesterday_close)/yesterday_close
                    if abs(rtn) > 0.05:
                        logger.warning(msg=f"Warning: KR-{kr_ticker}-Excessive absolute return", extra=extra)
                    elif abs(rtn) == 0:
                        logger.warning(msg=f"Warning: KR-{kr_ticker}-Price's the same as yesterday's", extra=extra)
                    else:
                        continue


def us_price_monitoring():
    """
    US only
    """
    extra = dict(user=args.user, activity="US_daily_price_uploaded", category="monitoring")
    logger.info(msg=f"PASS: start US_price monitoring. {TODAY:%Y-%m-%d}", extra=extra)

    YESTERDAY_= YESTERDAY.strftime("%Y%m%d")

    new_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../hive/eai/receive"))
    os.chdir(new_dir)
    sys.path.insert(0, new_dir)

    us_daily_data = pd.read_csv(f"gst_close_{YESTERDAY_}.txt", sep="|", names=["trd_dt","EXCHANGE","TICKER","close_prc","gross_rtn","close_prc_1","gross_rtn_1","risk_level","volume","shr_volume","aum","volume_1","shr_volume_1"])
    us_daily_data["TICKER"] = us_daily_data["TICKER"].apply(lambda x: str(x).replace(" ", ""))
    us_daily_data["risk_level"] = us_daily_data["risk_level"].apply(lambda x: str(x).replace(" ", ""))

    us_adj_value_df_1d = db.query.get_last_trading_date_price(YESTERDAY, "US")

    with db.session_local() as session:
        for meta in db.TbMeta.query().all():
            ticker = meta.ticker

            if meta.iso_code == "US":
                if meta.source == "bloomberg":
                    pass
                else:
                    us_ticker = ticker
                    today_close = us_daily_data[us_daily_data["TICKER"]==us_ticker].close_prc
                    if today_close.empty:
                        logger.warning(msg=f"Warning: US-{us_ticker}-No price", extra=extra)
                    else:
                        today_close = float(np.array2string(today_close.values, separator=',')[1:-1])
                        yesterday_close = us_adj_value_df_1d[us_adj_value_df_1d["stk_id"]==meta.stk_id].close_prc.values
                        yesterday_close = float(np.array2string(yesterday_close, separator=',')[1:-1])
                        rtn = (today_close-yesterday_close)/yesterday_close
                        if abs(rtn) > 0.05:
                            logger.warning(msg=f"Warning: US-{us_ticker}-Excessive absolute return", extra=extra)
                        elif abs(rtn) == 0:
                            logger.warning(msg=f"Warning: US-{us_ticker}-Price's the same as yesterday's", extra=extra)
                        else:
                            continue


def run_upload_historical_data() -> None:
    """replace the historical data with the data from INFOMAX"""
    extra = dict(user=args.user, activity="historical_data_replacing", category="script")
    extra2 = dict(user=args.user, activity="historical_data_replacing_check", category="monitoring")
    logger.info(msg=f"PASS: start historical data replacing. {TODAY:%Y-%m-%d}", extra=extra)
    
    global meta_data

    new_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../hive/eai/receive"))
    os.chdir(new_dir)
    sys.path.insert(0, new_dir)

    us_data = pd.read_csv("usday.txt", sep="|", names=["trd_dt","EXC","TICKER","close_prc","gross_rtn","volumn","shr_volumn"])
    us_data = us_data.drop(us_data[us_data["close_prc"]==0].index)
    us_data["TICKER"] = us_data["TICKER"].apply(lambda x: str(x).replace(" ", ""))

    kr_data = pd.read_csv("stkday.txt", sep="|", names=["trd_dt","TICKER","close_prc","gross_rtn","volumn","shr_volumn","aum"])
    kr_data = kr_data.drop(kr_data[kr_data["close_prc"]==0].index)
    kr_data["TICKER"] = kr_data["TICKER"].apply(lambda x: str(x).zfill(6))

    f= open("usmaster_listdate.txt", "r", encoding='utf-8')
    us_text_list = f.read().replace(" ", "").split("\n")
    f= open("stkmaster_listdate.txt", "r", encoding='utf-8')
    kr_text_list = f.read().replace(" ", "").split("\n")

    us_inception_date_table = pd.DataFrame()
    for each_line in us_text_list:
        i_list = []
        each_unit= each_line.split("|")   
        ticker = each_unit[1]
        inception_date = str(each_unit[3])
        i_list.append(ticker)
        i_list.append(inception_date)
        us_inception_date_table = pd.concat([us_inception_date_table, pd.DataFrame(i_list).T])
    us_inception_date_table.columns = ["ticker","inception_date"]

    kr_inception_date_table = pd.DataFrame()
    for each_line in kr_text_list:
        i_list = []
        each_unit= each_line.split("|")   
        ticker = each_unit[0]
        inception_date = str(each_unit[2])
        i_list.append(ticker)
        i_list.append(inception_date)
        kr_inception_date_table = pd.concat([kr_inception_date_table, pd.DataFrame(i_list).T])
    kr_inception_date_table.columns = ["ticker","inception_date"]


    def get_adj_value2(ticker:str, id:int, data:pd.DataFrame) -> pd.DataFrame:
        meta_data = data[data["TICKER"] == ticker] 
        if meta_data.empty:
            return None
        meta_data = meta_data[["trd_dt","close_prc","gross_rtn"]].reset_index().drop("index", axis=1)
        meta_data["gross_rtn"] = meta_data["gross_rtn"].apply(lambda x: x*0.01)
        meta_stk_id = pd.DataFrame(np.repeat(id, len(meta_data)), columns=["stk_id"])
        meta_data = pd.concat([meta_stk_id, meta_data], axis=1)

        if meta.iso_code == "US":
            if meta.source == "bloomberg":
                pass
            else:
                inception_date = int(us_inception_date_table[us_inception_date_table.ticker == ticker].inception_date.values)
                if meta_data["trd_dt"].min() < inception_date:
                    meta_data = meta_data[meta_data["trd_dt"] >= inception_date]
                    logger.warning(msg=f"Warning: US-{meta.ticker} has been cut on the inception date", extra=extra2)
                else:
                    pass
        if meta.iso_code == "KR":
            inception_date = int(kr_inception_date_table[kr_inception_date_table.ticker == ticker].inception_date.values)
            if meta_data["trd_dt"].min() < inception_date:
                meta_data = meta_data[meta_data["trd_dt"] >= inception_date]
                logger.warning(msg=f"Warning: US-{meta.ticker} has been cut on the inception date", extra=extra2)
            else:
                pass

        meta_data["trd_dt"] = pd.to_datetime(meta_data["trd_dt"], format="%Y%m%d").apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S.%f"))
        meta_data.iloc[-1,3] = 0
        sorted_meta_data = meta_data.sort_values('trd_dt', ascending=True)
        first_close_prc = sorted_meta_data.reset_index().drop("index", axis=1).close_prc[0]

        meta_data["adj_value"] = 0
        meta_data["adj_value"] = sorted_meta_data["gross_rtn"].apply(lambda x: x + 1).cumprod().apply(lambda x: x*first_close_prc)
        meta_data["adj_value"] = meta_data["adj_value"].apply(lambda x: 0 if abs(x) < 0.0001 else x)

        return meta_data
    
    for meta in db.TbMeta.query().all():
        if meta.iso_code == "US":
            if meta.source == "bloomberg":
                pass
            else:
                try:
                    meta_data = get_adj_value2(ticker=meta.ticker,id=meta.stk_id,data=us_data)
                    db.TbDailyBar.insert(meta_data)
                except:
                    print("US @ Ticker:", meta.ticker, "Error Occured | This must be delisted, you better check it out")
                    logger.warning(msg=f"Warning: US-{meta.ticker}-Uploading Failed", extra=extra2)
                    continue

        if meta.iso_code == "KR":
            try:
                meta_data = get_adj_value2(ticker=meta.ticker,id=meta.stk_id,data=kr_data) 
                db.TbDailyBar.insert(meta_data) 
            except:
                print("KR @ Ticker:", meta.ticker,"Error Occured | This must be delisted, you better check it out")
                logger.warning(msg=f"Warning: KR-{meta.ticker}-Uploading Failed", extra=extra2)
                continue

    # ETFs which were detected as errors
    logger.info("PASS: Historical price update complete.")


def data_separator(meta: db.query):
    """separator the file into several part and make them DataFrames for DB loading"""
    def get_adj_value(meta: db.query, data:pd.DataFrame) -> pd.Series:
        ticker_data = data[data["stk_id"] == meta.stk_id]
        adj_value_df = ticker_data.adj_value.values
        adj_value = np.array2string(adj_value_df, separator=',')[1:-1]
        adj_value = 0 if float(adj_value) < 0.0001 else float(adj_value)
        return adj_value

    # US TODAY(just today; one line data)
    if meta.iso_code == "US":
        if meta.source == "bloomberg":
            pass
        else:
            for each_line in us_text_list:
                each_unit= each_line.split("|")   
                if meta.ticker == each_unit[2]: 
                    date = int(each_unit[0])
                    close = each_unit[3]
                    gross_return = each_unit[4]
                    close_1 = each_unit[5]
                    gross_return_1 = each_unit[6]
                
                    adj_value_today = (1+float(gross_return)*0.01) * get_adj_value(meta=meta, data=us_adj_value_df_1d)
                    adj_value_yesterday = (1+float(gross_return_1)*0.01) * get_adj_value(meta=meta, data=us_adj_value_df_2d)
                    daily_bar_today_list = [str(meta.stk_id), str(date), close, float(gross_return)*0.01, adj_value_today]
                    daily_bar_yesterday_list = [str(meta.stk_id), str(us_adj_value_df_1d.trd_dt[0]), close_1, float(gross_return_1)*0.01, adj_value_yesterday]
                
                    risk_score = each_unit[7]

                    risk_score_list = [str(meta.stk_id), str(date), risk_score]
                
                    volume = each_unit[8]
                    shr_volume = each_unit[9]
                    aum = each_unit[10]
                
                    meta_updat_list = [str(date), str(meta.stk_id), aum, volume, shr_volume]
                
                    daily_bar_df = pd.DataFrame(daily_bar_today_list).T
                    daily_bar_df.columns = ["stk_id","trd_dt","close_prc","gross_rtn", "adj_value"]
                    daily_bar_1_df = pd.DataFrame(daily_bar_yesterday_list).T
                    daily_bar_1_df.columns = ["stk_id","trd_dt","close_prc","gross_rtn", "adj_value"]
                    meta_updat_df = pd.DataFrame(meta_updat_list).T
                    meta_updat_df.columns = ["trd_dt","stk_id","aum","volume","shr_volume"]
                    risk_score_df = pd.DataFrame(risk_score_list).T
                    risk_score_df.columns = ["stk_id","trd_dt","risk_score"]
                    return daily_bar_df, daily_bar_1_df, meta_updat_df, risk_score_df
                else:
                    continue
            
    # KR TODAY(just today; one line data)
    if meta.iso_code == "KR":
        for each_line in kr_text_list:
            each_unit = each_line.split("|")   
            if meta.ticker == each_unit[1]: 
                date = int(each_unit[0])
                close = each_unit[2]
                gross_return = each_unit[3]
                close_1 = each_unit[4]
                gross_return_1 = each_unit[5]

                adj_value_today = (1+float(gross_return)*0.01) * get_adj_value(meta=meta, data=kr_adj_value_df_1d)
                adj_value_yesterday = (1+float(gross_return_1)*0.01) * get_adj_value(meta=meta, data=kr_adj_value_df_2d)
                daily_bar_today_list = [str(meta.stk_id), str(date), close, float(gross_return)*0.01, adj_value_today]
                daily_bar_yesterday_list = [str(meta.stk_id), str(kr_adj_value_df_1d.trd_dt[0]), close_1, float(gross_return_1)*0.01, adj_value_yesterday]
                
                volume = each_unit[6]
                shr_volume = each_unit[7]
                aum = each_unit[8]
                
                meta_updat_list = [str(date), str(meta.stk_id), aum, volume, shr_volume]

                daily_bar_df = pd.DataFrame(daily_bar_today_list).T
                daily_bar_df.columns = ["stk_id","trd_dt","close_prc","gross_rtn", "adj_value"]
                daily_bar_1_df = pd.DataFrame(daily_bar_yesterday_list).T
                daily_bar_1_df.columns = ["stk_id","trd_dt","close_prc","gross_rtn", "adj_value"]
                meta_updat_df = pd.DataFrame(meta_updat_list).T
                meta_updat_df.columns = ["trd_dt","stk_id","aum","volume","shr_volume"]
                return daily_bar_df, daily_bar_1_df, meta_updat_df  
            else:
                continue

def run_update_daily_KR():
    """insert the daily data to DB tables"""
    extra = dict(user=args.user, activity="Daily_data_inserting", category="script")
    extra2 = dict(user=args.user, activity="Daily_data_inserting_check", category="monitoring")
    logger.info(msg=f"PASS: start daily data inserting. {TODAY:%Y-%m-%d}", extra=extra)

    global kr_adj_value_df_1d, kr_adj_value_df_2d,kr_text_list

    TODAY_ = TODAY.date()
    TODAY_8 = TODAY_.strftime("%Y%m%d")

    kr_adj_value_df_1d, kr_adj_value_df_2d = db.query.get_last_two_trading_dates_price(TODAY, "KR")

    new_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../hive/eai/receive"))
    os.chdir(new_dir)
    sys.path.insert(0, new_dir)

    kr_current_file_path,kr_new_file_path = f'stk_close.{TODAY_8}', f'stk_close_{TODAY_8}.txt'

    try:
        os.rename(kr_current_file_path, kr_new_file_path)
    except:
        pass
    try:
        f= open(f"stk_close_{TODAY_8}.txt", "r")
        kr_text_list = f.read().split("\n")
    except Exception as e:
        logger.warning(msg=f"Warning: KR - {e}", extra=extra2)
    
    with db.session_local() as session:
        for meta in db.TbMeta.query().all():
            gross_return_date = (
                session.query(sa.func.max(db.TbDailyBar.trd_dt))
                .filter(
                    db.TbDailyBar.gross_rtn.isnot(None),
                    db.TbDailyBar.stk_id == meta.stk_id,
                )
                .scalar()
            )
            close_date = (
                session.query(sa.func.max(db.TbDailyBar.trd_dt))
                .filter(
                    db.TbDailyBar.close_prc.isnot(None),
                    db.TbDailyBar.stk_id == meta.stk_id,
                )
                .scalar()
            )
            if not close_date is None and not gross_return_date is None:
                if close_date >= TODAY_ and gross_return_date >= TODAY_:
                    continue    

            daily_bar_today_df, daily_bar_lastday_df, meta_updat_df, risk_score_df = None, None, None, None
            if meta.iso_code == "KR":
                try:
                    daily_bar_today_df,daily_bar_lastday_df,meta_updat_df = data_separator(meta=meta)
                except:
                    print("KR |",meta.ticker, "(", meta.stk_id,")", "| This must be delisted. You better check it out")
                    logger.warning(msg=f"Warning: KR - {meta.ticker} - Uploading Failed", extra=extra2)
                    continue

            if daily_bar_today_df is not None:
                if not daily_bar_today_df.empty:
                    try:
                        db.TbDailyBar.insert(daily_bar_today_df)
                    except:
                        try:
                            db.TbDailyBar.update(daily_bar_today_df)
                        except:
                            logger.warning(msg=f"Warning: {meta.iso_code}-{meta.ticker}-TbDailyBar Uploading Failed", extra=extra2)
                            pass
            if daily_bar_lastday_df is not None:
                if not daily_bar_lastday_df.empty:
                    try:
                        db.TbDailyBar.update(daily_bar_lastday_df)
                    except:
                        logger.warning(msg=f"Warning: {meta.iso_code}-{meta.ticker}-TbDailyBar Updating Failed", extra=extra2)
            if meta_updat_df is not None:
                if not meta_updat_df.empty:
                    try:
                        db.TbMetaUpdat.insert(meta_updat_df)
                    except:
                        try:
                            db.TbMetaUpdat.update(meta_updat_df)
                        except:
                            logger.warning(msg=f"Warning: {meta.iso_code}-{meta.ticker}-TbMetaUpdat Uploading Failed", extra=extra2)
                            pass
            # if risk_score_df is not None:
            #     if not risk_score_df.empty:
            #         try:
            #            db.TbRiskScore.insert(risk_score_df)
            #         except:
            #             try:
            #                 db.TbRiskScore.update(risk_score_df)
            #             except:
            #                 logger.warning(msg=f"Warning: {meta.iso_code}-{meta.ticker}-TbRiskScore Uploading Failed", extra=extra2)
            #                 pass
            else:
                continue

    logger.info("PASS: price update complete.")    

def run_update_daily_US():
    """insert the daily data to DB tables"""
    extra = dict(user=args.user, activity="Daily_data_inserting", category="script")
    extra2 = dict(user=args.user, activity="Daily_data_inserting_check", category="monitoring")
    logger.info(msg=f"PASS: start daily data inserting. {TODAY:%Y-%m-%d}", extra=extra)

    global us_adj_value_df_1d,us_adj_value_df_2d,us_text_list

    TODAY_ = TODAY.date()
    YESTERDAY_ = TODAY_ - timedelta(days=1)
    YESTERDAY_8 = YESTERDAY_.strftime("%Y%m%d")

    us_adj_value_df_1d, us_adj_value_df_2d = db.query.get_last_two_trading_dates_price(YESTERDAY, "US")

    new_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../hive/eai/receive"))
    os.chdir(new_dir)
    sys.path.insert(0, new_dir)

    us_current_file_path,us_new_file_path = f'gst_close.{YESTERDAY_8}', f'gst_close_{YESTERDAY_8}.txt'

    try:
        os.rename(us_current_file_path, us_new_file_path)
    except:
        pass
    try:
        f= open(f"gst_close_{YESTERDAY_8}.txt", "r")
        us_text_list = f.read().replace(" ", "").split("\n")
    except Exception as e:
        logger.warning(msg=f"Warning: US - {meta.ticker} - {e}", extra=extra2)

    with db.session_local() as session:
        for meta in db.TbMeta.query().all():
            gross_return_date = (
                session.query(sa.func.max(db.TbDailyBar.trd_dt))
                .filter(
                    db.TbDailyBar.gross_rtn.isnot(None),
                    db.TbDailyBar.stk_id == meta.stk_id,
                )
                .scalar()
            )
            close_date = (
                session.query(sa.func.max(db.TbDailyBar.trd_dt))
                .filter(
                    db.TbDailyBar.close_prc.isnot(None),
                    db.TbDailyBar.stk_id == meta.stk_id,
                )
                .scalar()
            )
            if not close_date is None and not gross_return_date is None:
                if close_date >= YESTERDAY_ and gross_return_date >= YESTERDAY_:
                    continue    
            
            daily_bar_today_df,daily_bar_lastday_df,meta_updat_df,risk_score_df = None,None,None,None
            if meta.iso_code == "US":
                if meta.source == "bloomberg":
                    pass
                else:
                    try:    
                        daily_bar_today_df,daily_bar_lastday_df,meta_updat_df,risk_score_df = data_separator(meta=meta)
                    except:
                        print("US |",meta.ticker,"(", meta.stk_id,")", "| This must be delisted. You better check it out")
                        logger.warning(msg=f"Warning: US - {meta.ticker} - Uploading Failed", extra=extra2)
                        continue

            if daily_bar_today_df is not None:
                if not daily_bar_today_df.empty:
                    try:
                        db.TbDailyBar.insert(daily_bar_today_df)
                    except:
                        try:
                            db.TbDailyBar.update(daily_bar_today_df)
                        except:
                            logger.warning(msg=f"Warning: {meta.iso_code}-{meta.ticker}-TbDailyBar Uploading Failed", extra=extra2)
                            pass
            if daily_bar_lastday_df is not None:
                if not daily_bar_lastday_df.empty:
                    try:
                        db.TbDailyBar.update(daily_bar_lastday_df)
                    except:
                        logger.warning(msg=f"Warning: {meta.iso_code}-{meta.ticker}-TbDailyBar Updating Failed", extra=extra2)

            if meta_updat_df is not None:
                if not meta_updat_df.empty:
                    try:
                        db.TbMetaUpdat.insert(meta_updat_df)
                    except:
                        try:
                            db.TbMetaUpdat.update(meta_updat_df)
                        except:
                            logger.warning(msg=f"Warning: {meta.iso_code}-{meta.ticker}-TbMetaUpdat Uploading Failed", extra=extra2)
                            pass
            # if risk_score_df is not None:
            #     if not risk_score_df.empty:
            #         try:
            #            db.TbRiskScore.update(risk_score_df)
            #         except:
            #             try:
            #                 db.TbRiskScore.update(meta_updat_df)
            #             except:
            #                 logger.warning(msg=f"Warning: {meta.iso_code}-{meta.ticker}-TbRiskScore Uploading Failed", extra=extra2)
            #                 pass
            else:
                continue

    logger.info("PASS: price update complete.")    

if args.script:
    if args.script == "mlppredict":
        run_mlp_prediction()
    elif args.script == "mlp":
        run_mlp_allocation()
    elif args.script == "abl":
        if args.regime == "IML":
            run_abl_allocation(regime = "IML")
        elif args.regime == "GMM":
            run_abl_allocation(regime = "GMM")
        else:
            run_abl_allocation()
    elif args.script == "marketdb":
        #update_price_data_file()
        run_update_db_portfolio()
    elif args.script == "mlpupdate":
        DB_UPDATE = True
        run_mlp_allocation()
    elif args.script == "ablupdate":
        DB_UPDATE = True
        run_abl_allocation()
    elif args.script == "local":
        run_update_price()
        save_yesterday_timeseries_to_excel()
    elif args.script == "usdailydata":
        run_update_daily_US()
    elif args.script == "krdailydata":
        run_update_daily_KR()
    elif args.script == "historicaldata":
        run_upload_historical_data()
    elif args.script == "krmonitoring":
        kr_price_monitoring()
    elif args.script == "usmonitoring":
        us_price_monitoring()

else:
    DB_UPDATE = True
    run_mlp_prediction()
    time.sleep(1)
    run_mlp_allocation()
    time.sleep(1)
    run_abl_allocation()
    time.sleep(1)
    run_num_allocation()
    time.sleep(1)
    run_sum_allocation()
    time.sleep(1)
    update_shares()
    time.sleep(1)


