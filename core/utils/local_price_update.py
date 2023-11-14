import os
import sys
import logging
import pytz
import sqlalchemy as sa
import numpy as np
import pandas as pd
from datetime import datetime,  timedelta
from dateutil import parser


sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from hive import db

logger = logging.getLogger("sqlite")


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


def run_update_price(asofdate: datetime) -> None:
    """update daily asset price from meta tickers"""
    YESTERDAY = asofdate - timedelta(days=1)
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
    

def save_yesterday_timeseries_to_excel(asofdate: datetime, path="output/db/tb_timeseries.csv"):
    YESTERDAY = asofdate - timedelta(days=1)
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