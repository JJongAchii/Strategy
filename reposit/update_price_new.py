from hive import db

import numpy as np
import sqlalchemy as sa
import yfinance as yf
import pandas_datareader as pdr
from xbbg import blp

with db.session_local() as session:
    for meta in session.query(db.TbMeta).filter().all():
        latest_dt = (
            session.query(sa.func.max(db.TbDailyBar.trd_dt))
            .filter(db.TbDailyBar.stk_id == meta.stk_id)
            .scalar()
        )

        ticker = meta.ticker
        if meta.source == "naver":
            data = pdr.DataReader(ticker, "naver", "1980-1-1").astype(float)
            data["close_prc"] = data["Close"].replace(0, np.nan).ffill()
            data["gross_rtn"] = data["close_prc"].pct_change().fillna(0)
            data = data[["close_prc", "gross_rtn"]]

        elif meta.source == "yahoo":
            data = yf.download(ticker, "1980-1-1").astype(float)
            data["close_prc"] = data["Close"].replace(0, np.nan).ffill()
            data["gross_rtn"] = data["Adj Close"].pct_change().fillna(0)
            data = data[["close_prc", "gross_rtn"]]

        elif meta.source == "bloomberg":
            continue
            data = blp.bdh(meta.ticker + " Index", "PX_LAST", "1980-1-1")
            data.columns = ["Close"]
            data["close_prc"] = data["Close"].replace(0, np.nan).ffill()
            data["gross_rtn"] = data["Close"].pct_change().fillna(0)
            data = data[["close_prc", "gross_rtn"]]

        data.index.name = "trd_dt"
        data = data.reset_index()
        data["stk_id"] = meta.stk_id
        data["adj_value"] = data["gross_rtn"].add(1).cumprod()
        data["adj_value"] = data["adj_value"] * data["close_prc"].iloc[0]

        session.query(db.TbDailyBar).filter(db.TbDailyBar.stk_id == meta.stk_id).delete()
        session.commit()

        # if latest_dt:
        #     update_data = data[data["trd_dt"].dt.date <= latest_dt]
        #     db.TbDailyBar.update(update_data)
        #     insert_data = data[data["trd_dt"].dt.date > latest_dt]
        #     db.TbDailyBar.insert(insert_data, session=session)
        # else:
        db.TbDailyBar.insert(data, session=session)

        # data = data[data['adj_value2'] != data['adj_value']]
        # data['adj_value'] = data['adj_value2']
        session.commit()