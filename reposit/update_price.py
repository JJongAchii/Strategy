from hive import db

import numpy as np
import yfinance as yf
import pandas_datareader as pdr
from xbbg import blp

with db.session_local() as session:
    for meta in session.query(db.TbMeta).filter(db.TbMeta.source == "bloomberg").all():

        try:

            # ticker = meta.ticker
            # print(ticker)
            # if meta.source == "naver":
            #     continue
            #     data = pdr.DataReader(ticker, "naver", "1980-1-1").astype(float)
            #     data["close_prc"] = data["Close"].replace(0, np.nan).ffill()
            #     data['gross_rtn'] = data["close_prc"].pct_change().fillna(0)
            #     data = data[['close_prc', 'gross_rtn']]
            #
            # elif meta.source == "yahoo":
            #     continue
            #     data = yf.download(ticker, "1980-1-1").astype(float)
            #     data["close_prc"] = data["Close"].replace(0, np.nan).ffill()
            #     data['gross_rtn'] = data["Adj Close"].pct_change().fillna(0)
            #     data = data[['close_prc', 'gross_rtn']]

            if meta.source == "bloomberg":
                data = blp.bdh(meta.ticker + " Index", "PX_LAST", "1980-1-1")
                data.columns = ["Close"]
                data["close_prc"] = data["Close"].replace(0, np.nan).ffill()
                data['gross_rtn'] = data["Close"].pct_change().fillna(0)
                data = data[['close_prc', 'gross_rtn']]

            data.index.name = "trd_dt"
            data = data.reset_index()
            data["stk_id"] = meta.stk_id
            db.TbDailyBar.insert(data)

        except:
            print(meta.ticker, "failed.")

