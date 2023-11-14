from functools import lru_cache
from typing import Optional, Union, List
from dateutil import parser
import pandas as pd
import sqlalchemy as sa
from hive import db
from hive.db.client import engine, session_local
from hive.db.mixins import read_sql_query
from hive.db.models import TbRiskScore, TbStrategy, TbMeta, TbUniverse, TbTicker, TbDailyBar

tickers = ['SPY', 'AGG']
with session_local() as session:

    query = (
        session.query(TbDailyBar.trd_dt, TbDailyBar.gross_rtn, TbMeta.ticker)
        .join(TbMeta, TbMeta.stk_id == TbDailyBar.stk_id)
        .filter(TbMeta.ticker.in_(tickers))
    )
    data = read_sql_query(query)
    print(data)