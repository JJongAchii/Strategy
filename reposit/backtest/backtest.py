import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.analytics.pa import metrics as mt
import pandas as pd
from hive import db

"""get portfolio allocation"""
# with db.session_local() as session:
#     query = (
#         session.query(
#             db.TbBacktestAlloc.rebal_dt.label("date"),
#             db.TbPort.portfolio,
#             db.TbMeta.ticker,
#             db.TbTicker.ticker_bloomberg,
#             db.TbUniverse.strg_asset_class,
#             db.TbBacktestAlloc.weights
#         )
#         .join(db.TbPort, db.TbPort.port_id == db.TbBacktestAlloc.port_id)
#         .join(db.TbMeta, db.TbMeta.stk_id == db.TbBacktestAlloc.stk_id)
#         .join(db.TbTicker, db.TbTicker.stk_id == db.TbBacktestAlloc.stk_id)
#         .join(db.TbUniverse, db.TbUniverse.stk_id == db.TbBacktestAlloc.stk_id)
#         .filter(db.TbUniverse.strategy_id == 2)
#         .distinct()
#     )
#     data = db.read_sql_query(query)
#     data.to_clipboard()


price = pd.read_clipboard().set_index("DATE")
price.index = pd.to_datetime(price.index)

ret = mt.ann_returns(price_df=price)
vol = mt.ann_volatilities(price_df=price)
sharp = mt.sharpe_ratios(price_df=price)
mdd = mt.max_drawdowns(price_df=price)

result = pd.concat([ret, vol, sharp, mdd], axis=1)
result.columns = ['ann_returns', 'ann_vol', 'sharp_ratio', 'mdd']
result.to_clipboard()
