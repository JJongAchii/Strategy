from hive import db
import pandas as pd
from datetime import date


# with db.session_local() as session:
#     query = (
#         session.query(
#             db.TbDailyBar.trd_dt,
#             db.TbDailyBar.stk_id,
#             db.TbDailyBar.close_prc,
#             db.TbDailyBar.gross_rtn,
#             db.TbDailyBar.adj_value
#         )
#         .filter(db.TbDailyBar.trd_dt >= '2023-04-03')
#     )
#
#     data = db.read_sql_query(query)
#     data.to_csv("dkd.csv")
#     print(data)

# with db.session_local() as session:
#     query = (
#         session.query(
#             db.TbPortAlloc,
#             db.TbMeta.ticker,
#         )
#         .join(db.TbMeta, db.TbMeta.stk_id == db.TbPortAlloc.stk_id)
#         .filter(db.TbPortAlloc.rebal_dt == '2023-04-03')
#         .filter(db.TbPortAlloc.port_id == 3)
#     )
#
#     data = db.read_sql_query(query)
#     print(data)
