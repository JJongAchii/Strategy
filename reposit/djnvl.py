from hive import db
import sqlalchemy as sa
import pandas as pd

with db.session_local() as session:
    query = (
        session.query(
            db.TbPortBook,
            db.TbRiskScore.risk_score
        )
        .join(db.TbRiskScore, db.TbRiskScore.stk_id == db.TbPortBook.stk_id)
        .filter(db.TbPortBook.port_id == 9)
    )
    data = db.read_sql_query(query)
    data.to_clipboard()

# with db.session_local() as session:
#     query = (
#         session.query(
#             db.TbDailyBar.trd_dt,
#             db.TbDailyBar.stk_id,
#             db.TbDailyBar.gross_rtn
#         )
#         .filter(
#             sa.and_(
#                 db.TbDailyBar.trd_dt >= '2018-01-02',
#                 db.TbDailyBar.stk_id.in_([7, 85])
#             )
#         )
#     )
#     data = db.read_sql_query(query).sort_values("trd_dt")
#     data = data.pivot(index="trd_dt", columns="stk_id", values="gross_rtn")
#     print(data)
    # bm = r.sum(axis=1).add(1).cumprod() * 1000
    # bm.to_clipboard()
