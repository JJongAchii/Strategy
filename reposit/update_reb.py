from hive_old import db
import sqlalchemy as sa
import pandas as pd


with db.session_local() as session:

    market = "KR"
    level = 3

    uni = pd.read_csv("../core/KR_3_allocation.csv", index_col='ticker')

    query = (
        session.query(
            db.TbPortfolio.id.label('portfolio_id'),
            db.TbMeta.id.label('meta_id'),
            db.TbMeta.ticker.label('ticker')
        )
        .filter(db.TbPortfolio.portfolio.like(f"%{market}_{level}%"))
    )
    data = db.read_sql_query(query)

    uni.index = uni.index.astype(str)
    uni['date'] = '2023-03-02'
    uni = uni.rename(columns={'2023-03-02 weight': 'weights'})
    uni = uni.merge(data, on='ticker')[['date', 'portfolio_id', 'meta_id', 'weights']]
    print(uni)