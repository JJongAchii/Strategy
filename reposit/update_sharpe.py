from hive_old import db
import sqlalchemy as sa

for id in [1, 2, 3, 4, 5, 6]:
    result = db.TbPortfolioValue.query_df(portfolio_id=id).sort_values('date')
    print(result)


    def mdd(x):
        return (x / x.expanding().max() - 1).min()


    def sharpe(x):
        re = x.pct_change().dropna().fillna(0)

        return re.mean() / re.std() * (252 ** 0.5)


    result["mdd_1y"] = result.value.rolling(252).apply(mdd)
    result["sharp_1y"] = result.value.rolling(252).apply(sharpe)

    import numpy as np

    result = result.replace(np.nan, None)

    with db.session_local() as session:
        session.bulk_update_mappings(db.TbPortfolioValue, result.to_dict("records"))
        session.commit()

    result.set_index("date").mdd_1y.plot()