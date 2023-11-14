import pandas as pd

from hive import db
from xbbg import blp


# data = blp.bdh("OEUSKLAC Index", "PX_LAST", start_date="1993-1-1").stack().stack().reset_index()
# data.columns = ['trd_dt', 'field', 'ticker', 'value']
# data['macro_id'] = 1
# data['adj_value'] = data.value
#
#
# db.TbMacro.insert([{"macro_id": 1, "macro_name": data.ticker[0]}])
# db.TbMacroData.delete()
# db.TbMacroData.insert(data)


with db.session_local() as session:
    query = session.query(
        db.TbMeta.stk_id,
        db.TbTicker.ticker_bloomberg
    ).join(db.TbMeta).filter(db.TbMeta.source == "bloomberg")

    ticker_mapper = {
        rec.ticker_bloomberg: rec.stk_id
        for rec in query.all()
    }

    data = blp.bdh(list(ticker_mapper.keys()), "PX_LAST", start_date="1993-1-1").stack().stack().reset_index()
    data.columns = ['trd_dt', 'field', 'ticker', 'close_prc']

    data['stk_id'] = data['ticker'].map(ticker_mapper)
    data['adj_value'] = data['close_prc']

    data.to_csv("index_price.csv")

    session.query(db.TbDailyBar).filter(db.TbDailyBar.stk_id.in_(list(ticker_mapper.values()))).delete()
    session.flush()

    db.TbDailyBar.insert(data, session=session)
    session.commit()



