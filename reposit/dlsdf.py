from hive import db

with db.session_local() as session:
    query = (
        session.query(
            db.TbBacktestAlloc.rebal_dt,
            db.TbBacktestAlloc.weights,
            db.TbMeta.ticker,
            db.TbMeta.name,
            db.TbUniverse.strg_asset_class
        )
        .join(db.TbUniverse, db.TbUniverse.stk_id == db.TbBacktestAlloc.stk_id)
        .join(db.TbMeta, db.TbMeta.stk_id == db.TbBacktestAlloc.stk_id)
        .filter(db.TbBacktestAlloc.rebal_dt == '2021-02-01')
        .filter(db.TbBacktestAlloc.port_id == 6)
        .distinct()
    )

    data = db.read_sql_query(query)

    data.to_clipboard()