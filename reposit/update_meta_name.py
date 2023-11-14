from hive_old import db
from xbbg import blp

with db.session_local() as session:

    for meta in session.query(db.TbMeta).filter(db.TbMeta.id <= 1231).all():
        if meta.source not in ["naver", "yahoo", "bloomberg"]:
            continue
        if meta.iso_code == "KR":
            field = "NAME_KOREAN"
        else:
            field = "LONG_COMP_NAME"
        try:
            name = blp.bdp(meta.ticker_bloomberg, field).values[0][0]
        except:
            continue
        meta.name = name
        session.add(meta)
        session.flush()

    session.commit()
