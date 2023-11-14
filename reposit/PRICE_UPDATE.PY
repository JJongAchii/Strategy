from hive import db
import pandas_datareader as pdr
import yfinance as yf

for meta in db.TbMeta.query().all():

    if meta.source == "naver":
        continue
        data = pdr.DataReader(meta.ticker_naver, "naver", "1900-1-1").astype(float)
        data.index.name = "date"
        data.columns = data.columns.str.upper()
        data.columns = data.columns.str.replace(" ", "_")
        data["GROSS_RETURN"] = data["CLOSE"].pct_change().fillna(0)
        data["INDEX"] = data["GROSS_RETURN"].add(1).cumprod()
        data.columns.name = "field"
        data = data.stack()
        data.name = "value"
        data = data.reset_index()
        data["meta_id"] = meta.id
        db.TbTimeSeries.delete(meta_id=meta.id)
        db.TbTimeSeries.insert(data)
        print(meta.ticker, "update complete")

    elif meta.source == "yahoo":
        data = yf.download(meta.ticker_yahoo, "1980-1-1")
        data.columns = data.columns.str.upper()
        data.columns = data.columns.str.replace(" ", "_")
        data["GROSS_RETURN"] = data["ADJ_CLOSE"].pct_change().fillna(0)
        data["INDEX"] = data["GROSS_RETURN"].add(1).cumprod()
        data = data.drop(labels=["ADJ_CLOSE"], axis=1)
        data.columns.name = "field"
        data.index.name = "date"
        data = data.stack()
        data.name = "value"
        data = data.reset_index()
        data["meta_id"] = meta.id
        db.TbTimeSeries.delete(meta_id=meta.id)
        db.TbTimeSeries.insert(data)
        print(meta.ticker, "update complete")
    # elif meta.source == "fred":
    #     continue
    #     data = pdr.DataReader(meta.ticker_fred, "fred", "1900-1-1").astype(float)
    #     data.index.name = "date"
    #     data.columns = ["value"]
    #     data = data.reset_index().dropna()
    #     data["meta_id"] = meta.id
    #     data["field"] = "index"

    # elif meta.source == "bloomberg":
    #     if meta.security_class.lower() == "equity":
    #         flds = "TOT_RETURN_INDEX_GROSS_DVDS"
    #         data = blp.bdh(
    #             tickers=meta.ticker_bloomberg, flds=flds, start_date="1900-1-1"
    #         ).dropna()
    #         data.columns = ["value"]
    #         data = data.pct_change().fillna(0)
    #         data["field"] = "gross_return"
    #     else:
    #         flds = "PX_LAST"
    #         data = blp.bdh(
    #             tickers=meta.ticker_bloomberg, flds=flds, start_date="1900-1-1"
    #         ).dropna()
    #         data.columns = ["value"]
    #         data["field"] = "index"

    #     data.index.name = "date"
    #     data = data.reset_index()
    #     data["meta_id"] = meta.id
    #     db.TbTimeSeries.insert(data)
    #     print(f"{meta.ticker} | complete!")

    # else:
    #     continue
