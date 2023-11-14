import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))


from hive import db
from datetime import date, timedelta
from dateutil import parser
import pandas as pd
import yfinance as yf

# data = pd.read_excel("//Mac/Home/Downloads/kb_universe_v2.xlsx")
# tickers = data["Ticker"].tolist()

asofdate = parser.parse("2023-04-01")
start_date = asofdate - pd.DateOffset(years=3)

price = yf.download(tickers=["BAR"])["Adj Close"]
price_3Y = price.loc[start_date:asofdate]


# price_3Y.index = pd.to_datetime(price_3Y)

def cal(p: pd.Series) -> float:

    start = p.dropna().index[0]
    diff = (start - start_date).days

    if diff >= 3:
        return 0

    return p.resample("W").last().pct_change().fillna(0).std() * (52**0.5)


result = price_3Y.aggregate(cal, axis=0)

print(result)