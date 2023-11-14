import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))


from hive import db
from datetime import date, timedelta
from dateutil import parser
import pandas as pd
import numpy as np
import yfinance as yf

# data = pd.read_excel("//Mac/Home/Downloads/kb_universe_v2.xlsx")
# tickers = data["Ticker"].tolist()

asofdate = parser.parse("2023-04-01")
start_date = asofdate - pd.DateOffset(years=3)

price = yf.download(tickers=["EDV"])["Adj Close"]
price = price.pct_change()
price_3Y = price.loc[start_date:asofdate]


# price_3Y.index = pd.to_datetime(price_3Y)


def cal(p: pd.Series) -> float:

    loss_rate_2p5 = np.percentile(p, 2.5)
    absolute_loss_rate = abs(loss_rate_2p5)
    annualization_adjustment = np.sqrt(250)
    result = absolute_loss_rate * annualization_adjustment

    return result

result = cal(p=price_3Y)
print(result)


