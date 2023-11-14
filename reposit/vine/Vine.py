import pandas as pd
import pandas_datareader as pdr

uni = pd.read_clipboard()

prices = list()

for ticker in uni.squeeze():
    try:
        price = pdr.DataReader(ticker, "naver", start="1990-1-1")["Close"].astype(float)
        price.name = ticker
        prices.append(price)
    except Exception as exc:
        print(ticker, " does not exist.", exc)

prices = pd.concat(prices, axis=1)
prices.index = pd.to_datetime(prices.index)
prices = prices.sort_index()
prices