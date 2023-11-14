# PGA ETF allocation
import pandas_datareader as pdr
import pandas as pd

#allocations = pd.read_clipboard(index_col="ticker")
allocations = pd.read_excel("Factsheets Vertical.xlsm", sheet_name="2023-05", index_col='ticker').drop(columns='name')

navs = []
for allocation in allocations:

    weights = allocations[allocation]

    prices = []

    for asset, weight in weights.items():
        p = pdr.DataReader(asset.replace(".KS", ""), data_source="naver", start='1990-1-1')['Close'].astype(float)
        p.name = asset
        prices.append(p)
    prices = pd.concat(prices, axis=1)

    w = prices.loc["2023-3-31":] * weights.divide(prices.loc["2023-3-31"])
    n = w.sum(axis=1)
    n.name = allocation
    navs.append(n)

pd.concat(navs, axis=1).to_clipboard()