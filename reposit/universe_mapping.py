import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

universe = pd.read_csv("universe.csv")
price = pd.read_csv("price.csv", index_col=["date"], parse_dates=["date"])
allo = pd.read_csv("us5_allocation.csv", index_col=["date"], parse_dates=["date"])
kr_universe = universe[universe.market == "KS"]
kr_tickers = kr_universe.ticker.tolist()
kr_weights = {}

maps = []
for date, weights in allo.iterrows():
    print(date)
    kr_weights[date] = {}

    weights = weights[weights != 0].dropna().sort_values(ascending=False)

    p = price.loc[:date].iloc[-252:].dropna(thresh=200, axis=1).dropna()
    mask = p.columns.isin(kr_tickers)
    filtered_df = p.loc[:, mask]

    for asset, weight in weights.items():

        ac = universe[universe.ticker == asset].asset_class.iloc[0]

        kractickers = kr_universe[kr_universe.asset_class == ac].ticker.tolist()
        mask = filtered_df.columns.isin(kractickers)
        final_filtered_df = filtered_df.loc[:, mask]

        us_norm = (p[asset] - p[asset].mean()) / p[asset].std()
        distances = {}
        for kr in final_filtered_df:
            kr_norm = (final_filtered_df[kr] - final_filtered_df[kr].mean()) / final_filtered_df[kr].std()
            distance = euclidean_distances(kr_norm.values.reshape(1, -1), us_norm.values.reshape(1, -1))
            distances[kr] = distance[0][0]

        distances = pd.Series(distances).sort_values()

        select = distances.index[0]

        maps.append(pd.Series({"date": date, "us": asset, "w": weight, "kr": select}))
        print(asset, weight, select)
        kr_weights[date].update({select: weight})
        filtered_df = filtered_df.drop(select, axis=1)

print(kr_weights)