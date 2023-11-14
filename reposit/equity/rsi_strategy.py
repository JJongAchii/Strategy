
import yfinance as yf
import pandas as pd
import pandas_datareader as pdr


def calculate_rsi(price: pd.DataFrame):

    price_diff = price.diff()

    plus_df = price_diff.copy()
    plus_df[price_diff < 0] = 0
    minus_df = abs(price_diff.copy())
    minus_df[price_diff > 0] = 0

    AU14 = plus_df.ewm(com=13).mean()
    AD14 = minus_df.ewm(com=13).mean()
    RSI14 = 100 - 100 / (1 + AU14 / AD14)

    signal = RSI14.ewm(span=9).mean()


    AU2 = plus_df.ewm(com=1).mean()
    AD2 = minus_df.ewm(com=1).mean()
    RSI2 = 100 - 100 / (1 + AU2 / AD2)

    return RSI14, RSI2


def simulate(price: pd.DataFrame):

    rsi14, rsi2 = calculate_rsi(price)

    MA200 = price.rolling(200).mean()
    MA5 = price.rolling(5).mean()

    #종가와 5일 이평 이격도 Dataframe
    disparity = (price - MA5) / MA5

    holdings = pd.Series()
    weights = pd.Series()
    
    for i, trade_date in enumerate(price.index):
        
        if trade_date < price.index[i-1]:
            continue
        
        buy_signal_asset = price.loc[price.index == price.index[i-1], 
                          (price.iloc[i-1] > MA200.iloc[i-1]) &
                            (price.iloc[i-1] < MA5.iloc[i-1]) &
                              (rsi14.iloc[i-1] < 50) &
                              (rsi2.iloc[i-2] > 5) & (rsi2.iloc[i-1] < 5)]
        
        hold_asset_rsi = rsi2.loc[rsi2.index == rsi2.index[i-1], holdings]
        rsi_sell_signal_asset = hold_asset_rsi[hold_asset_rsi > 5].dropna(axis=1).columns.tolist()
        limit5_asset = holdings[holdings.index == price.index[i-5]].tolist()
        delete_asset = rsi_sell_signal_asset + limit5_asset
        
        holdings = holdings[~holdings.isin(delete_asset)]
        

        if buy_signal_asset.empty is False:
            buy_signal_asset = buy_signal_asset[buy_signal_asset.columns[~buy_signal_asset.columns.isin(holdings)]]
            buy_signal_asset = buy_signal_asset.columns.tolist()
            sorted_asset = disparity.loc[disparity.index == disparity.index[i-1], buy_signal_asset].sort_values(by=disparity.index[i-1], axis=1)            

            add_asset_size = 20 - holdings.size
            if add_asset_size < 0:
                add_asset_size = 0
            
            add_asset = sorted_asset[sorted_asset.columns[:add_asset_size]].columns
            add_asset = pd.Series(add_asset, index=[trade_date] * len(add_asset))
            
            holdings = pd.concat([holdings, add_asset])
            

            add_weights = pd.Series(holdings.values, index=[trade_date] * len(holdings))
            weights = pd.concat([weights, add_weights])
    weights_df = pd.DataFrame({'ticker': weights, 'value': 0.05})
    weights_df = weights_df.pivot(columns="ticker", values='value')
    weights_df.to_clipboard()
    #weights_df.to_csv("reposit/us_equity_weights.csv")
    
    
def calulate_nav(price: pd.DataFrame, weight: pd.DataFrame):
    
    returns = price.pct_change() + 1
    
    book = returns[weight.shift(1).notnull()].cumprod() * weight.shift(1) * 1000
    # book = (returns * weight.shift(1)).cumprod()
    nav = book.sum(axis=1) + (20 - book.count(axis=1)) * 50
    nav.to_clipboard()            

            
            
            
        # data = price.loc[price.index[i-1], :][price.iloc[i-1] > MA200.iloc[i-1]]




if __name__ == "__main__":

    price = pd.read_csv("reposit/kr_equity_price.csv", index_col="DATE", parse_dates=["DATE"])
    weight = pd.read_csv("reposit/kr_equity_weights.csv", index_col="DATE", parse_dates=["DATE"])
    #result = simulate(price=price)
    nav = calulate_nav(price=price, weight=weight)