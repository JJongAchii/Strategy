import pandas as pd
import pandas_datareader as pdr



def rsi(
    close: pd.DataFrame,
    length: int = 14,
    drift: int = 1,
    offset: int = 0
) -> pd.DataFrame:
    """
    Indicator: Relative Strength Index (RSI)

    Args:
        close (pd.DataFrame): _description_
        length (int, optional): _description_. Defaults to 14.
        drift (int, optional): _description_. Defaults to 1.
        offset (int, optional): _description_. Defaults to 0.

    Returns:
        pd.DataFrame: _description_
    """
    
    # Implement Date-Based close Sorting
    close = close.sort_index()
    
    # Calculate Result
    close_diff = close.diff(drift)
    
    positive = close_diff.copy()
    negative = close_diff.copy()
    positive[positive < 0] = 0  # Make negatives 0 for the postive series
    negative[negative > 0] = 0  # Make postives 0 for the negative series

    positive_avg = positive.ewm(com=length, min_periods=length).mean()
    negative_avg = negative.ewm(com=length, min_periods=length).mean()
    
    rsi = 100 * positive_avg / (positive_avg + negative_avg.abs())
    
    return rsi.shift(offset)


def simulate(price: pd.DataFrame):

    rsi21 = rsi(close=price, length=21)
    rsi5 = rsi(close=price, length=5)

    MA200 = price.rolling(200).mean()
    MA21 = price.rolling(21).mean()
    
    #종가와 21일 이평 이격도 Dataframe
    disparity = (price - MA21) / MA21

    holdings = pd.DataFrame()
    weights = pd.Series()
    
    previous_month = None
    
    for i, trade_date in enumerate(price.index):
        
        if trade_date < price.index[i-1]:
            continue
        
        current_month = trade_date.month

        if previous_month is None or current_month != previous_month:
            
            buy_signal_asset = price.loc[price.index == price.index[i-1], 
                            (price.iloc[i-1] > MA200.iloc[i-1]) &
                                (rsi21.iloc[i-1] < 50) &
                                (rsi5.iloc[i-2] > 20)&(rsi5.iloc[i-1] < 20)]
            
            if buy_signal_asset.empty is False:
                buy_signal_asset = buy_signal_asset[buy_signal_asset.columns[~buy_signal_asset.columns.isin(holdings)]]
                buy_signal_asset = buy_signal_asset.columns.tolist()
                sorted_asset = disparity.loc[disparity.index == disparity.index[i-1], buy_signal_asset].sort_values(by=disparity.index[i-1], axis=1)            
                
                add_asset = sorted_asset[sorted_asset.columns[:5]].columns
                
                add_asset = pd.DataFrame({"ticker": add_asset, "weight": 1 / len(add_asset)})
                add_asset.index = [trade_date] * len(add_asset)

                holdings = pd.concat([holdings, add_asset])
                
        previous_month = current_month     
    
    weights_df = pd.DataFrame({'ticker': holdings.ticker, 'value': holdings.weight})
    weights_df.index.name = "DATE"
    weights_df = weights_df.pivot(columns="ticker", values='value')
    
    return weights_df


def simulate_monthly(price: pd.DataFrame):

    rsi21 = rsi(close=price, length=21)
    rsi5 = rsi(close=price, length=5)

    MA200 = price.rolling(200).mean()
    MA21 = price.rolling(21).mean()
    
    #종가와 21일 이평 이격도 Dataframe
    disparity = (price - MA21) / MA21

    holdings = pd.DataFrame()
    weights = pd.Series()
    
    previous_month = None
    
    for i, trade_date in enumerate(price.index):
        
        if trade_date < price.index[i-1]:
            continue
        
        current_month = trade_date.month

        if previous_month is None or current_month != previous_month:
            
            buy_signal_asset = price.loc[price.index == price.index[i-1], 
                            (price.iloc[i-1] > MA200.iloc[i-1]) &
                                (rsi21.iloc[i-1] < 50) &
                                (rsi5.iloc[i-2] > 20)&(rsi5.iloc[i-1] < 20)]
            
            if buy_signal_asset.empty is False:
                buy_signal_asset = buy_signal_asset.columns.tolist()
                sorted_asset = disparity.loc[disparity.index == disparity.index[i-1], buy_signal_asset].sort_values(by=disparity.index[i-1], axis=1)            
                
                add_asset = sorted_asset[sorted_asset.columns[:5]].columns
                
                add_asset = pd.DataFrame({"ticker": add_asset, "weight": 1 / len(add_asset)})
                add_asset.index = [trade_date] * len(add_asset)

                holdings = pd.concat([holdings, add_asset])
                
        previous_month = current_month     
    
    weights_df = pd.DataFrame({'ticker': holdings.ticker, 'value': holdings.weight})
    weights_df.index.name = "DATE"
    weights_df = weights_df.pivot(columns="ticker", values='value')
    
    return weights_df


def rebalance(price: pd.DataFrame):

    rsi21 = rsi(close=price, length=21)
    rsi5 = rsi(close=price, length=5)

    MA200 = price.rolling(200).mean()
    MA21 = price.rolling(21).mean()
    
    #종가와 21일 이평 이격도 Dataframe
    disparity = (price - MA21) / MA21

    holdings = pd.DataFrame()
    
    buy_signal_asset = price.loc[price.index == price.index[-1], 
                    (price.iloc[-1] > MA200.iloc[-1]) &
                        (rsi21.iloc[-1] < 50) &
                        (rsi5.iloc[-2] > 20)&(rsi5.iloc[-1] < 20)]
    
    buy_signal_asset = buy_signal_asset[buy_signal_asset.columns[~buy_signal_asset.columns.isin(holdings)]]
    buy_signal_asset = buy_signal_asset.columns.tolist()
    sorted_asset = disparity.loc[disparity.index == disparity.index[-1], buy_signal_asset].sort_values(by=disparity.index[-1], axis=1)            
    
    add_asset = sorted_asset[sorted_asset.columns[:5]].columns
    add_asset = pd.DataFrame({"ticker": add_asset, "weight": 1 / len(add_asset)})

    return add_asset


if __name__ == "__main__":

    price = pd.read_csv("reposit/equity/us_equity_close.csv", index_col="DATE", parse_dates=["DATE"])
    #weight = pd.read_csv("reposit/us_equity_weights.csv", index_col="DATE", parse_dates=["DATE"])
    result = simulate_monthly(price=price)
    result.to_csv("reposit/equity/rsi_result_monthly.csv")