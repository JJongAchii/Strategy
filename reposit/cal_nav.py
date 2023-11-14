import os
import sys
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))
from hive import db
import pandas as pd


def price_apply_fx(
    price: Optional[pd.DataFrame] = None,
    fx: Optional[pd.DataFrame] = None,
    currency = "KRW"
) -> pd.DataFrame:
    """Apply foreign exchange rates to adjust prices in different currencies.

    Args:
        price (Optional[pd.DataFrame], optional): DataFrame containing prices. Defaults to None.
        fx (Optional[pd.DataFrame], optional): DataFrame containing foreign exchange rates. Defaults to None.
        currency (str, optional): Target currency for conversion. Defaults to "KRW".

    Returns:
        pd.DataFrame: DataFrame with adjusted prices.
    """
    
    market = db.get_market(weight.columns.tolist())['iso_code']
    
    if fx is None:
        fx = db.get_fx()

    fx.index = pd.to_datetime(fx.index)
    
    price.loc[:, market == "US"] = price.loc[:, market == "US"].multiply(
            (fx.close_prc[fx.currency == "USD"] / fx.close_prc[fx.currency == currency]), axis=0
        ).reindex_like(price)
    
    price.loc[:, market == "KR"] = price.loc[:, market == "KR"].multiply(
            (fx.close_prc[fx.currency == "KRW"] / fx.close_prc[fx.currency == currency]), axis=0
        ).reindex_like(price)
    
    return price


def cal_nav(
    weight: pd.DataFrame,
    price: Optional[pd.DataFrame] = None,
    fx: Optional[pd.DataFrame] = None,
    start_date: ... = None,
    end_date: ... = None,
    currency: ... = None,
) -> pd.DataFrame:
    """_summary_

    Args:
        weight (pd.DataFrame): _description_
        price (Optional[pd.DataFrame], optional): _description_. Defaults to None.
        fx (Optional[pd.DataFrame], optional): _description_. Defaults to None.
        start_date (None, optional): _description_. Defaults to None.
        currency (str, optional): _description_. Defaults to "KRW".

    Returns:
        pd.DataFrame: _description_
    """
    
    weight.columns = [str(column).zfill(6) if isinstance(column, int) else column for column in weight.columns]
    
    #get price from database within tickers
    if price is None:
        price = db.get_price(tuple(weight.columns))
        
    start_date = start_date or weight.index[0]
    start_date = pd.to_datetime(start_date)
    price.index = pd.to_datetime(price.index)
    
    if end_date is not None:
        price = price.loc[start_date:end_date]
    else:
        price = price.loc[start_date:]
    
    if currency is not None:
        price = price_apply_fx(price=price, fx=fx, currency=currency)
    
    book = pd.DataFrame(columns=["Date", "ticker", "weights"])
    nav = pd.DataFrame([[start_date, 1000]], columns=["Date", "value"])
    
    rebal_list = weight.index.unique()
    
    for i, rebal_date in enumerate(rebal_list):
        rebal_weights = weight[weight.index == rebal_date].stack().reset_index(level='Date', drop=True)
        
        if i == len(rebal_list) - 1:
            end_rebal = price.index[-1]
        else:
            end_rebal = rebal_list[i+1]

        price_slice = price[(price.index >= rebal_date) & (price.index <= end_rebal)][rebal_weights.index]
        if price_slice.empty:
            continue
        price_returns = price_slice / price_slice.iloc[0]
        price_weights = price_returns.multiply(rebal_weights, axis=1)

        weights = price_weights.div(price_weights.sum(axis=1), axis=0)[:-1]
        value = nav.value.iloc[-1] * price_weights.sum(axis=1)
        value = value[1:].reset_index()
        value.columns = ["Date", "value"]
        
        weights = weights.stack().reset_index()
        weights.columns = ["Date", "ticker", "weights"]
        
        book = book.append(weights)
        nav = nav.append(value)
        
    book = book.set_index("Date")
    nav = nav.set_index("Date")

    return book, nav



if __name__ == "__main__":
    weight = pd.read_excel("result/mlp_allocation_us.xlsx", sheet_name="US_5", index_col="Date")
    price = pd.read_csv("result/price_us.csv", index_col="trd_dt")
    fx = pd.read_csv("reposit/fx.csv", index_col='Date')
    #book, nav = cal_nav(weight=weight)
    book, nav = cal_nav(weight=weight, price=price)
    print(book, nav)
    book.to_clipboard()