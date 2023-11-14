import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))

from typing import Optional
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

from hive import db
from core.analytics.pa import metrics


def store_nav_results(func):
    """Decorator for storing nav results"""
    book_results = {}
    nav_results = {}
    
    def wrapper(
        weight: pd.DataFrame,
        strategy_name: Optional[str] = None,
        price: Optional[pd.DataFrame] = None,
        fx: Optional[pd.DataFrame] = None,
        start_date: ... = None,
        end_date: ... = None,
        currency: ... = None
    ):
        book, nav = func(weight, price, fx, start_date, end_date, currency)
        
        if strategy_name:
            params = f"{strategy_name}"
        else:
            params = f"strategy_{wrapper.count}"
            wrapper.count += 1

        book_results[params] = book
        nav_results[params] = nav
        
        return book_results, nav_results
    
    wrapper.count = 1
    return wrapper


def price_apply_fx(
    price: Optional[pd.DataFrame] = None,
    fx: Optional[pd.DataFrame] = None,
    currency = "KRW"
) -> pd.DataFrame:
    """
    Apply foreign exchange rates to adjust prices in different currencies.

    Args:
        price (Optional[pd.DataFrame], optional): DataFrame containing prices. Defaults to None.
        fx (Optional[pd.DataFrame], optional): DataFrame containing foreign exchange rates. Defaults to None.
        currency (str, optional): Target currency for conversion. Defaults to "KRW".

    Returns:
        pd.DataFrame: DataFrame with adjusted prices.
    """
    
    market = db.get_market(price.columns.tolist())['iso_code']
    
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


@store_nav_results
def calculate_nav(
    weight: pd.DataFrame,
    price: Optional[pd.DataFrame] = None,
    fx: Optional[pd.DataFrame] = None,
    start_date: ... = None,
    end_date: ... = None,
    currency: ... = None,
) -> pd.DataFrame:
    """
    Calculate the net asset value (NAV) and portfolio holdings 
    based on the provided weight DataFrame and price data.

    Args:
        weight (pd.DataFrame): DataFrame containing the portfolio weights with tickers as columns and dates as index.
        price (Optional[pd.DataFrame], optional): DataFrame containing the price data. Defaults to None, which retrieves prices from a database.
        fx (Optional[pd.DataFrame], optional): DataFrame containing the foreign exchange rates. Defaults to None.
        start_date (Optional[Union[str, pd.Timestamp]], optional): Start date of the analysis. Defaults to None, which uses the earliest date in the weight DataFrame.
        end_date (Optional[Union[str, pd.Timestamp]], optional): End date of the analysis. Defaults to None, which uses the latest date in the price data.
        currency (Optional[str], optional): Currency for converting prices. Defaults to None, which means that either "KRW" or "USD" will be used.

    Returns:
        pd.DataFrame: A tuple containing the portfolio holdings (book) DataFrame and the NAV (nav) DataFrame.
    """
    
    weight.columns = [str(column).zfill(6) if isinstance(column, int) else column for column in weight.columns]
    
    #get price from database within tickers
    if price is None:
        price = db.get_price(tuple(weight.columns))
        
    start_date = start_date or weight.index[0]
    start_date = pd.to_datetime(start_date)
    price.index = pd.to_datetime(price.index)
    weight.index = pd.to_datetime(weight.index)

    
    if end_date is not None:
        price = price.loc[start_date:end_date]
        weight = weight.loc[start_date:end_date]
    else:
        price = price.loc[start_date:]
        weight = weight.loc[start_date:]
    
    if currency is not None:
        price = price_apply_fx(price=price, fx=fx, currency=currency)
    
    book = pd.DataFrame(columns=["Date", "ticker", "weights"])
    nav = pd.DataFrame([[start_date, 1000]], columns=["Date", "value"])
    
    rebal_list = weight.index.unique()
    
    for i, rebal_date in enumerate(rebal_list):
        if i == 0:
            last_nav = nav.value.iloc[-1]
        else:
            last_nav = nav.value.iloc[-2]
        rebal_weights = weight[weight.index == rebal_date].stack()
        rebal_weights.index = rebal_weights.index.droplevel(0)
        
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
        value = last_nav * price_weights.sum(axis=1)
        value = value[1:].reset_index()
        value.columns = ["Date", "value"]
        
        weights = weights.stack().reset_index()
        weights.columns = ["Date", "ticker", "weights"]
        
        book = book.append(weights)
        nav = nav.append(value)
        
    book = book.set_index("Date")
    nav = nav.set_index("Date")

    return book, nav


def result_metrics(nav: pd.DataFrame) -> None:
    """
    Display the performance metrics calculated from the provided DataFrame.

    Args:
        nav (pd.DataFrame): The DataFrame containing the net asset values (NAV) data.
    """
    ann_returns = metrics.ann_returns(nav)
    ann_volatilities = metrics.ann_volatilities(nav)
    sharpe_ratios = metrics.sharpe_ratios(nav)
    max_drawdowns = metrics.max_drawdowns(nav)
    skewness = metrics.skewness(nav)
    kurtosis = metrics.kurtosis(nav)
    value_at_risk = metrics.value_at_risk(nav)
    conditional_value_at_risk = metrics.conditional_value_at_risk(nav)
    
    # Prepare the data as a list 
    data = [
        ["Annualized Returns", ann_returns.values.tolist()],
        ["Annualized Volatilities", ann_volatilities.values.tolist()],
        ["Sharpe Ratios", sharpe_ratios.values.tolist()],
        ["Max DrawDowns", max_drawdowns.values.tolist()],
        ["Skewness", skewness.values.tolist()],
        ["Kurtosis", kurtosis.values.tolist()],
        ["Value at Risk", value_at_risk.values.tolist()],
        ["Conditional Value at Risk", conditional_value_at_risk.values.tolist()]
    ]
    
    data = [[label] + values for label, values in data]
    print(tabulate(data, headers=["Metrics"] + nav.columns.tolist(), tablefmt="fancy_grid"))
    
    # Plot NAV over time
    plt.figure(figsize=(10, 6))
    for column in nav.columns:
        plt.plot(nav.index, nav[column], label=column)
    plt.xlabel('Date')
    plt.ylabel('Net Asset Value (NAV)')
    plt.title('Net Asset Value Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # weight = pd.read_excel("result/mlp_allocation_us.xlsx", sheet_name="US_5", index_col="Date")
    # price = pd.read_csv("result/price_us.csv", index_col="trd_dt")
    #fx = pd.read_csv("reposit/fx.csv", index_col='Date')
    price = pd.read_csv("reposit/equity/us_equity_close.csv", index_col="DATE", parse_dates=["DATE"])
    weight = pd.read_csv("reposit/equity/rsi_result.csv", index_col="DATE", parse_dates=["DATE"])
    
    book, nav = calculate_nav(weight=weight, price=price)
    print(nav)
    result_metrics(nav=nav)