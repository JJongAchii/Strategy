import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../..")))

import pandas as pd
import cvxpy as cp
from typing import Optional
import logging

from hive import db
import sqlalchemy as sa
from core.analytics import backtest


logger = logging.getLogger("sqlite")


class VirtualAccount:
    
    def __init__(
        self,
        strategy: str = "mlp",
        market: str = "us",
        level: int = 5,
        portfolio_value: float = None,
        trade_price: str = "close",
    ) -> None:
        
        strategy = strategy.upper()
        market = market.upper()
        self.shares_result = pd.DataFrame(columns=["rebal_dt", "ticker", "shares"])
        self.ap_weights = pd.DataFrame(columns=["rebal_dt", "ticker", "ap_weights"])
        self.portfolio_value = portfolio_value
        if portfolio_value is None:
            self.portfolio_value = 3_000_000 if market == "KR" else 2500
        
        self.data = db.get_alloc_weight_for_shares(
            strategy=strategy, market=market, level=level
        )
        self.weights = self.data.pivot(index="rebal_dt", columns="ticker", values="weights")
        self.weights.index.name = "Date"
        
        
    def calculate_shares(
        self,
        w: pd.Series,
        p: pd.Series,
        min_shares: int = 1,
        portfolio_value: float = 3_000_000
    ) -> pd.Series:
        num_asset = len(w)

        shares = cp.Variable(num_asset, integer=True)
        cash = portfolio_value - p.values @ shares

        bound = cp.Variable(num_asset)

        eta = w.values * portfolio_value - cp.multiply(shares, p.values)
        obj = cp.sum(bound) + cash
        cons = [eta <= bound, eta >= -bound, cash >= 0, shares >= min_shares]

        prob = cp.Problem(objective=cp.Minimize(obj), constraints=cons)

        prob.solve(verbose=False)

        return pd.Series(index=w.index, data=shares.value, name="shares")

    
    def calculate_virtual_account_nav(
        self,
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
            close_price = db.get_close_price(tuple(weight.columns))
        
        start_date = start_date or weight.index[0]
        start_date = pd.to_datetime(start_date)
        price.index = pd.to_datetime(price.index)
        weight.index = pd.to_datetime(weight.index)
        close_price.index = pd.to_datetime(close_price.index)
        
        if end_date is not None:
            price = price.loc[start_date:end_date]
            weight = weight.loc[start_date:end_date]
        else:
            price = price.loc[start_date:]
            weight = weight.loc[start_date:]
        
        if currency is not None:
            price = backtest.price_apply_fx(price=price, fx=fx, currency=currency)
            
        book = pd.DataFrame(columns=["Date", "ticker", "weights"])
        nav = pd.DataFrame([[start_date, self.portfolio_value]], columns=["Date", "value"])
        rebal_list = weight.index.unique()
        
        for i, rebal_date in enumerate(rebal_list):
            if i == 0:
                last_nav = nav.value.iloc[-1]
            else:
                last_nav = nav.value.iloc[-2]
            rebal_weights = weight[weight.index == rebal_date].stack()
            rebal_weights.index = rebal_weights.index.droplevel(0)
            previous_date = close_price[close_price.index < rebal_date].last_valid_index()
            rebal_price = close_price[close_price.index == previous_date][rebal_weights.index].squeeze()
            
            shares = self.calculate_shares(p=rebal_price, w=rebal_weights, portfolio_value=last_nav)
            
            result = pd.DataFrame({'rebal_dt': rebal_date, 'ticker': shares.index, 'shares': shares}).reset_index()
            self.shares_result = self.shares_result.append(result)
            
            capital = rebal_price * shares
            cash = last_nav - capital.sum()
            
            weights = pd.DataFrame({'rebal_dt': rebal_date, 'ap_weights': capital/last_nav}).reset_index()
            self.ap_weights = self.ap_weights.append(weights)
            
            if i == len(rebal_list) - 1:
                end_rebal = price.index[-1]
            else:
                end_rebal = rebal_list[i+1]

            price_slice = price[(price.index >= rebal_date) & (price.index <= end_rebal)][rebal_weights.index]
            if price_slice.empty:
                continue
            price_returns = price_slice / price_slice.iloc[0]
            price_returns = price_returns.multiply(capital, axis=1)
            
            weights = price_returns.div(price_returns.sum(axis=1), axis=0)[:-1]
            value = price_returns.sum(axis=1) + cash
            value = value[1:].reset_index()
            value.columns = ["Date", "value"]
            
            weights = weights.stack().reset_index()
            weights.columns = ["Date", "ticker", "weights"]
            
            book = book.append(weights)
            nav = nav.append(value)
        
        book = book.set_index("Date")
        nav = nav.set_index("Date")

        return book, nav
            

    def update_shares_db(self): 
        
        updated_sh = pd.merge(self.data, self.shares_result, on=["rebal_dt", "ticker"], how="left")
        updated_sh['shares'] = updated_sh['shares_y']

        updated_sh = updated_sh.drop(['shares_x', 'shares_y'], axis=1)

        updated_sh = pd.merge(updated_sh, self.ap_weights, on=["rebal_dt", "ticker"], how="left")
        updated_sh['ap_weights'] = updated_sh['ap_weights_y']
        
        updated_sh = updated_sh.drop(['ap_weights_x', 'ap_weights_y'], axis=1)
        
        db.TbPortAlloc.update(updated_sh)


if __name__ == "__main__":
    
    account = VirtualAccount(strategy="mlp", market="kr", level=5)
    #result = account.update_shares()
    book, nav = account.calculate_virtual_account_nav(weight=account.weights)
    book.to_clipboard()
    
    # account.update_shares_db()
    # from core.analytics.backtest import strategy_analytics
    
    # result = strategy_analytics.result_metrics(nav=nav)
    