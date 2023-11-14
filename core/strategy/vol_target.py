import pandas as pd
from hive import db
from datetime import datetime, date, timedelta
import numpy as np
class VolTargetStrategy:

    def get_vol_target_weights(self, ticker, target_vol=0, asofdate=datetime.now(), observation_year = 10, window = 5, risk_free_rate= 0.02, leverage_cap = 1.0):
        """
        A portfolio consists of one asset and cash.
        To stabilize the volatility of the portfolio, cash weighting is increased whenever volatility is high and vice versa.
        Weights of asset is decided as this formula
        
        weights_asset = min[(target_vol/realized_vol), leverage_cap]

        leverage_cap can be 1 when borrowing is not possible.
        When borrowing is possible, any number can be possible.

        Referrence: https://quantpedia.com/an-introduction-to-volatility-targeting/

        Args:
            ticker (str): Ticker \n
            target_vol (float): Target Volatility, Default value is 0.0, in which case, yearly standard deviation of the returns in the observation period is set as target Volatility.\n
            asofdate (datetime): calcualting date \n
            observation_year (int): observation period(year) of Time series Data \n
            window (int): To make the realized volatilty we use rolling windows. This is size of the window(day unit) \n
            risk_free_rate (float): Risk Free rate \n
            leverage_cap (float): limit of the leverage \n

        Returns:
            DataFrame \n
            weights : weights of the portfolio's assets \n
            prices : prices of the portfolio's assets \n
        
        Examples::
        
            from core.strategy.vol_target import VolTargetStrategy
            import matplotlib.pyplot as plt
            from core.analytics import backtest
            import pandas as pd
            from datetime import datetime, date, timedelta

            strategy = VolTargetStrategy()

            tickers=['LEGATRUU Index', 'MXWD Index', 'SPGSCITR Index', 'BIL']
            target_vols = [0.0, 0.10, 0.20]
            asofdate = (datetime.now()- timedelta(days=int(5*365.25)))
            observation_year = 15
            risk_free_rates = [0.0, 0.05]
            leverage_caps = [1.0, 2.0]

            i=0
            for ticker in tickers:
                for target_vol in target_vols:
                    for leverage_cap in leverage_caps:
                        i=i+1
                        if target_vol == 0.0:
                            s_vol="Mean(Vol)"
                        else:
                            s_vol="{:.0%}".format(target_vol)
                        print(f' {ticker}')
                        print(f' Target Vol:{s_vol}, Leverage Cap:{leverage_cap}')
                        weights, prices = strategy.get_vol_target_weights(ticker = ticker, target_vol = target_vol, asofdate = asofdate, observation_year = observation_year, risk_free_rate = 0.0, leverage_cap = leverage_cap)
                        book, nav = backtest.calculate_nav(weight = weights, price = prices)
                        compare=pd.merge(left = prices[ticker]/prices.iloc[0,0], right = nav[f'strategy_{i}']/1000, how = 'inner', left_index = True, right_index=True)
                        compare.columns=[ticker,'Volatility Target Strategy']
                        ax=compare.plot(figsize=(16,8))
                        ax.text(prices.index[0], 0.95, f' Target Vol:{s_vol}, Leverage Cap:{leverage_cap}')
                        plt.show()
                        del[[compare]]
                        del(nav[f'strategy_{i}'])        
        """
        querystr = "(trd_dt > '"+ (asofdate - timedelta(days=int(observation_year*365.25))).strftime("%Y-%m-%d") + "') and  (trd_dt < '"+ (asofdate).strftime("%Y-%m-%d") + "')"
        querystr2 = "trd_dt < '"+ (asofdate).strftime("%Y-%m-%d") + "'"
        price = db.get_price(ticker).query(querystr).sort_index()
        if len(price)==0:
            list_ticker=[ticker]
            price = db.get_macro_data_from_ticker(list_ticker, asofdate - timedelta(days=int(observation_year*365.25))).query(querystr2).sort_index()
        
        returns = price.sort_index().pct_change()
        vol = returns.std()*(252 ** 0.5) 
        if target_vol == 0:
            target_vol = vol # default value: mean value

        # realized vol calculation: windows size day vol
        realized_vol = returns.rolling(window=window).std()*(252 ** 0.5) 
        leverage = (target_vol/realized_vol).clip(upper=leverage_cap)

        # weights calculation
        weights_asset = leverage
        weights_cash = 1 - weights_asset

        # cash produce
        cash = []
        for i in range(len(price)):
            cash.append(price.iloc[0,0]*(1+risk_free_rate)**(float(i)/252.0))
        cash=pd.DataFrame(cash, index = price.index, columns=['cash'])

        weights = pd.merge(left=weights_asset, right= weights_cash, how= 'inner', left_index=True, right_index=True)
        weights.columns=[ticker,'cash']
        prices = pd.merge(left=price, right= cash, how= 'inner', left_index=True, right_index=True)
        prices.columns=[ticker,'cash']
        return weights, prices
