import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from typing import Optional, Union, List
from sklearn import linear_model


sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from jjongdb import db
from reposit.strategy.strategy_report import *


def resample_data(price: pd.DataFrame, freq: str = "M", type: str = "head") -> pd.DataFrame:
    """resampling daily data"""
    if type == "head":
        res_price = price.groupby([price.index.year, price.index.month]).head(1)
    elif type == "tail":
        res_price = price.groupby([price.index.year, price.index.month]).tail(1)
    
    if freq == "Y":
        max_date_value = res_price.iloc[-1]
        if type == "head":
            res_price = res_price[res_price.index.month == 1]
        elif type == "tail":
            res_price = res_price[res_price.index.month == 12]
        res_price = res_price.append(max_date_value)
    return res_price


def volatility_1year(price: pd.DataFrame):
    """calculate 1 year volatility """
    vol = price.pct_change().iloc[-252:].std() * np.sqrt(252)
        
    return vol


def cal_monthly_momentum(price: pd.DataFrame):
    
    price_tail = resample_data(price=price, freq="M", type="tail")[-13:]
    
    if len(price_tail) < 13:
        return
    
    monthly_momentum = price_tail.iloc[-1].div(price_tail) - 1
    
    return monthly_momentum


def binary_from_momentum(momentum: pd.DataFrame):
    
    return momentum.applymap(lambda x: 1 if x > 0 else 0)


def absolute_momentum(price: pd.DataFrame):
    monthly_mmt = cal_monthly_momentum(price=price)

    if monthly_mmt is None:
        return
    abs_mmt = binary_from_momentum(momentum=monthly_mmt)[:-1]
    abs_mmt_score = abs_mmt.mean()

    return abs_mmt_score


def weighted_momentum(price: pd.DataFrame):
    monthly_mmt = cal_monthly_momentum(price=price)

    if monthly_mmt is None:
        return
    weighted_mmt = binary_from_momentum(momentum=monthly_mmt)[:-1]
    
    weighted_mmt = weighted_mmt.mul([x + 1 for x in range(len(weighted_mmt))], axis=0)

    weighted_mmt_score = weighted_mmt.mean()

    return weighted_mmt_score


def regression(
    dependent_y: pd.DataFrame,
    independent_x: pd.DataFrame,
    sample_weight: Optional[pd.Series] = None,
    method: str = "lasso",
    positive: bool = False,
    fit_intercept: bool = False,
) -> pd.DataFrame:

    if method.lower() == "linear":
        regression_model = linear_model.LinearRegression(
            positive=positive, fit_intercept=fit_intercept,
        )

    elif method.lower() == "lasso":
        regression_model = linear_model.LassoCV(
            positive=positive, fit_intercept=fit_intercept,
            random_state=255,
        )
    else:
        raise NotImplementedError(f"method {method} not implemnted.")

    itx = dependent_y.index.intersection(independent_x.index)
    dependent_y = dependent_y.loc[itx].pct_change().fillna(0)
    independent_x = independent_x.loc[itx].pct_change().fillna(0)

    results = []
    
    for y in dependent_y:
        result = pd.Series(
            data=regression_model.fit(
                X=independent_x,
                y=dependent_y[y],
                sample_weight=sample_weight,
            ).coef_,
            index=independent_x.columns,
            name=y,
        )

        result["score"] = regression_model.score(
                X=independent_x,
                y=dependent_y[y],
                sample_weight=sample_weight,
            )

        results.append(result)

    return pd.concat(results, axis=1).T


class Backtest:
    
    def __init__(
        self,
        strategy_name: str = None,
    ) -> None:
        
        self.strategy_name = strategy_name
    
    
    def universe(self, tickers: Union[str, List] = None):
        
        return tickers
        

    def data(
        self,
        tickers: Union[str, List] = None,
        source: str = "yf"
    ) -> pd.DataFrame:
        
        if source == "yf":
            data = yf.download(tickers=tickers)["Adj Close"]
        elif source == "db":
            pass
            
        return data
        
    
    def rebalance(
        self,
        price: pd.DataFrame,
        method: str = "eq",
        freq: str = "M",
        custom_weight: dict = None,
        offensive: List = None,
        defensive: List = None,
        start: ... = None,
        end: ... = None
    ) -> pd.DataFrame:
        """_summary_

        Args:
            method (str, optional): _description_. Defaults to "eq".
            freq (str, optional): _description_. Defaults to "M".
            weight (dict, optional): 
                using when method == "custom"
                ex) weight={"SPY": 0.6, "IEF":0.4}. Defaults to None.
            offensive (List, optional): 
                using when method == "VAA_agg"
                ex) offensive=["SPY", "EFA", "EEM", "AGG"]. Defaults to None.
            defensive (List, optional): 
                using when method == "VAA_agg"
                ex) defensive=["LQD", "IEF", "SHY"]. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
        
        price = price.loc[start:end].dropna()
        
        if method == "eq":
            """equal weights all assets"""
            weights = resample_data(price=price, freq=freq)
            weights[:] = 1 / len(price.columns)
            
        elif method == "custom":
            """defined weights each assets"""
            weights = resample_data(price=price, freq=freq)
            weights[:] = np.nan
            for key, value in custom_weight.items():
                weights[key] = value
                
        elif method == "target_vol":
            weights = TargetVol().simulate(price=price)

        elif method == "abs_mmt":
            weights = AbsoluteMomentum().simulate(price=price)

        elif method == "dual_mmt":
            weights = DualMomentum().simulate(price=price)

        elif method == "dual_mmt2":
            weights = DualMomentum2().simulate(price=price)

        elif method == "weighted_mmt":
            weights = WeightedMomentum().simulate(price=price)
        
        elif method == "meb_mmt":
            weights = MebFaberMomentum().simulate(price=price)
        
        elif method == "GTAA":
            
            weights = GTAA().simulate(price=price)
            
        elif method == "VAA_agg":

            weights = VAA().aggressive_vaa(
                price=price, 
                offensive=offensive, 
                defensive=defensive
            )
            
        return weights
            
            
    def result(
        self,
        price: pd.DataFrame,
        weight: pd.DataFrame,
        start: ... = None,
        end: ... = None,
    ) -> pd.DataFrame:
        
        book, nav = calculate_nav(
                        price=price, 
                        weight=weight, 
                        strategy_name=self.strategy_name,
                        start_date=start,
                        end_date=end        
                    )
        merge = pd.concat(nav.values(), axis=1)
        merge.columns = nav.keys()
        nav = merge.fillna(method='ffill')
        
        return nav


    def report(
        self,
        nav: pd.DataFrame,
        start: ... = None,
        end: ... = None
    ):
        nav_slice = nav.loc[start:end]
        nav_slice = nav_slice.pct_change().add(1).cumprod() * 1000
        
        result = result_metrics(nav=nav_slice)

        return result
    
    
class TargetVol:
    
    def simulate(self, price: pd.DataFrame):
        """simulate all time series"""
        
        previous_month = None
        price.index = pd.to_datetime(price.index)
        price = price.dropna()
        
        weights_df = pd.DataFrame()
        
        for trade_date in price.index:
            
            yesterday = trade_date - timedelta(days=1)
            
            price_slice = price[:yesterday]
        
            current_month = trade_date.month

            if previous_month is None or current_month != previous_month:
                
                vol = volatility_1year(price=price_slice)
                
                ratio = np.prod(vol) / vol
                weights = ratio / ratio.sum()
                
                weights = weights.reset_index()
                weights.columns = ["ticker", "weights"]
                weights["date"] = trade_date
                
                weights_df = pd.concat([weights_df, weights], axis=0)
                
            
            previous_month = current_month 
        
        return weights_df.pivot(index="date", columns="ticker", values="weights")


class AbsoluteMomentum:
    def simulate(self, price: pd.DataFrame):
        
        weights_df = pd.DataFrame()
        
        rebal_date = resample_data(price=price, freq="M", type="head").index
        
        # simulate historical date
        for rebal_date in rebal_date:
            yesterday = rebal_date - timedelta(days=1)
        
            price_slice = price[:yesterday]
            
            if not price_slice.empty:
                
                abs_mmt_score = absolute_momentum(price=price_slice)
                
                if abs_mmt_score is None:
                    continue
                weights = abs_mmt_score.div(abs_mmt_score.sum()).reset_index()
                
                weights.columns = ["ticker", "weights"]
                weights["rebal_date"] = rebal_date
                weights_df = pd.concat([weights_df, weights], axis=0)
                
        return weights_df.pivot(index="rebal_date", columns="ticker", values="weights")


class DualMomentum:
    def simulate(self, price: pd.DataFrame):
        
        weights_df = pd.DataFrame()
        
        rebal_date = resample_data(price=price, freq="M", type="head").index
        
        # simulate historical date
        for rebal_date in rebal_date:
            yesterday = rebal_date - timedelta(days=1)
        
            price_slice = price[:yesterday]
            
            if not price_slice.empty:
                
                abs_mmt_score = absolute_momentum(price=price_slice)
                
                if abs_mmt_score is None:
                    continue
                
                dual_mmt_score = abs_mmt_score.nlargest(4)
                
                weights = dual_mmt_score.div(dual_mmt_score.sum()).reset_index()
                
                weights.columns = ["ticker", "weights"]
                weights["rebal_date"] = rebal_date
                weights_df = pd.concat([weights_df, weights], axis=0)
                
        return weights_df.pivot(index="rebal_date", columns="ticker", values="weights")

def absolute_momentum2(price: pd.DataFrame):
    monthly_mmt = cal_monthly_momentum(price=price)

    if monthly_mmt is None:
        return
    abs_mmt = binary_from_momentum(momentum=monthly_mmt)[:-3]
    
    abs_mmt_score = abs_mmt.mean()

    return abs_mmt_score


class DualMomentum2:
    def simulate(self, price: pd.DataFrame):
        
        weights_df = pd.DataFrame()
        
        rebal_date = resample_data(price=price, freq="M", type="head").index
        
        # simulate historical date
        for rebal_date in rebal_date:
            yesterday = rebal_date - timedelta(days=1)
        
            price_slice = price[:yesterday]
            
            if not price_slice.empty:
                
                abs_mmt_score = absolute_momentum2(price=price_slice)
                
                if abs_mmt_score is None:
                    continue
                
                dual_mmt_score = abs_mmt_score.nlargest(4)
                
                weights = dual_mmt_score.div(dual_mmt_score.sum()).reset_index()
                
                weights.columns = ["ticker", "weights"]
                weights["rebal_date"] = rebal_date
                weights_df = pd.concat([weights_df, weights], axis=0)
                
        return weights_df.pivot(index="rebal_date", columns="ticker", values="weights")

class WeightedMomentum:
    def simulate(self, price: pd.DataFrame):
        
        weights_df = pd.DataFrame()
        
        rebal_date = resample_data(price=price, freq="M", type="head").index
        
        # simulate historical date
        for rebal_date in rebal_date:
            yesterday = rebal_date - timedelta(days=1)
        
            price_slice = price[:yesterday]
            
            if not price_slice.empty:
                
                weighted_mmt_score = weighted_momentum(price=price_slice)
                # print(abs_mmt_score)
                if weighted_mmt_score is None:
                    continue
                weights = weighted_mmt_score.div(weighted_mmt_score.sum()).reset_index()
                
                weights.columns = ["ticker", "weights"]
                weights["rebal_date"] = rebal_date
                weights_df = pd.concat([weights_df, weights], axis=0)
                
        return weights_df.pivot(index="rebal_date", columns="ticker", values="weights")


class MebFaberMomentum:
    def simulate(self, price: pd.DataFrame):
        
        weights_df = pd.DataFrame()
        
        rebal_date = resample_data(price=price, freq="M", type="head").index
        
        for rebal_date in rebal_date:
            yesterday = rebal_date - timedelta(days=1)
            
            price_slice = price[:yesterday]
            
            if not price_slice.empty:
                
                weights = resample_data(price=price_slice, freq="M", type="tail")
                
                sma3 = weights.rolling(window=3).mean()
                sma10 = weights.rolling(window=10).mean()
                
                signal = sma3 - sma10

                weights = signal.iloc[-1].apply(lambda x: 1 if x > 0 else 0)
                weights = weights / weights.sum()
                weights = weights.reset_index()
                weights.columns = ["ticker", "weights"]
                weights["rebal_date"] = rebal_date
                weights_df = pd.concat([weights_df, weights], axis=0)
                
        return weights_df.pivot(index="rebal_date", columns="ticker", values="weights")[9:]
                

class GTAA:
    
    def cal_return(self, price: pd.DataFrame) -> pd.DataFrame:
        """calculate cum returns 3months ~ 12months without previous month"""
        
        price = price.resample("M").last()
        dict_returns = {}
        
        for month in range(3, 13):
            cum_returns = price.pct_change(periods=month)
            
            dict_returns[month] = cum_returns.iloc[-2]
        
        month_returns = pd.DataFrame(dict_returns).T
        
        return month_returns

    def volatility_1year(self, price: pd.DataFrame):
        vol = price.pct_change().iloc[-252:].std() * np.sqrt(252)
            
        return vol
        
        # return price.aggregate(cal, axis=0)


    def cal(self, p: pd.Series) -> float:
        
        loss_rate_2p5 = np.percentile(p.pct_change().dropna(), 2.5)
        absolute_loss_rate = abs(loss_rate_2p5)
        annualization_adjustment = np.sqrt(250)
        result = absolute_loss_rate * annualization_adjustment
        
        return result


    def dual_momentum(self, price: pd.DataFrame) -> pd.Series:
        """calculate momentum score"""

        month_returns = self.cal_return(price=price)
        vol = volatility_1year(price=price)
        
        weight = 1 / vol
        
        abs_mom = month_returns[month_returns > 0]
        rel_mom = abs_mom.apply(lambda row: row.nlargest(4), axis=1)
        
        dual_mom = rel_mom.copy()
        for col in rel_mom.columns:
            
            dual_mom.loc[rel_mom[col].notna(), col] = weight[col]

        dual_mom = dual_mom.div(dual_mom.sum(axis=1), axis=0)
        dual_mom = dual_mom.sum() / 10
        
        return dual_mom


    def simulate(self, price: pd.DataFrame):
        """simulate all time series"""
        
        previous_month = None
        price.index = pd.to_datetime(price.index)
        
        weights_df = pd.DataFrame()
        
        for trade_date in price.index:
            
            yesterday = trade_date - timedelta(days=1)
        
            price_slice = price[:yesterday].dropna(axis=1, how="all")
            current_month = trade_date.month

            if previous_month is None or current_month != previous_month:
                
                null_ticker = price_slice.columns[price_slice.iloc[-252:].isna().any()]
                count_ticker = price_slice.columns.difference(null_ticker)
                
                if len(count_ticker) >= len(price.columns):
                    
                    weights = self.dual_momentum(price=price_slice)
                    weights = weights.reset_index()
                    weights.columns = ["ticker", "weights"]
                    weights["date"] = trade_date
                    
                    weights_df = pd.concat([weights_df, weights], axis=0)
            
            previous_month = current_month 
        
        return weights_df.pivot(index="date", columns="ticker", values="weights")    


class VAA:
    
    def cal_momentum_score(self, price: pd.DataFrame) -> pd.DataFrame:
        """calculate cum returns 3months ~ 12months without previous month"""
        
        price = resample_data(price=price, freq="M", type="tail")
        
        dict_score = {}
        
        for month in [1, 3, 6, 12]:
            cum_returns = price.pct_change(periods=month)
            
            dict_score[month] = cum_returns.iloc[-1] * 12 / month
        
        score = pd.DataFrame(dict_score).T
        
        return score.sum()
    
    def aggressive_vaa(
        self, 
        price: pd.DataFrame, 
        offensive: Optional[List] = None,
        defensive: Optional[List] = None
    ):

        previous_month = None
        price.index = pd.to_datetime(price.index)
        
        weights_df = pd.DataFrame()
        
        for trade_date in price.index:
            
            yesterday = trade_date - timedelta(days=1)
            
            price_slice = price[:yesterday].dropna(axis=1, how="all")
            current_month = trade_date.month

            if previous_month is None or current_month != previous_month:
                
                null_ticker = price_slice.columns[price_slice.iloc[-252:].isna().any()]
                count_ticker = price_slice.columns.difference(null_ticker)
                
                if len(count_ticker) >= len(price.columns):
                    
                    score = self.cal_momentum_score(price=price_slice)
                    all_positive = all(score[item] > 0 for item in offensive)
                    
                    if all_positive:
                        reb_asset = score[offensive].idxmax()
                    else:
                        reb_asset = score[defensive].idxmax()

                    weights = {"date": trade_date, "ticker": reb_asset, "weights": 1}
                    
                    weights_df = weights_df.append(weights, ignore_index=True)
                    
            previous_month = current_month 
        
        return weights_df.pivot(index="date", columns="ticker", values="weights")    

        
if __name__ == "__main__":
    
    # price = pd.read_csv("strategy.csv", index_col="Date")
    # weights = resample_date(price=price, freq="Y")
    
    # bt = Backtest(universe=["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY"], source="yf")
    # bt.rebalance(method="VAA_agg", offensive=["SPY", "EFA", "EEM", "AGG"], defensive=["LQD", "IEF", "SHY"])
    # bt.report()
    
    bt = Backtest()
    universe = bt.universe(tickers=["SPY", "GLD", "TLT"])
    price = bt.data(tickers=universe, source="yf")
    weights = bt.rebalance(price=price, method="dual_mmt2")
    weights.to_clipboard()