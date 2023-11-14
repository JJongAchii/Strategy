"""mlp strategy"""
import os
import sys
import argparse
import logging
from typing import Optional
from pydantic.main import BaseModel
from dateutil import parser
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from hive import db
from config import get_args, ALLOC_FOLDER
from datetime import datetime, timedelta
logger = logging.getLogger("sqlite")

args = get_args()
# args.date = "2023-07-03"
TODAY = parser.parse(args.date)
YESTERDAY = TODAY - timedelta(days=1)

class ThemeRotationStrategy:
    
    def __init__(self, strategy: str, portfolio_id: int, rolling_period: int, risk_period: int, theme_numbers: int) -> None: 
        """_summary_

        Args:
            strategy (bool): strategy code
            portfolio_id (int): portfolio_id
            rolling_perioed (int): portfolio update period
            risk_period (int): portfolio risk calculation size
            theme_numbers (int): number of ETFs in the theme 
        """
        self.strategy = strategy
        self.port_id = portfolio_id
        self.rolling_period = rolling_period
        self.risk_period = risk_period
        self.theme_numbers = theme_numbers
        
    def calc_weighted_return(self,df_price: pd.DataFrame, method: str) -> tuple[pd.DataFrame,pd.Series]:
        """_summary_

        calculate the weighted return of the theme portfolio
        
        Args:
            df_price (pd.DataFrame): price data of selected themes
            method (str): weighting method - equal weight / risk parity
            risk_period (int, optional): length of the dates required for standard deviation calculation.

        Returns:
            weight (pd.DataFrame): daily weighted return of each ticker
            weighted_return_sum (pd.Series): sum of daily weighted return of each ticker
        """

        if method == "riskparity":
            
            df_price = df_price.iloc[- self.risk_period:]
            df_return = df_price.pct_change()
            df_return.dropna(axis = 'columns',how = 'all',inplace=True)  
            # print(df_return.iloc[-self.rolling_period:])
            df_risk_vol = pd.DataFrame(1 / df_return.std()).fillna(0)
            weight = df_risk_vol / (df_risk_vol.sum())
            # print(weight)
            weight = weight.T
            weight.index = df_return.tail(1).index
            # print(weight)
            weight_ = pd.DataFrame(np.repeat(weight.values, len(df_return), axis=0))
            weight_.columns = weight.columns
            weight_.index = df_return.index
            # print(weight_)
            weighted_return = pd.DataFrame(df_return * weight_)
            
            weighted_return_sum = weighted_return.sum(axis=1) 
            
            weight = weight.iloc[-self.rolling_period:]  
            weighted_return = weighted_return.iloc[-self.rolling_period:]
            weighted_return_sum = weighted_return_sum.iloc[-self.rolling_period:]
            
        elif method == "equalweight":
            df_price = df_price.iloc[-self.rolling_period:]
            df_return = df_price.pct_change()
            df_return.dropna(axis = 'columns',how = 'all',inplace=True)
            # print(df_return)
            if len(df_return.columns) < 1:
                weight = pd.DataFrame()
                weight.index = [date]
                weight.index.name = 'trd_dt'
                weighted_return = pd.DataFrame()
                weighted_return.index = [date]
                weighted_return.index.name = 'trd_dt'
                weighted_return_sum = weighted_return.sum(axis=1)   
                
            else:
                
                weight = pd.DataFrame(df_return.copy().iloc[-1])
                weight = weight.T
                weight.iloc[0] = 1/len(df_return.columns)
                weight.index = df_return.tail(1).index 
                # print(weight)
                # weight *= (1/self.theme_numbers)
                weighted_return = pd.DataFrame(df_return * (1/len(df_return.columns)))
                # print(weighted_return)
                weighted_return_sum = weighted_return.sum(axis=1)
                # print(weighted_return_sum)
                # weighted_return = pd.DataFrame(weighted_return.iloc[-1])
                # # print(weighted_return)
        
        # print(weight)
        # print(weighted_return)
        # print(weighted_return_sum)
        return weight, weighted_return_sum

    @staticmethod
    def get_stk_id(df_weight: pd.DataFrame, stk_id_index: dict) -> pd.DataFrame:
        """_summary_

        Build a columnn based matrix with stk_id as columns
        
        Args:
            df_weight (pd.DataFrame): daily weight of each ticker of each theme (multi-level)
            stk_id_index (dict): a dictionary with ticker as keys and stk_id as values

        Returns:
            df_weight (pd.DataFrame): a column based matrix with stk_id as columns and daily weights of each ticker of the portfolio as row values
        """
        stk_id_lst = []
        df_weight_ticker = df_weight.copy()
        cols = [a[1] for a in df_weight_ticker.columns]
        
        for i in range(len(cols)):
            
            stk_id_lst.append(stk_id_index[cols[i]])
        
        df_weight_ticker.columns = stk_id_lst
        
        return df_weight_ticker

    def build_rolling_cumreturn_rank(self, df_return: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        1. Calculate the cumulative return of each theme portfolio
        2. Rank the theme portfolio based on the cumulative return 
         
        Args:
            df_return (pd.DataFrame): daily return of the theme 

        Returns:
            df_rank(pd.DataFrame): ranked cumulative return of each theme
        """
       
        date_lst = []
        theme_lst = []
        
        df_return_ = df_return.iloc[-self.rolling_period:]  
        # print(df_return_)
        df_cumreturn = (df_return_ + 1).cumprod()
        # print(df_cumreturn.T)
        df_rank = df_cumreturn.rank(1, ascending=False, method='first')
        df_rank_ = df_rank.iloc[-1][df_rank.iloc[-1].isin(range(1,self.theme_numbers+1))]
        date_lst.append(df_rank.index[-1])
        theme_lst.append(df_rank_.index.tolist())
        
        df_rank = pd.DataFrame({'date':date_lst,'theme':theme_lst})
        df_rank = df_rank.set_index('date')
        
        return df_rank
    
    def calc_portfolio_return(self, df_portfolio:pd.DataFrame, df_ticker_weight: pd.DataFrame, df_theme_return: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """_summary_

        build a column based matrix with stk_id as columns and daily weights of each ticker of the portfolio as row values
        
        Args:
            df_portfolio (pd.DataFrame): a matrix with a theme column showing the list of ranked themes 
            stk_id_dic (dict): a dictionary with tickers as keys and stk_id as values
            df_ticker_weight (pd.DataFrame): ticker weight for each theme portfolio 
            df_theme_return (pd.DataFrame): daily theme portfolio return

        Returns:
            df_port_members_weight (pd.DataFrame): a column based matrix with stk_id as columns and daily weights of each ticker of the portfolio as row values
        """
        
        df_port_members_return = pd.DataFrame()
        df_port_members_weight = pd.DataFrame()
        port_members_lst = []
        port_ret_lst = []
        port_members_weight = []
        # print(df_return)
        df_theme_return_ = df_theme_return.iloc[- self.rolling_period:]
        themes = df_portfolio['theme'].iloc[0]
        df_theme_return_ = df_theme_return_[themes]
        # print(df_return_)
        weight_cols = [a for a in df_ticker_weight.columns]
        weight_cols_ = [a for a in weight_cols if a[0] in themes]
        df_ticker_weight_ = df_ticker_weight[weight_cols_]
        theme_weight_cols = [a for a in df_ticker_weight_.columns]
        theme_weight_cols_ = list(set([a[0] for a in theme_weight_cols if a[0] in themes]))
        df_theme_return_ = df_theme_return_[theme_weight_cols_]
        # df_return_ = pd.DataFrame(df_return_.iloc[-1])
        # print(df_ticker_weight_)
        # print(df_ticker_weight_.sum(axis=1))
        df_ticker_weight_ = (df_ticker_weight_ / len(set(theme_weight_cols_)))
        # print(df_ticker_weight_)
        # print(df_ticker_weight_.sum(axis=1))
       
        # print(df_return_.sum(axis=1))
        # print(len(df_return_.columns))
        df_port_ret = (df_theme_return_.sum(axis=1) / len(df_theme_return_.columns))
        # print(df_port_ret)
        port_ret_lst.append(df_port_ret.tail(1))
        port_members_weight.append(df_ticker_weight_)
        # print(df_theme_weight_) 
        df_port_members_return = df_port_members_return._append(port_ret_lst)
        df_port_members_weight = df_port_members_weight._append(port_members_weight)
        # df_port_members_weight = self.get_stk_id(df_port_members_weight,stk_id_dic)
        
        return df_port_members_weight, df_port_members_return
    
    def build_tb_port_alloc_data(self, df_port_members_weight):
        """_summary_

        Update TbPortAlloc table 
        
        Args:
            df_port_members_weight (pd.DataFrame): a column based matrix with stk_id as columns and daily weights of each ticker of the portfolio as row values

       
        """
        members_weight_lst = []
        
        for i in range(len(df_port_members_weight)):
            
            df_weight = df_port_members_weight.iloc[i]   
            df_weight_ = pd.DataFrame(df_weight.dropna())
            
            rebal_dt = df_weight_.columns[0]
            df_weight_ = df_weight_.rename(columns={rebal_dt: 'weights'})
            df_weight_['rebal_dt'] = [rebal_dt] * len(df_weight_)

            df_weight_ = df_weight_.reset_index()
            df_weight_ = df_weight_.rename(columns={'index': 'stk_id'})
            
            df_weight_['port_id'] = [self.port_id] * len(df_weight_)
            df_weight_['shares'] = [0] * len(df_weight_)
            df_weight_['ap_weights'] = [0] * len(df_weight_)
            df_weight_ = df_weight_[['rebal_dt','port_id','stk_id','weights','shares','ap_weights']]
            
            members_weight_lst.append(df_weight_)
        
        df_port_alloc = pd.DataFrame()
        df_port_alloc = df_port_alloc.append(members_weight_lst)
        df_port_alloc.index = range(len(df_port_alloc))
        
        print(df_port_alloc)
        if args.database == "true":  
            
            try:
                db.TbPortAlloc.insert(df_port_alloc)
                
            except:
                try:
                    db.TbPortAlloc.update(df_port_alloc)
                except:
                    
                    db_alloc = db.get_alloc_weight(strategy=self.strategy)
                    db_alloc = db_alloc[db_alloc.rebal_dt == TODAY]
                    db_alloc = db_alloc[df_port_alloc.columns.tolist()]
                    db_alloc_stkid = db_alloc['stk_id'].tolist()
                    db.delete_asset_port_alloc(rebal_dt=TODAY, port_id=self.port_id, stk_id=db_alloc_stkid)
                    db.TbPortAlloc.insert(df_port_alloc)
                    
    
    @staticmethod
    def clean_weights(weights: pd.Series, decimals: int = 4, tot_weight=None) -> pd.Series:
        """Clean weights based on the number decimals and maintain the total of weights.

        Args:
            weights (pd.Series): asset weights.
            decimals (int, optional): number of decimals to be rounded for
                weight. Defaults to 4.

        Returns:
            pd.Series: clean asset weights.
        """
        # clip weight values by minimum and maximum.
        if not tot_weight:
            tot_weight = weights.sum().round(4)
        weights = weights.round(decimals=decimals)
        # repeat round and weight calculation.
        for _ in range(10):
            weights = weights / weights.sum() * tot_weight
            weights = weights.round(decimals=decimals)
            if weights.sum() == tot_weight:
                return weights
        # if residual remains after repeated rounding.
        # allocate the the residual weight on the max weight.
        residual = tot_weight - weights.sum()
        # !!! Error may occur when there are two max weights???
        weights.iloc[np.argmax(weights)] += np.round(residual, decimals=decimals)
        return weights

    @staticmethod
    def calc_adjusted_portfolio_weight(weights: pd.Series, w_max: float = 0.6, w_min: float = 0.02) -> dict:
            
        max_error = {}
        min_error = {}
        
        for key in weights.keys():

            if weights[key] > w_max:

                max_error[key] = weights[key]

            elif weights[key] < w_min:

                min_error[key] = weights[key]

        if len(max_error) > 0:
            
            final_weights_keys = weights.keys()
            final_weights_keys = [a for a in final_weights_keys if a not in max_error.keys()]
            
            max_left = weights.sum()
            diff = 0
            for err_key in max_error.keys():
                diff += max_error[err_key] - w_max
                weights[err_key] = w_max
                max_left -= max_error[err_key]
            
            for key in final_weights_keys:
                weights[key] += (diff * weights[key] / max_left)

        if len(min_error) > 0:
            
            final_weights_keys = weights.keys()
            final_weights_keys = [a for a in final_weights_keys if a not in min_error.keys()]
            
            min_left = weights.sum()
            diff = 0
            for err_key in min_error.keys():
                diff += abs(w_min - min_error[err_key]) 
                weights[err_key] = w_min
                min_left -= min_error[err_key]

            for key in final_weights_keys:
                weights[key] -= (diff * weights[key] / min_left)
        
        return pd.Series(weights)

def run_theme_allocation( date: datetime, excluded: list = ['Ageing Society Opportunities', 'Efficient Energy', 'Fintech Innovation','Future Education','Millennials','Robotics']) -> tuple[pd.DataFrame,pd.DataFrame]:
    """_summary_

    1. Calculate the portfolio return of each theme
    2. Rank the theme return based on a specific period and select the best n themes
    3. Build a combined portfolio using the selected n themes
    
    """
    args.database = "true"
    TODAY = date
    YESTERDAY = TODAY - timedelta(days=1)
    
    extra = dict(user=args.user, activity="theme_allocation", category="script")

    if TODAY.date() != db.get_start_trading_date(market="KR", asofdate=TODAY):
        logger.info(msg=f"[SKIP] Theme allocation. {TODAY:%Y-%m-%d}", extra=extra) 
        return

    logger.info(msg=f"[PASS] Start Theme allocation. {TODAY:%Y-%m-%d}", extra=extra)
    
    rotation = ThemeRotationStrategy('THEME_US',110, 20,252,5)   
    
    universe = db.load_universe(f"THEME_US")
    universe = universe.sort_values(by=['stk_id'])
    
    ticker_dic = {}
    stk_id_dic = {}
    
    for i in range(len(universe)):
        
        ticker_dic[universe['stk_id'].iloc[i]] = universe.index[i]

    universe_stk_id = [int(a) for a in ticker_dic.keys()]

    for i in range(len(universe)):
        
        stk_id_dic[universe.index[i]] = universe['stk_id'].iloc[i]

    prices = db.get_price(tickers=", ".join(universe.index.tolist())).loc[:YESTERDAY]
    tb_universe = db.get_universe_theme(universe_stk_id)
    db_theme = tb_universe[tb_universe['remark'].isnull() != True]
    db_theme = db_theme[~db_theme['remark'].isin(excluded)]
   
    effective_themes = sorted(db_theme['remark'].unique())
    df_theme_return = pd.DataFrame()
    df_ticker_weight = pd.DataFrame()
    empty_themes = []
    for i in range(len(effective_themes)):
        
        df_effective_theme_ = db_theme[db_theme['remark'] == effective_themes[i]]
        effective_tickers = []
        for j in range(len(df_effective_theme_)):
            
            effective_tickers.append(ticker_dic[df_effective_theme_['stk_id'].iloc[j]])
         
        df_effective_price = prices[effective_tickers].copy()
        df_effective_price_rolling_period = df_effective_price.iloc[-rotation.rolling_period:]
        df_effective_price_rolling_period.dropna(axis= 'columns',how='all',inplace=True)
        if len(df_effective_price_rolling_period.columns) < 1:
            empty_themes.append(effective_themes[i])
            continue
        weight, weighted_return = rotation.calc_weighted_return(df_effective_price,"equalweight")
        weight.columns = [(effective_themes[i],a) for a in weight.columns]
        if len(df_ticker_weight) < 1:
            df_ticker_weight = weight.copy()
        else:
            df_ticker_weight = pd.merge(df_ticker_weight,weight, on = 'trd_dt',how='outer',sort=True)
         
        weighted_return = weighted_return.reset_index()
        if len(df_theme_return) < 1:
            df_theme_return = weighted_return.copy()
        else:
            df_theme_return = pd.merge(df_theme_return,weighted_return, on = 'trd_dt',how='outer',sort=True) 
    
    effective_themes = [a for a in effective_themes if a not in empty_themes] 
    # print(df_ticker_weight) 
    # print(df_ticker_weight.sum(axis=1))
    # print(df_theme_return)
   
    df_theme_return = df_theme_return.set_index('trd_dt')
    df_theme_return.columns = effective_themes
    df_rank = rotation.build_rolling_cumreturn_rank(df_theme_return)
    df_rank.index = [date]
    # print(df_rank)
    df_port_members_weight,df_port_ret = rotation.calc_portfolio_return(df_rank, df_ticker_weight, df_theme_return)
    df_port_members_weight.index = [date]
    df_port_ret.index = [date]
    
    common_dates = df_rank.index.intersection(df_port_members_weight.index)
    df_rank = df_rank[df_rank.index.isin(common_dates)]
    df_port_members_weight = df_port_members_weight[df_port_members_weight.index.isin(common_dates)]
    
    df_port_members_weight_t = df_port_members_weight.T   
    df_port_members_weight_t_ = pd.DataFrame(df_port_members_weight_t).squeeze()
    df_port_members_weight_t = rotation.calc_adjusted_portfolio_weight(weights = df_port_members_weight_t_)
    
    df_port_members_weight_t = pd.DataFrame(df_port_members_weight_t).T
    df_port_members_weight_t_ = df_port_members_weight_t.squeeze()
    df_port_members_weight_t = rotation.clean_weights(weights = df_port_members_weight_t_, decimals=4)
   
    df_port_members_weight = pd.DataFrame(df_port_members_weight_t).T
    # print(df_port_members_weight.sum(axis=1))
    df_weight_stk_id = rotation.get_stk_id(df_port_members_weight,stk_id_dic)

    rotation.build_tb_port_alloc_data(df_weight_stk_id)
                     
    logger.info(msg=f"[PASS] End ThemeRotation allocation. {TODAY:%Y-%m-%d}")
    # print(df_port_members_weight)
    return df_port_members_weight, df_port_ret

def run_daily_theme_allocation(excluded: list = ['Ageing Society Opportunities', 'Efficient Energy', 'Fintech Innovation','Future Education','Millennials','Robotics']) -> tuple[pd.DataFrame,pd.DataFrame]:
    """_summary_

    1. Calculate the portfolio return of each theme
    2. Rank the theme return based on a specific period and select the best n themes
    3. Build a combined portfolio using the selected n themes
    
    """
    print(args.date)
    extra = dict(user=args.user, activity="theme_allocation", category="script")

    if TODAY.date() != db.get_start_trading_date(market="KR", asofdate=TODAY):
        logger.info(msg=f"[SKIP] Theme allocation. {TODAY:%Y-%m-%d}", extra=extra) 
        return

    logger.info(msg=f"[PASS] Start Theme allocation. {TODAY:%Y-%m-%d}", extra=extra)
    
    rotation = ThemeRotationStrategy('THEME_US',110, 20,252,5)   
    
    universe = db.load_universe(f"THEME_US")
    universe = universe.sort_values(by=['stk_id'])
    
    ticker_dic = {}
    stk_id_dic = {}
    
    for i in range(len(universe)):
        
        ticker_dic[universe['stk_id'].iloc[i]] = universe.index[i]

    universe_stk_id = [int(a) for a in ticker_dic.keys()]

    for i in range(len(universe)):
        
        stk_id_dic[universe.index[i]] = universe['stk_id'].iloc[i]

    prices = db.get_price(tickers=", ".join(universe.index.tolist())).loc[:YESTERDAY]
    tb_universe = db.get_universe_theme(universe_stk_id)
    db_theme = tb_universe[tb_universe['remark'].isnull() != True]
    db_theme = db_theme[~db_theme['remark'].isin(excluded)]
   
    effective_themes = sorted(db_theme['remark'].unique())
    df_theme_return = pd.DataFrame()
    df_ticker_weight = pd.DataFrame()
    empty_themes = []
    for i in range(len(effective_themes)):
        
        df_effective_theme_ = db_theme[db_theme['remark'] == effective_themes[i]]
        effective_tickers = []
        for j in range(len(df_effective_theme_)):
            
            effective_tickers.append(ticker_dic[df_effective_theme_['stk_id'].iloc[j]])
         
        df_effective_price = prices[effective_tickers].copy()
        df_effective_price_rolling_period = df_effective_price.iloc[-rotation.rolling_period:]
        df_effective_price_rolling_period.dropna(axis= 'columns',how='all',inplace=True)
        if len(df_effective_price_rolling_period.columns) < 1:
            empty_themes.append(effective_themes[i])
            continue
        weight, weighted_return = rotation.calc_weighted_return(df_effective_price,"equalweight")
        weight.columns = [(effective_themes[i],a) for a in weight.columns]
        if len(df_ticker_weight) < 1:
            df_ticker_weight = weight.copy()
        else:
            df_ticker_weight = pd.merge(df_ticker_weight,weight, on = 'trd_dt',how='outer',sort=True)
         
        weighted_return = weighted_return.reset_index()
        
        effective_themes_ = [a for a in effective_themes[:j] if a not in empty_themes] 
        if len(df_theme_return) < 1:
            df_theme_return = weighted_return.copy()
        else:
            df_theme_return = pd.merge(df_theme_return,weighted_return, on = 'trd_dt',how='outer',sort=True) 
          
            df_theme_return.columns = ['trd_dt'] + [*range(len(df_theme_return.columns)-1)]
        
        # print(df_theme_return)
    
    effective_themes = [a for a in effective_themes if a not in empty_themes] 
    # print(df_ticker_weight) 
    # print(df_ticker_weight.sum(axis=1))
    # print(df_theme_return)
   
    df_theme_return = df_theme_return.set_index('trd_dt')
    df_theme_return.columns = effective_themes
    df_rank = rotation.build_rolling_cumreturn_rank(df_theme_return)
    # df_rank.index = [date]
    print(df_rank)
    df_port_members_weight,df_port_ret = rotation.calc_portfolio_return(df_rank, df_ticker_weight, df_theme_return)
    # df_port_members_weight.index = [date]
    # df_port_ret.index = [date]
    
    common_dates = df_rank.index.intersection(df_port_members_weight.index)
    df_rank = df_rank[df_rank.index.isin(common_dates)]
    df_port_members_weight = df_port_members_weight[df_port_members_weight.index.isin(common_dates)]
    
    df_port_members_weight_t = df_port_members_weight.T   
    df_port_members_weight_t_ = pd.DataFrame(df_port_members_weight_t).squeeze()
    df_port_members_weight_t = rotation.calc_adjusted_portfolio_weight(weights = df_port_members_weight_t_)
    
    df_port_members_weight_t = pd.DataFrame(df_port_members_weight_t).T
    df_port_members_weight_t_ = df_port_members_weight_t.squeeze()
    df_port_members_weight_t = rotation.clean_weights(weights = df_port_members_weight_t_, decimals=4)
   
    df_port_members_weight = pd.DataFrame(df_port_members_weight_t).T
    # print(df_port_members_weight.sum(axis=1))
    df_weight_stk_id = rotation.get_stk_id(df_port_members_weight,stk_id_dic)

    rotation.build_tb_port_alloc_data(df_weight_stk_id)
                     
    logger.info(msg=f"[PASS] End ThemeRotation allocation. {TODAY:%Y-%m-%d}")
    print(df_port_members_weight)
    return df_port_members_weight, df_port_ret

if __name__ == "__main__":
    
    args = get_args()
    
    args.script = "theme_hist"
    
    if args.script == "theme_hist":
        try:
            from core.strategy import themestrategy
            df_port_ret = pd.DataFrame()
            port_ret_lst = []
            df_port_weight = pd.DataFrame()
            port_weight_lst = []
            date_lst = pd.date_range(start='2018-01-02', end='2023-07-03') 
            for i in range(len(date_lst)):
                # print(date_lst[i])
                if date_lst[i] == db.get_start_trading_date(market="KR", asofdate=date_lst[i]):
                    
                    port_weight, port_ret = themestrategy.run_theme_allocation(date_lst[i])
                    # print(port_ret)
                    port_ret = pd.DataFrame(port_ret.T)
                    port_ret.columns = ['daily_ret']
                    # print(port_ret)
                    # print(port_weight.columns)
                    # print(port_weight)
                    cols = [a[1] for a in port_weight.columns.tolist()]
                    port_weight.columns = cols
                    # print(port_weight)
                    port_weight = port_weight.sort_index(axis=1)
                    print(port_weight)
                    port_ret_lst.append(port_ret)
                    port_weight_lst.append(port_weight)
                    
                else:
                    port_weight = port_weight_lst[-1]
                
                    prices = db.get_price(tickers=", ".join(cols)).loc[:date_lst[i]]
                    # print(prices)   
                    df_ret = prices.pct_change()
                    # print(df_ret)
                    port_weight.index = [date_lst[i]]
                    port_weight.index.name = 'trd_dt'
                    
                    port_ret = port_weight * df_ret.tail(1)   
                    # print(df_port_ret)
                    port_ret = pd.DataFrame(port_ret.sum(axis=1))
                    port_ret.columns = ['daily_ret']
                    # print(port_ret)
                    # print(port_weight)   
                    port_ret_lst.append(port_ret)
                    port_weight_lst.append(port_weight) 
                    # print(port_weight_lst[-1])          
            
            df_port_weight = df_port_weight.append(port_weight_lst)
            print(df_port_weight)    
            # df_port_weight.to_excel('C:/Users/sukeh/Downloads/theme_rotation_riskparity_port_120day_tracking_5_themes.xlsx')
            df_port_weight = df_port_weight.drop_duplicates()
            print(df_port_weight)       
            # print(port_ret_lst)
            df_port_ret = df_port_ret.append(port_ret_lst)
            # print(df_port_ret)
            df_port_ret = df_port_ret.drop_duplicates()
            print(df_port_ret)
            df_port_ret.columns = ['daily_ret']
            df_port_ret['cum_ret'] = (df_port_ret['daily_ret'] + 1).cumprod()        
            print(df_port_ret)
            # df_port_ret.to_excel('C:/Users/sukeh/Downloads/theme_rotation_riskparity_120day_tracking_5_themes.xlsx')
        except Exception as error:
            extra_ = dict(user=args.user, activity=args.script, category="monitoring")
            
            logger.error(msg=f'[ERROR] {args.script}\n{error}', extra=extra_)
    
    elif args.script == "theme":
        # print(args.script)
        # print(args.date)
        from core.strategy import themestrategy
        themestrategy.run_daily_theme_allocation()
        
        
        
        