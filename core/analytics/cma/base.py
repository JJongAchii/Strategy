from xbbg import blp
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Optional
from datetime import datetime,date
import os
import sys
import logging
from sklearn import linear_model
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.abspath(__file__), "../../../.."))
from hive import db

from config import get_args

logger = logging.getLogger("sqlite")
args = get_args()


class BBGQuery:
    
    def GetBDHData(universe:list, fld_name:str,startdate:str,enddate:str) -> pd.DataFrame:
        
        """_summary_

        Get bloomberg bdh type data (Bloomberg Data History)
        
        Args:
            universe (list): ticker pool
            fld_name (str): bloomberg field name
            startdate (str): data request startdate
            enddate (str): data request enddate
            
        Returns:
            df(pd.DataFrame): bloomberg bdh type data
        """
        arr = []
        df = pd.DataFrame()
        for i in range(len(universe)):
            
            df_ticker = blp.bdh(universe[i],fld_name,startdate,enddate)            
            df_ticker_ = pd.DataFrame(df_ticker)
           
            df_ticker_ = df_ticker_.reset_index()
           
            try:
                df_ticker_.columns = ['index',universe[i]]
              
            except:
                continue   
            
            if i < 1:
                df = df_ticker_.copy()
            else:
                df = pd.merge(df, df_ticker_,how='outer',on='index',sort=True)    
                   
        return df
    
    def GetBDSData(universe, fld_name:str) -> pd.DataFrame:
        
        """_summary_

        Get bloomberg bds type data (large data sets/bulk data)
        
        Args:
            universe (list): ticker pool
            fld_name (str): bloomberg field name
            
        Returns:
            df(pd.DataFrame): bloomberg bds type data
        """
        arr = []
        df = pd.DataFrame()
        
        for i in range(len(universe)):
                
            arr.append(blp.bds(universe[i],fld_name))
        
        df = df.append(arr)
             
        return df  
    
class SetBBGFields:
    
    pass
       

class SetEquitiesBBGFields(SetBBGFields):
    
    def __init__(self,factors:dict) -> None:
        
        self.factors = factors
  
class SetBBGFieldsInput:
    
    base_path = 'C:/Users/sukeh/Downloads/dneuro/data/' # str(Path.home() / 'Downloads/dneuro/data/') 
    today = pd.Timestamp(datetime.today().date())
    today_str = today.strftime('%Y%m%d')
 
class GetBBGFieldsValue(SetBBGFieldsInput):   
   
    def get_index_universe(self,universe) -> list:
        
        """_summary_

        Build historical index components list
         
        Returns:
            universe_lst(list): list of historical daily index components
        """
        path_ = self.base_path + self.index + '_historical_index_components.pkl'
        if os.path.isfile(path_):
            df_universe_lst = pd.read_pickle(path_)
            universe_lst = df_universe_lst['index_components'].tolist()
            return universe_lst
        else:
            universe_lst = []
            for i in range(len(universe)): 
            
                universe_lst = universe_lst + universe['index_constituents'].iloc[i]
                # print(i)
            universe_lst = sorted(list(set(universe_lst)))
            df_universe_lst = pd.DataFrame({'index_components':universe_lst})
            df_universe_lst.to_pickle(self.base_path + self.index + '_historical_index_components.pkl')
            return universe_lst
       
    def __init__(self, lookback_period:int,startdate:str,enddate:str,index:str,universe: pd.DataFrame,factors: dict ) -> None: 
        
        super(GetBBGFieldsValue,self).__init__()
        
        self.lookback_period = lookback_period
        self.startdate = startdate
        self.enddate = enddate
        self.index = index
        self.universe = universe
        self.universe_lst = self.get_index_universe(universe)
        self.factors = factors
                    
    
    def set_eqt_bbg_fields(self) -> SetEquitiesBBGFields:
        
        """_summary_
        
        Set factors for SetEquitiesBBGFields
        
        Returns:
            eqtsetbbgflds(SetEquitiesBBGFields): SetEquitiesBBGFields type variable
        """
        eqtsetbbgflds = SetEquitiesBBGFields(self.factors)
        return eqtsetbbgflds
        
    def set_bbg_fields_input(self) -> SetBBGFieldsInput:
        
        """_summary_
        
        Set BBGFieldsInput
        
        Returns:
            bbginput(SetBBGFieldsInput): base_path, today, today_str
        """
        bbginput = SetBBGFieldsInput()
        return bbginput
        
    @staticmethod
    def get_members_weight(df_index: pd.DataFrame, df_marcap: pd.DataFrame)-> pd.DataFrame: 
        
        """_summary_

        Build a marketcap weighted index components matrix (pd.DataFrame)
        
        Args:
            df_index (pd.DataFrame): index component data
            df_marcap (pd.DataFrame): marketcap data
            
        Returns:
            df_weights(pd.DataFrame): marketcap weighted index components
        """
        df_weights = pd.DataFrame()
        for i in range(len(df_index)): 
      
            universe = df_index['index_constituents'].iloc[i]
            tickers = df_marcap.iloc[i].index.tolist()
            tickers_ = [a for a in tickers if a in universe]
            df_marcap_date = df_marcap.iloc[i][df_marcap.iloc[i].index.isin(tickers_)]
            marcap_sum = df_marcap_date.sum()
            df_marcap_date_ = df_marcap_date / marcap_sum
            df_marcap_date_ = pd.DataFrame(df_marcap_date_)
            df_marcap_date_ = df_marcap_date_.reset_index()
            if len(df_weights) < 1:
                df_weights = df_marcap_date_.copy()    
            else:
                if len(df_marcap_date_) > 0:
                    df_weights = pd.merge(df_weights,df_marcap_date_,how='outer',on='index',sort=True)
        
        df_weights = df_weights.set_index('index')
        df_weights = df_weights.T
       
        return df_weights
    
    def get_daily_index_component_weights(self) -> pd.DataFrame:
        
        """_summary_

        Check and verify if a daily index components matrix holds valid index components 
        
        Returns:
            df_weights(pd.DataFrame): valid daily index components matrix (pd.DataFrame)
        """
        bbginput = self.set_bbg_fields_input()
        
        universe = self.universe
        base_path = bbginput.base_path
        index = self.index
        startdate = self.startdate
        enddate = self.enddate
        universe_lst = self.universe_lst
       
        path = base_path + index + '_historical_marcap.pkl'
        if os.path.isfile(path):
            
            path2 = base_path + index + '_historical_daily_index_component_weights.pkl'
            if os.path.isfile(path2):
                df_weights = pd.read_pickle(path2)
                return df_weights
                
            else:
                df_marcap = pd.read_pickle(path)
                df_marcap_ = df_marcap.set_index('index')
                df_marcap_.index.name = 'date'
                universe_ = universe.set_index('date')
                common_dates = df_marcap_.index.intersection(universe_.index)
                df_universe = universe_[universe_.index.isin(common_dates)]
                df_marcap_ = df_marcap_[df_marcap_.index.isin(common_dates)]
                df_weights = self.get_members_weight(df_universe, df_marcap_)
                path = base_path + index + '_historical_daily_index_component_weights.pkl'
                df_weights.to_pickle(path)
                
                return df_weights
        else:
            df_marcap = BBGQuery.GetBDHData(universe_lst, "CUR_MKT_CAP",startdate,enddate)
            df_marcap.to_pickle(path)
            path2 = base_path + index + '_historical_daily_index_component_weights.pkl'
            if os.path.isfile(path2):
                df_weights = pd.read_pickle(path2)
                return df_weights
            else:
                df_marcap_ = df_marcap.set_index('index')
                df_marcap_.index.name = 'date'
                universe_ = universe.set_index('date')
                common_dates = df_marcap_.index.intersection(universe_.index)
                df_universe = universe_[universe_.index.isin(common_dates)]
                df_marcap_ = df_marcap_[df_marcap_.index.isin(common_dates)]
                
                df_weights = self.get_members_weight(df_universe, df_marcap_)
                path = base_path + index + '_historical_daily_index_component_weights.pkl'
                df_weights.to_pickle(path)
                
                return df_weights
            
            
    def build_bdh_data(self) -> None:
        
        """
        _summary_
        
        Check if Bloomberg data exists for the following field name data: dvd_yld, spot_pe_ratio, long_term_pe_ratio, earnings_growth, size
        If there is no such data, build such data using GetBDHData
        
        """
        eqtsetbbgflds = self.set_eqt_bbg_fields()
        bbginput = self.set_bbg_fields_input()
    
        for key in eqtsetbbgflds.factors.keys():
            
            if key == "dvd_yld":
                
                path = bbginput.base_path + self.index + '_historical_dvd_yld.pkl'
                if os.path.isfile(path):
                    self.factors[key] = pd.read_pickle(path)
                    continue
                else:
                    df_dvd_yld = BBGQuery.GetBDHData(self.universe_lst,self.factors[key],self.startdate,self.enddate)
                    df_dvd_yld.to_pickle(path)
                    self.factors[key] = df_dvd_yld
                    
            elif key == "spot_pe_ratio":
                
                path = bbginput.base_path + self.index + '_historical_spot_pe_ratio.pkl'   
                if os.path.isfile(path):
                    self.factors[key] = pd.read_pickle(path)
                    continue
                else:
                    df_spot_pe_ratio = BBGQuery.GetBDHData(self.universe_lst,self.factors[key],self.startdate,self.enddate)
                    self.factors[key] = df_spot_pe_ratio
                    df_spot_pe_ratio.to_pickle(path)
                    
            elif key == "long_term_pe_ratio":
                
                path = bbginput.base_path + self.index + '_historical_long_term_pe_ratio.pkl'   
                if os.path.isfile(path):
                    self.factors[key] = pd.read_pickle(path)
                    continue
                else:
                    df_long_term_pe_ratio = BBGQuery.GetBDHData(self.universe_lst,self.factors[key],self.startdate,self.enddate)
                    self.factors[key] = df_long_term_pe_ratio
                    df_long_term_pe_ratio.to_pickle(path)
            
            elif key == "earnings_growth": 
                
                path = bbginput.base_path + self.index + '_historical_eps_growth.pkl'   
                if os.path.isfile(path):
                    self.factors[key] = pd.read_pickle(path)
                    continue
                else:
                    df_earnings_growth = BBGQuery.GetBDHData(self.universe_lst,eqtsetbbgflds.factors[key],self.startdate,self.enddate)
                    self.factors[key] = df_earnings_growth
                    df_earnings_growth.to_pickle(path)
                    
            elif key == "size": 
                
                path = bbginput.base_path + self.index + '_historical_marcap.pkl'   
                if os.path.isfile(path):
                    self.factors[key] = pd.read_pickle(path)
                    continue
                else:
                    df_size = BBGQuery.GetBDHData(self.universe_lst,self.factors[key],self.startdate,self.enddate)
                    self.factors[key] = df_size
                    df_size.to_pickle(path)

        
    def build_rolling_data(self, df: pd.DataFrame, groupby: str, freq:str, period:int) -> pd.DataFrame:
        
        """_summary_

        change the update frequency of the original data from date to month
        
        Args:
            df (pd.DataFrame): Bloomberg field data
            groupby (str): string type sign for dataframe group operation
            freq (str): string type date abbreviation: M for month, D for day
            period (int): size of rolling period
            
        Returns:
            df_grouped (pd.DataFrame): Bloomberg field data after sum/mean groupby operation 
        """
        if freq == 'M':
            
            dates = df.index.tolist()
            dates = [datetime.strptime(str(a.year) + str(a.month).zfill(2),'%Y%m') for a in dates]
            
            df['year_month'] = dates
            df_grouped_first = df.groupby('year_month').head(1)   
            print(df_grouped_first.index)
            grouped_dates =  df_grouped_first.index.tolist() 
            if groupby == "sum":
                df_grouped = df.groupby('year_month').sum()
              
                df_grouped = df_grouped.rolling(period).sum().fillna(0)
               
            elif groupby == "mean":
                df_grouped = df.groupby('year_month').mean()
                df_grouped = df_grouped.rolling(period).mean().fillna(0)
            
            df_grouped.index.name = 'index'
            
            return df_grouped
        
        
    def build_deflated_data(self,df:pd.DataFrame, df_deflator:pd.DataFrame, startdate:str, enddate:str, freq:str, period: int) -> pd.DataFrame:
        

        """_summary_

        deflate the original data using a deflator if the data cannot be compared crosssectionally 
        
        Args:
            df (pd.DataFrame): Bloomberg field data
            df_deflator (pd.DataFrame): deflator data
            startdate (str): deflation startdate
            enddate (str): deflation enddate
            freq (str): date type abbreviation
            period (int): size of data used for groupby operation
            
        Returns:
            df_grouped (pd.DataFrame): deflated and grouped data
        """
        
        df = df[df.index >= datetime.strptime(startdate,"%Y%m%d").date()]
        df_deflator = df_deflator[df_deflator.index >= datetime.strptime(startdate,"%Y%m%d").date()]
          
        tickers = df.columns.tolist()
        deflator_tickers = df_deflator.columns.tolist()
        
        common_tickers = sorted(list(set(tickers).intersection(set(deflator_tickers))))
        
        df = df[common_tickers]
        df_deflator = df_deflator[common_tickers]
        
        common_dates = df.index.intersection(df_deflator.index)
        df = df.loc[common_dates]
        df_deflator = df_deflator.loc[common_dates]
        
        if freq == 'D':
        
            df = df/df_deflator
            
            return df
        
        if freq == 'M':
            
            df_grouped = self.build_rolling_data(df,"sum",freq,period)
            
            deflator_dates = df_deflator.index.tolist()
            deflator_dates = [datetime.strptime(str(a.year) + str(a.month).zfill(2),'%Y%m') for a in deflator_dates]
            
            df_deflator['year_month'] = deflator_dates  
                  
            df_deflator_grouped = df_deflator.groupby('year_month').head(1)
            
            df_deflator_grouped = df_deflator_grouped.set_index('year_month')
            df_deflator_grouped = df_deflator_grouped.rolling(period).mean().fillna(0)
            
            
            df_deflator_grouped.index.name = 'index'
            
            df_grouped = df_grouped/df_deflator_grouped
           
            return df_grouped
                
    def build_column_based_buyback_data(self,df) -> pd.DataFrame:
        
        """_summary_

        build_column_based_buyback_data builds a matrix with buyback_data for each ticker.
        If the length of buyback data is different between each ticker, the function combines such tickers including all missing data. 
        
        Args:
            df (pd.DataFrame): buyback data
        Returns:
        
            df_buyback (pd.DataFrame): column based buyback data
        """
        tickers = df.index.unique().tolist()
        
        df_buyback = pd.DataFrame()
        for i in range(len(tickers)):
            
            df_ticker = df[df.index == tickers[i]]
            df_ticker = df_ticker.sort_values(by=['buyback_date'])
            buyback_dates = sorted(df_ticker['buyback_date'].unique())
            
            df_common_dates = pd.DataFrame()
            arr = []
            for j in range(len(buyback_dates)):
                
                df_ticker_ = df_ticker[df_ticker['buyback_date'] == buyback_dates[j]]
                
                buyback_cumsum_value = df_ticker_['buyback_value'].sum()
                df_ticker_ = df_ticker_[['buyback_date','buyback_value']].iloc[-1]
                df_ticker_['buyback_value'] = buyback_cumsum_value
                arr.append(df_ticker_)
          
            df_common_dates = pd.DataFrame(arr)        
            df_common_dates.columns = ['buyback_date',tickers[i]]           
           
            
            if i == 0:
                df_buyback = df_common_dates.copy()
            else:
                df_buyback = pd.merge(df_buyback, df_common_dates,on='buyback_date',how='outer',sort=True).fillna(0)

        df_buyback = df_buyback.set_index('buyback_date')
      
        return df_buyback
            
    def build_bds_data(self) -> None:
        
        """_summary_
        
        BDS (Bloomberg Data Set) is for large data sets/ bulk data.
        build_bda_data checks if buyback_yield data exists.
        If there is no buyback_yield data, the function requests buyback_yield data using Bloomberg API.
        """
        eqtsetbbgflds = self.set_eqt_bbg_fields()
        bbginput = self.set_bbg_fields_input()
        
        
        for key in eqtsetbbgflds.factors.keys():
            
            if key == "buyback_yld":
            
                path = bbginput.base_path + self.index + '_historical_buyback_yld.pkl'
                 
                if os.path.isfile(path):
                    df_buyback = pd.read_pickle(path)
             
                    self.factors[key] = df_buyback 
                    continue
                else:
                    df_buyback = BBGQuery.GetBDSData(self.universe_lst,self.factors[key])
                    
                    self.factors[key] = df_buyback
                    df_buyback.to_pickle(path)

                   
class CapitalMarketAssumptions:
    
    @staticmethod
    def get_historical_residuals(data:pd.DataFrame, index:str, period:int, forecastperiod:int) -> pd.DataFrame: 

        """_summary_

        Estimate the future value of Bloomberg field data
        
        Args:
            data (pd.DataFrame): Bloomberg field data
            index (str): Bloomberg index ticker
            period (int): size of data to be trained 
            forecastperiod (int): forecast interval size
        
        Returns:
            df_predicted (pd.DataFrame): estimate of Bloomberg field data
        """
        date_lst = []
        intercept_lst = []
        coef_lst = []
        reg_score_lst = []
        residuals_lst = []
        factor_coef_lst = []
        df_residuals = pd.DataFrame()
        df_coef = pd.DataFrame()
        predicted_lst = []
        df_predicted = pd.DataFrame()
        
        for j in range(0,len(data)-period,forecastperiod):

            date_index = data.index[j+period]
            
            df_train_x = data['avg'].iloc[j:j+period].fillna(method='ffill')
            df_train_y = data['first'].iloc[j:j+period].fillna(method='ffill')

            df_train_x = df_train_x.fillna(0)
            df_train_y = df_train_y.fillna(0)
            
            df_train_x = df_train_x.replace(np.inf, 0)
            df_train_x = df_train_x.replace(-np.inf, 0)
            
            df_train_y = df_train_y.replace(np.inf, 0)
            df_train_y = df_train_y.replace(-np.inf, 0)
            
            date_pred = data.index[j+period:j+period+forecastperiod].tolist()

            X, y = df_train_x.fillna(method='ffill').values.reshape(-1,1),df_train_y.fillna(method='ffill')

            df_test_X = data['avg'].iloc[j+period:j+period+forecastperiod].fillna(method='ffill')
            actual_y = data['first'].iloc[j+period:j+period+forecastperiod].fillna(method='ffill')
                  
            df_test_X = df_test_X.fillna(0)
            actual_y = actual_y.fillna(0)
            
            df_test_X = df_test_X.replace(np.inf, 0)
            df_test_X = df_test_X.replace(-np.inf, 0)
            
            actual_y = actual_y.replace(np.inf, 0)
            actual_y = actual_y.replace(-np.inf, 0)
            
            df_test_X = df_test_X.values.reshape(-1,1)
            
            regression_model = linear_model.LinearRegression(
                positive=False, fit_intercept=False
            )

            try:
                regression_model.fit(
                    X=X,
                    y=y,
                )
            except Exception as e:
                print(e)
                continue
            
            predicted = regression_model.predict(df_test_X)
            residuals = (actual_y - predicted).tolist()
            date_lst.append(date_index)
            intercept_lst.append(regression_model.intercept_)
            coef_lst.append(regression_model.coef_.tolist())
            reg_score_lst.append(regression_model.score(X, y))
            
            df_res = pd.DataFrame({'date': date_pred, index : residuals})
            df_explained = pd.DataFrame({'date': date_pred, index : predicted})
            residuals_lst.append(df_res)
            predicted_lst.append(df_explained)
          
        
        if len(residuals_lst) > 0:
            df_residuals = df_residuals.append(residuals_lst)
        
        if len(predicted_lst) > 0:
            df_predicted = df_predicted.append(predicted_lst)

        if len(factor_coef_lst) > 0:
            df_coef = df_coef.append(factor_coef_lst)
        
        return df_predicted