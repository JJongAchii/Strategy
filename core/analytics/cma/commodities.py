import pandas as pd
import numpy as np
from datetime import datetime,date
import sys
import os
import logging
import warnings
warnings.filterwarnings('ignore')
from base import CapitalMarketAssumptions
from core.analytics.cma.liquidities import LiquiditiesCapitalMarketAssumptions
sys.path.insert(0, os.path.join(os.path.abspath(__file__), "../../../.."))
from hive import db
logger = logging.getLogger("sqlite")
from config import get_args


class CommoditiesCapitalMarketAssumptions(CapitalMarketAssumptions):
    
    def __init__(self, spot_index: str, roll_index: str, inflation_index: str, asofdate: str) -> None:
        
        self.spot_index: str = spot_index
        self.roll_index: str = roll_index
        self.inflation_index: str = inflation_index
        self.asofdate = datetime.strptime(asofdate,'%Y%m%d')
        self.data: pd.DataFrame = self.get_macro_data()

    def get_macro_data(self) -> pd.DataFrame:
        
        """_summary_

        Get raw macro data from TB_MACRO_DATA table and macro id from TB_MACRO table.
        Build a raw macro data metrix.
        
        Returns:
            df_data (pd.DataFrame): macro data from tb_macro_data table 
        """
        
        macro_data = db.get_tb_macro_data() 
        
        macro_data = macro_data[(macro_data['ticker'] == (self.spot_index)) | (macro_data['future_ticker'] == (self.spot_index)) |
                                (macro_data['ticker'] == (self.roll_index)) | (macro_data['future_ticker'] == (self.roll_index)) |
                                (macro_data['ticker'] == (self.inflation_index)) | (macro_data['future_ticker'] == (self.inflation_index))
                                ]
        
        
        df_data = macro_data[['trd_dt','ticker', 'future_ticker', 'value','adj_value']]
        df_data['trd_dt'] = pd.to_datetime(df_data['trd_dt']) 
        df_data = df_data.set_index('trd_dt')
        
        logger.info(msg="[PASS] built commodity macro data") 
        return df_data
    
    def get_monthly_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        convert and store daily macro data to monthly first data and monthly average data 
        This function is used when it is necessary to merge monthly first data and monthly average data using pandas concat property.
        
        Args:
            data (pd.DataFrame): Bloomberg field data

        Returns:
            monthly_data (pd.DataFrame): monthly first data and monthly avg data
        """
        monthly_data = data.to_period(freq='M')
        monthly_first_data = monthly_data.groupby(by='trd_dt').first()
        monthly_avg_data = monthly_data.groupby(by='trd_dt').mean()
        monthly_data = pd.concat([monthly_first_data,monthly_avg_data],axis=1)
        monthly_data.columns = ['first','avg']
    
        return monthly_data
    
    def get_monthly_db_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        """_summary_
        
        convert and store daily macro data to monthly first data and monthly average data 
        This function is used when it is necessary to merge monthly first data and monthly average data using merge property.
        
        Args:
            data (pd.DataFrame): Bloomberg field data

        Returns:
            monthly_data (pd.DataFrame): monthly first data and monthly avg data
        """
        monthly_data = data.to_period(freq='M')
        
        monthly_first_data = monthly_data.groupby(by='trd_dt').first()
        monthly_avg_data = monthly_data.groupby(by='trd_dt').mean()
       
        monthly_data = monthly_first_data.merge(monthly_avg_data,on='trd_dt',how='inner',sort=True)
        monthly_data.columns = ['ticker','future_ticker','first_value','first_adj_value','avg_value', 'avg_adj_value']
        monthly_data = monthly_data[['ticker','first_adj_value','avg_adj_value']]
        monthly_data.columns = ['ticker','first','avg']
        monthly_data = monthly_data[['first','avg']]
        
        return monthly_data
    
    def calc_roll_yield(self) -> pd.DataFrame:
        
        """_summary_

        Calculate the difference between daily change of rolled data and daily change of spot data
        
        Returns:
            df_spread (pd.DataFrame): difference between daily change of rolled data and daily change of spot data
        """
        spot_index = self.spot_index
        roll_index = self.roll_index
        df = self.data
      
        df_spot = df[(df['ticker'] == spot_index) | (df['future_ticker'] == spot_index)]
        df_roll = df[(df['ticker'] == roll_index) | (df['future_ticker'] == roll_index)]
        
        df_spot = df_spot.sort_values(by=['trd_dt'])
        df_roll = df_roll.sort_values(by=['trd_dt'])
        
        common_dates = df_spot.index.intersection(df_roll.index)
        
        df_spot = df_spot[df_spot.index.isin(common_dates)]
        df_roll = df_roll[df_roll.index.isin(common_dates)]
        
        df_spot['pct_change'] = df_spot['adj_value'].pct_change()
        df_roll['pct_change'] = df_roll['adj_value'].pct_change()
        df_spread = df_roll['pct_change'] - df_spot['pct_change']   
       
        return df_spread
    
    def calc_spot_return(self) -> pd.DataFrame:
        
        """_summary_
        
        Calculate monthly spread and monthly avg spread data
        
        Use the long-term historical average of real spot monthly returns, and adjust that for expected inflation
        over a 10-year time horizon.
        
        Returns:
            df_spread (pd.DataFrame): _description_
        """
        df_monthly_spot = self.get_monthly_db_data(self.data[self.data['ticker'] == self.spot_index])
        df_monthly_inflation = self.get_monthly_db_data(self.data[self.data['ticker'] == self.inflation_index])
    
        df_monthly_spot['first_pct_change'] = df_monthly_spot['first'].pct_change()
        df_monthly_inflation['first_pct_change'] = df_monthly_inflation['first'].pct_change()
        df_first_spread = df_monthly_spot['first_pct_change'] - df_monthly_inflation['first_pct_change']
        df_monthly_spot['avg_pct_change'] = df_monthly_spot['avg'].pct_change()
        df_monthly_inflation['avg_pct_change'] = df_monthly_inflation['avg'].pct_change()
        df_avg_spread = df_monthly_spot['avg_pct_change'] - df_monthly_inflation['avg_pct_change']
        df_spread = pd.concat([df_first_spread,df_avg_spread],axis=1)
        df_spread.columns = ['first','avg']
     
        return df_spread

def run_commodities_market_assumption():
    
    """
    _summary_
    
    run_commodities_market_assumption estimates the Bloomberg field value using a regression method
    
    """
    
    cm = CommoditiesCapitalMarketAssumptions('EMUSTRUU Index','LEGATRUU Index', 'T5YIFR Index', '19900101')
    spread = cm.calc_roll_yield()
    
    monthly_spread = cm.get_monthly_data(spread)
    
    monthly_spot = cm.calc_spot_return()
    
    monthly_spread_predicted = cm.get_historical_residuals(monthly_spread, 'GLOBAL COMMODITY',120,12)
    monthly_spot_predicted = cm.get_historical_residuals(monthly_spot,'GLOBAL COMMODITY',120,12)

    lc = LiquiditiesCapitalMarketAssumptions('USGG3M Index',"19900101")
   
    monthly_data = lc.get_monthly_data()
    liquidity_predicted = lc.get_historical_residuals(monthly_data,'USGG3M Index',120,12)

    monthly_collateral_predicted = liquidity_predicted.copy()     
    
    monthly_spread_predicted = monthly_spread_predicted.set_index('date')
    monthly_spot_predicted = monthly_spot_predicted.set_index('date')
    monthly_collateral_predicted = monthly_collateral_predicted.set_index('date')
    
    common_dates = monthly_spread_predicted.index.intersection(monthly_spot_predicted.index).intersection(monthly_collateral_predicted.index)
    
    monthly_spread_predicted = monthly_spread_predicted[monthly_spread_predicted.index.isin(common_dates)]
    monthly_spot_predicted = monthly_spot_predicted[monthly_spot_predicted.index.isin(common_dates)]
    monthly_collateral_predicted = monthly_collateral_predicted[monthly_collateral_predicted.index.isin(common_dates)]
    
    monthly_collateral_predicted = monthly_collateral_predicted.rename(columns={'USGG3M Index': 'GLOBAL COMMODITY'})
    
    commodities_total_yield = monthly_spread_predicted + monthly_spot_predicted + monthly_collateral_predicted
    
    print('commodities_total_yield: ', commodities_total_yield)