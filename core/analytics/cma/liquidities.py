import pandas as pd
import numpy as np
from datetime import datetime,date
import sys
import os
import logging
import warnings
warnings.filterwarnings('ignore')
from base import CapitalMarketAssumptions
sys.path.insert(0, os.path.join(os.path.abspath(__file__), "../../../.."))
from hive import db
logger = logging.getLogger("sqlite")
from config import get_args


class LiquiditiesCapitalMarketAssumptions(CapitalMarketAssumptions):
    
    def __init__(self, index: str, asofdate: str) -> None: 
        
        self.index: str = index
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
        
        macro_id = db.get_macro_id()
        macro_id_index = macro_id[(macro_id['ticker'] == self.index) | (macro_id['future_ticker'] == self.index)]['macro_id'].values[0]
        macro_data = macro_data[macro_data['macro_id'] == macro_id_index]
       
        df_data = macro_data[['trd_dt','macro_id','value','adj_value']]
        df_data['trd_dt'] = pd.to_datetime(df_data['trd_dt']) 
        df_data = df_data.set_index('trd_dt')
      
        logger.info(msg="[PASS] built liquidity macro data") 
        return df_data
    
    def get_monthly_data(self) -> pd.DataFrame:
        
        """_summary_

        convert and store daily macro data to monthly first data and monthly average data 
        This function is used when it is necessary to merge monthly first data and monthly average data using merge property.

        Returns:
            monthly_data (pd.DataFrame): monthly first bloomberg field data and monthly average bloomberg field data
        """
        data = self.data
        monthly_data = data.to_period(freq='M')
        monthly_first_data = monthly_data.groupby(by='trd_dt').first()[['macro_id','adj_value']]
        monthly_avg_data = monthly_data.groupby(by='trd_dt').mean()[['macro_id','adj_value']]
       
        monthly_data = monthly_first_data.merge(monthly_avg_data,on='trd_dt',how='inner',sort=True)
       
        monthly_data = monthly_data[['macro_id_x','adj_value_x','adj_value_y']]
        monthly_data.columns = ['macro_id','first','avg']
        
        return monthly_data
        

def run_liquidities_market_assumption():
    
    """
    _summary_
    
    run_liquidities_market_assumption estimates the Bloomberg field value using a regression method
    
    """
    lc = LiquiditiesCapitalMarketAssumptions('USGG3M Index',"19900101")
   
    monthly_data = lc.get_monthly_data()
    liquidity_predicted = lc.get_historical_residuals(monthly_data,'USGG3M Index',120,12)
    liquidity_predicted = liquidity_predicted.set_index('date')
    print(liquidity_predicted)