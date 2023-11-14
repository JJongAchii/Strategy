import pandas as pd
import numpy as np
from datetime import datetime,date
import sys
import os
import logging
import warnings
warnings.filterwarnings('ignore')
from base import CapitalMarketAssumptions
from core.analytics.cma.base import GetBBGFieldsValue
sys.path.insert(0, os.path.join(os.path.abspath(__file__), "../../../.."))
from hive import db
logger = logging.getLogger("sqlite")
from config import get_args

class EquitiesCapitalMarketAssumptions:
             
    def __init__(self,window : int = 250, min_length : int = 2500, startdate : str = '20130101', enddate : str = '20230101', index : str = 'russell1000',
    dic: dict = {'dvd_yld': 'EQY_DVD_YLD_IND', 'buyback_yld': 'STOCK_BUYBACK_HISTORY','spot_pe_ratio': 'PE_RATIO',\
        'long_term_pe_ratio': 'LONG_TERM_PRICE_EARNINGS_RATIO','earnings_growth': 'EPS_GROWTH','size': 'CUR_MKT_CAP'}, 
    universe : pd.DataFrame = pd.read_pickle("C:/Users/sukeh/Downloads/dneuro/data/russell1000_constituents_20000101_20240101.pkl"),
    exponent: int = 10
):      
        self.window = window
        self.min_length = min_length
        self.startdate = startdate
        self.enddate = enddate
        self.index = index
        
        self.bbgvalue = GetBBGFieldsValue(min_length,startdate,enddate,index,universe,dic)
        self.weight = self.bbgvalue.get_daily_index_component_weights()  
        self.exponent = exponent
        self.bbgvalue.build_bdh_data()
        self.long_term_pe_ratio = self.bbgvalue.factors['long_term_pe_ratio'].set_index('index')
        self.spot_pe_ratio =  self.bbgvalue.factors['spot_pe_ratio'].set_index('index')   
        self.earnings_growth = self.bbgvalue.factors['earnings_growth'].set_index('index')
        self.dvd_yld = self.bbgvalue.factors['dvd_yld'].set_index('index')
        self.size = self.bbgvalue.factors['size'].set_index('index')
        
        self.bbgvalue.build_bds_data()
        self.buyback = self.bbgvalue.factors['buyback_yld']
        self.buyback = self.bbgvalue.build_column_based_buyback_data(self.buyback)
        
        self.buyback_yld = self.bbgvalue.build_deflated_data(self.buyback,self.size,self.startdate,self.enddate,"M",12) 

        
    def calc_daily_index_component_weighted(self, df) -> pd.DataFrame:
        """_summary_

        calculate the daily index components of the building block components 
        
        Args:
            df (pd.DataFrame): column based building block components 

        Returns:
            df_index_components (pd.DataFrame): index components of the building block component's tickers
        """
        df_weight = self.weight
        df.replace([np.inf, -np.inf], 0, inplace=True)
        common_dates = df_weight.index.intersection(df.index)
        df = df.loc[common_dates]
        df_weight = df_weight.loc[common_dates]
        df_index_components = pd.DataFrame()
        arr = []
        for i in range(len(common_dates)):
        
            df_weight_date = df_weight.iloc[i]
            df_weight_date = pd.DataFrame(df_weight_date[df_weight_date.isnull() != True])
           
            df_date = pd.DataFrame(df.iloc[i])
            df_date = df_date[df_date.index.isin(df_weight_date.index)]
            df_weight_date = df_weight_date[df_weight_date.index.isin(df_date.index)]
            
            df_date = df_date.fillna(0)
            
            
            if sum(df_weight_date.values) != 1:               
                df_date = df_date * df_weight_date * (1/sum(df_weight_date.values))               
            else:
                df_date = df_date * df_weight_date
            
            arr.append(df_date.T)

        df_index_components = df_index_components.append(arr).fillna(0) 
        df_index_components.replace([np.inf, -np.inf], 0, inplace=True)
        
        return df_index_components
     
    def calc_n_year_average(self) -> float:
        """_summary_

        calculate the total yield by summing all building block components of equities capital market assumptions
        
        Returns:
            total_yld (float): total_yld = valuation_change_mean + earnings_growth_mean + buyback_yld_mean + dvd_yld_mean
        """
        startdate = self.startdate
        enddate = self.enddate
        long_term_pe_ratio = self.long_term_pe_ratio
        spot_pe_ratio = self.spot_pe_ratio
        earnings_growth = self.earnings_growth
        buyback_yld = self.buyback_yld
        dvd_yld = self.dvd_yld
        valuation_change = self.calc_valuation_change()
        
        valuation_change = valuation_change[(valuation_change.index >= datetime.strptime(startdate,'%Y%m%d').date()) & (valuation_change.index <= datetime.strptime(enddate,'%Y%m%d').date())]
        long_term_pe_ratio = long_term_pe_ratio[(long_term_pe_ratio.index >= datetime.strptime(startdate,'%Y%m%d').date()) & (long_term_pe_ratio.index <= datetime.strptime(enddate,'%Y%m%d').date()) ]
        spot_pe_ratio = spot_pe_ratio[(spot_pe_ratio.index >= datetime.strptime(startdate,'%Y%m%d').date()) & (spot_pe_ratio.index <= datetime.strptime(enddate,'%Y%m%d').date()) ]
        earnings_growth = earnings_growth[(earnings_growth.index >= datetime.strptime(startdate,'%Y%m%d').date()) & (earnings_growth.index <= datetime.strptime(enddate,'%Y%m%d').date()) ]
        buyback_yld = buyback_yld[(buyback_yld.index >= datetime.strptime(startdate,'%Y%m%d')) & (buyback_yld.index <= datetime.strptime(enddate,'%Y%m%d')) ]
        dvd_yld = dvd_yld[(dvd_yld.index >= datetime.strptime(startdate,'%Y%m%d').date()) & (dvd_yld.index <= datetime.strptime(enddate,'%Y%m%d').date()) ]
        
        valuation_change_ = self.calc_daily_index_component_weighted(valuation_change)
        # multiply 0.01 for the earnings growth Bloomberg data
        earnings_growth_ = 0.01 * self.calc_daily_index_component_weighted(earnings_growth)
        buyback_yld_ = self.calc_daily_index_component_weighted(buyback_yld)
        # multiply 0.01 for the dividend yield Bloomberg data
        dvd_yld_ = 0.01 * self.calc_daily_index_component_weighted(dvd_yld)
        
        crosssectional_valuation_change = valuation_change_.sum(axis=1)
        crosssectional_earnins_growth = earnings_growth_.sum(axis=1)
        crosssectional_buyback_yld = buyback_yld_.sum(axis=1)
        crosssectional_dvd_yld = dvd_yld_.sum(axis=1)
        
        valuation_change_mean = crosssectional_valuation_change.mean()
        earnings_growth_mean = crosssectional_earnins_growth.mean()
        buyback_yld_mean = crosssectional_buyback_yld.mean()
        dvd_yld_mean = crosssectional_dvd_yld.mean()
       
        total_yld = valuation_change_mean + earnings_growth_mean + buyback_yld_mean + dvd_yld_mean
        
        return total_yld
    
    def calc_valuation_change(self) -> pd.DataFrame:
        """_summary_
        
        valuation change = {((long-term mean of the P/E ratio) / (P/E current ratio))**(1/10)} - 1

        extreme dislocations in P/E (high or low versus the average) have a larger impact on estimated returns
        
        Returns:
            valuation_change(pd.DataFrame): column based matrix of single ticker valuation change  
        """
        long_term_pe_ratio = self.long_term_pe_ratio
        spot_pe_ratio = self.spot_pe_ratio
        
        itx = long_term_pe_ratio.index.intersection(spot_pe_ratio.index) \
        .intersection(self.weight.index)
        
        long_term_pe_ratio = long_term_pe_ratio.loc[itx]
        spot_pe_ratio = spot_pe_ratio.loc[itx]
        weight = self.weight.loc[itx]
        
        tickers = long_term_pe_ratio.T.index.intersection(spot_pe_ratio.T.index).tolist()
        long_term_pe_ratio = long_term_pe_ratio[tickers]
        spot_pe_ratio = spot_pe_ratio[tickers]
        
        long_term_pe_ratio_mean = long_term_pe_ratio.copy()
        spot_pe_ratio_mean = spot_pe_ratio.copy()
        
        for i in range(len(tickers)):
           
            long_term_pe_ratio_mean[tickers[i]] =  long_term_pe_ratio[tickers[i]].rolling(window = self.min_length,min_periods = int(self.min_length)).mean()           
            spot_pe_ratio_mean[tickers[i]] =  spot_pe_ratio[tickers[i]].rolling(window = self.min_length,min_periods = int(self.min_length)).mean() 
           
        valuation_change = (long_term_pe_ratio_mean / spot_pe_ratio_mean)**(1/self.exponent) - 1
        
        return valuation_change
    
def run_equities_market_assumption():
    
    
    """
    _summary_
    
    run_equities_market_assumption estimates the Bloomberg field value using a regression method
    
    """
    
    total_yld = EquitiesCapitalMarketAssumptions(window = 250, min_length  = 2500, startdate = '20130101', enddate  = '20230101', index = 'russell1000',
    dic = {'dvd_yld': 'EQY_DVD_YLD_IND', 'buyback_yld': 'STOCK_BUYBACK_HISTORY','spot_pe_ratio': 'PE_RATIO',\
        'long_term_pe_ratio': 'LONG_TERM_PRICE_EARNINGS_RATIO','earnings_growth': 'EPS_GROWTH','size': 'CUR_MKT_CAP'}, 
    universe  = pd.read_pickle("C:/Users/sukeh/Downloads/dneuro/data/russell1000_constituents_20000101_20240101.pkl"),
    exponent = 10).calc_n_year_average()
    
    print('equities total yield: ', total_yld)
        
 