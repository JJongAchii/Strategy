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


class FixedIncomeCapitalMarketAssumptions(CapitalMarketAssumptions):
    
    def __init__(self, long_term_change_period: int, roll_period: int = 2,valuation_change_maturity: int = 2) -> None: 
        
        self.long_term_change_period = long_term_change_period
        self.roll_period = roll_period
        self.valuation_change_maturity = valuation_change_maturity
        pass
    
    def get_monthly_data(self, data: pd.DataFrame) -> pd.DataFrame:
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
        monthly_data.columns = ['first','avg']
        
        return monthly_data
    
    def estimate_long_term_mean_pe_ratio(self,):
        """_summary_

        estimate the ending yield (forward) using the USGG10YR index and US CORP AAA 10YR spread
        
        To calculate the ending (estimated) yield, we examine how the current (starting) yield curve could move over time as a result of changes \
        in Treasury interest rates and in the credit spreads over US Treasury interest rates.
        
        Returns:
            df_ending_yield (pd.DataFrame): current yield curve + changes in Treasury interest rates + changes in credit spreads over US Treasuries

        """
        df_interest_rates = pd.read_pickle('C:/Users/sukeh/Downloads/usgg10yr_index.pkl')
        df_credit_spreads = pd.read_pickle('C:/Users/sukeh/Downloads/us_corp_aaa_10yr_spread_index.pkl')
        
        common_dates = df_interest_rates.index.intersection(df_credit_spreads.index)
        
        df_interest_rates = df_interest_rates[df_interest_rates.index.isin(common_dates)]
        df_credit_spreads = df_credit_spreads[df_credit_spreads.index.isin(common_dates)]
       
        ending_yield = df_interest_rates.values.flatten() + (df_interest_rates.shift(-self.long_term_change_period).values.flatten() / df_interest_rates.values.flatten()) \
                                + (df_credit_spreads.shift(-self.long_term_change_period).values.flatten() / df_credit_spreads.values.flatten())
      
        ending_yield = [a for a in ending_yield if np.isnan(a) != True]
       
        idx = df_interest_rates.index.tolist()
        idx = idx[self.long_term_change_period:]

        df_ending_yield = pd.DataFrame({'trd_dt': idx, 'ending_yield' : ending_yield})
        df_ending_yield['trd_dt'] = pd.to_datetime(df_ending_yield['trd_dt'])
        df_ending_yield = df_ending_yield.set_index('trd_dt')

        return df_ending_yield

    def estimate_yield_total_yield(self):
        """_summary_

        Calculate current roll return using USGG2yr Index and USGG3yr Index
        
        Interest rate on current yield curve at: 
        6-year maturity = 2.69%
        5-year maturity = 2.56%

        Interest rate on future yield curve at:
        6-year maturity = 3.35%
        5-year maturity = 3.26%

        Current roll return = -5 x (2.56% - 2.69%) = 0.65%
        Future roll return = -5 x (3.26% - 3.35%) = 0.45%
        Roll return = (0.65% + 0.45%)/2 = 0.55%
        
        Returns:
            df_current_roll_return (pd.DataFrame): current roll return
        """
        df_usgg2yr = pd.read_pickle('C:/Users/sukeh/Downloads/usgg2yr_index.pkl')
        df_usgg3yr = pd.read_pickle('C:/Users/sukeh/Downloads/usgg3yr_index.pkl')

        common_dates = df_usgg2yr.index.intersection(df_usgg3yr.index)
        
        df_usgg2yr = df_usgg2yr[df_usgg2yr.index.isin(common_dates)]
        df_usgg3yr = df_usgg3yr[df_usgg3yr.index.isin(common_dates)]

        current_roll_return = -self.roll_period * (df_usgg2yr.values.flatten() - df_usgg3yr.values.flatten())
        df_current_roll_return = pd.DataFrame({'trd_dt': df_usgg2yr.index,  'roll_return': current_roll_return})
        df_current_roll_return['trd_dt'] = pd.to_datetime(df_current_roll_return['trd_dt'])
        df_current_roll_return = df_current_roll_return.set_index('trd_dt')

        return df_current_roll_return
    
    def estimate_valuation_change(self) -> pd.DataFrame:
        """_summary_

        calculate the valuation change estimate 
        Maturity = 6 years
        Current yield = 3.12%
        Ending yield = 4.06%

        Valuation change = [1 - 6 x (4.06% - 3.12%)]** 1/10 - 1 = -0.59%
        
        Returns:
            
            df_val_change (pd.DataFrame): _description_
        """
        df_2yr = pd.read_pickle('C:/Users/sukeh/Downloads/usgg2yr_index.pkl')
        df_3yr = pd.read_pickle('C:/Users/sukeh/Downloads/usgg3yr_index.pkl')
        df_5yr = pd.read_pickle('C:/Users/sukeh/Downloads/usgg5yr_index.pkl')
        
        common_dates = df_2yr.index.intersection(df_3yr.index).intersection(df_5yr.index)
        
        df_2yr = df_2yr[df_2yr.index.isin(common_dates)]
        df_3yr = df_3yr[df_3yr.index.isin(common_dates)]
        df_5yr = df_5yr[df_5yr.index.isin(common_dates)]
     
        diff = df_5yr.values.flatten() - df_3yr.values.flatten()
        ending_yield = diff + df_2yr.values.flatten() 

        current_end_diff = ending_yield - df_2yr.values.flatten()
        before_pow = [1-(self.valuation_change_maturity * (current_end_diff))][0]
        before_pow = [0 if a < 0 else a for a in before_pow]
        valuation_change = np.power(before_pow,(1/10)) - 1
        df_val_change = pd.DataFrame({'trd_dt': df_2yr.index, 'valuation_change': valuation_change})
        df_val_change['trd_dt'] = pd.to_datetime(df_val_change['trd_dt'])
        df_val_change = df_val_change.set_index('trd_dt')
        df_val_change = df_val_change.fillna(0)

        return df_val_change

def run_fixedincome_market_assumption():
    
    """
    _summary_
    
    run_fixedincome_market_assumption estimates the Bloomberg field value using a regression method
    
    """
    
    fi = FixedIncomeCapitalMarketAssumptions(250,2,2)
    long_term_mean_pe_ratio = fi.estimate_long_term_mean_pe_ratio()
    total_yield = fi.estimate_yield_total_yield()
    valuation_change = fi.estimate_valuation_change()

    monthly_long_term_mean_pe_ratio = fi.get_monthly_data(long_term_mean_pe_ratio)
    monthly_total_yield = fi.get_monthly_data(total_yield)
    monthly_valuation_change = fi.get_monthly_data(valuation_change)

    monthly_long_term_mean_pe_ratio_predicted = fi.get_historical_residuals(monthly_long_term_mean_pe_ratio,'USGG2YR',120,12)
    monthly_long_term_mean_pe_ratio_predicted = monthly_long_term_mean_pe_ratio_predicted.set_index('date')
    print('monthly_long_term_mean_pe_ratio_predicted: ', monthly_long_term_mean_pe_ratio_predicted)
    monthly_total_yield_predicted = fi.get_historical_residuals(monthly_total_yield,'USGG2YR',120,12)
    monthly_total_yield_predicted = monthly_total_yield_predicted.set_index('date')
    print('monthly_total_yield_predicted: ', monthly_total_yield_predicted)
    monthly_valuation_change_predicted = fi.get_historical_residuals(monthly_valuation_change,'USGG2YR',120,12)
    monthly_valuation_change_predicted = monthly_valuation_change_predicted.set_index('date')
    print('monthly_valuation_change_predicted: ', monthly_valuation_change_predicted)
    
    fixedincome_total_yield_predicted = monthly_long_term_mean_pe_ratio_predicted + monthly_total_yield_predicted + monthly_valuation_change_predicted
    print('fixedincome_total_yield_predicted: ', fixedincome_total_yield_predicted)