import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Optional
from pydantic import BaseModel
from dateutil import parser

sys.path.insert(0, os.path.join(os.path.abspath(__file__), "../../.."))
from hive import db
from config import get_args
from core.factor.factor_analysis import (
    exposure, 
    excess_performance, 
    risk_weighted_performance, 
    exposures_implied_performance
)

logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)

logger.info(f"running factor application script {TODAY:%Y-%m-%d}")


class FactorLensSettings(BaseModel):
    """_summary_
    assign bloomberg ticker to pre-defined factors.
    Args:
        BaseModel (_type_): _description_
    """
    rate: str = "H04792US"  
    equity: str = "MXCXDMHR"
    uscredit: str = "LUACTRUU"
    eucredit: str = "H02549US" 
    usjunk: str = "LF98TRUU"
    eujunk: str = "LP01TRUH"
    commodity: str = "BCOMTR"
    localequity: str = "SPXT"
    shortvol: str = "PUT"
    localinflation: str = "BCIT5T"
    currency: str = "DXY"
    emergingequity: str = "M1EF"
    emergingbond: str = "EMUSTRUU"
    developedequity: str = "M1WD"
    developedbond: str = "LEGATRUU"
    momentum: str = "M1WD000$"
    value: str = "M1WD000V"
    growth: str = "M1WD000G"
    lowvol: str = "M1WDMVOL"
    smallcap: str = "M1WDSC"
    quality: str = "M1WDQU"


class FactorLens:
    """_summary_
     
    get raw macro data from TB_MACRO_DATA table and macro id from TB_MACRO table.
    build a raw macro data metrix.
    calculate style factors and macro factors from raw macro data.
   
    """
    
    def __init__(self) -> None: 
        """_summary_
        
            build a data frame with historical tb_lens data and check the latest tb_lens update date for each factor 
        
        """
        self.rate: Optional[pd.Series(dtype=float)] = None
        self.equity: Optional[pd.Series(dtype=float)] = None
        self.credit: Optional[pd.Series(dtype=float)] = None
        self.commodity: Optional[pd.Series(dtype=float)] = None
        self.emerging: Optional[pd.Series(dtype=float)] = None
        self.currency: Optional[pd.Series(dtype=float)] = None
        self.localequity: Optional[pd.Series(dtype=float)] = None
        self.localinflation: Optional[pd.Series(dtype=float)] = None
        self.shortvol: Optional[pd.Series(dtype=float)] = None
        self.momentum: Optional[pd.Series(dtype=float)] = None
        self.value: Optional[pd.Series(dtype=float)] = None
        self.growth: Optional[pd.Series(dtype=float)] = None
        self.smallcap: Optional[pd.Series(dtype=float)] = None
        self.lowvol: Optional[pd.Series(dtype=float)] = None
        self.factor_last_date: dict = {}
        price_factor = db.get_lens(TODAY)
        
        self.data = self.get_macro_data()
        data = self.data.copy()
        if len(price_factor) < 1:
        
            self.factor_last_date['rate'] = data['rate'].dropna().index[0] 
            self.factor_last_date['equity'] = data['equity'].dropna().index[0] 
            self.factor_last_date['credit'] = max(data['uscredit'].dropna().index[0],data['eucredit'].dropna().index[0],data['usjunk'].dropna().index[0],data['eujunk'].dropna().index[0])
            self.factor_last_date['commodity'] = data['commodity'].dropna().index[0] 
            self.factor_last_date['emerging'] = max(data['emergingequity'].dropna().index[0],data['emergingequity'].dropna().index[0])
            self.factor_last_date['currency'] = data['currency'].dropna().index[0] 
            self.factor_last_date['localequity'] = data['localequity'].dropna().index[0] 
            self.factor_last_date['localinflation'] = data['localinflation'].dropna().index[0] 
            self.factor_last_date['shortvol'] = data['shortvol'].dropna().index[0] 
            self.factor_last_date['momentum'] = data['momentum'].dropna().index[0] 
            self.factor_last_date['value'] = data['value'].dropna().index[0]
            self.factor_last_date['growth'] = data['growth'].dropna().index[0]
            self.factor_last_date['smallcap'] = data['smallcap'].dropna().index[0] 
            self.factor_last_date['lowvol'] = data['lowvol'].dropna().index[0] 
            
        for i in range(len(price_factor.columns)):
            
            price_factor_ = price_factor.copy()
            price_factor_ = price_factor_.iloc[:,i].dropna()
            self.factor_last_date[price_factor.columns[i]] = price_factor_.index[-1]
            
        self.factors_lst = list()
        self.process()

    def get_macro_data(self) -> pd.DataFrame:
        """_summary_
        
        get raw macro data from TB_MACRO_DATA table and macro id from TB_MACRO table.
        build a raw macro data metrix with 21-column raw macro price.
        
        """
        
        factors = []
        tickers = []
        
        factor_lens = FactorLensSettings().dict()
        for name, ticker in factor_lens.items():

            factors.append(name)
            tickers.append(ticker)
        
        macro_data_ticker = db.get_tb_macro_data()
        tickers_ = [a + ' Index' for a in tickers]
        macro_data_ticker = macro_data_ticker[(macro_data_ticker.ticker.isin(tickers_)) | (macro_data_ticker.future_ticker.isin(tickers_))]
        
        df_data = pd.DataFrame()
        for i in range(len(tickers)):
            
            df_ticker = macro_data_ticker[macro_data_ticker['ticker'] == tickers_[i]]
            
            if len(df_ticker) > 0:
                df_ticker = df_ticker[['trd_dt','adj_value']]
                df_ticker.columns = ['trd_dt',factors[i]]
            else:
                continue
            
            if len(df_data) < 1:
                df_data = df_ticker.copy()
            else:
                df_data = pd.merge(df_data,df_ticker,on='trd_dt',how='outer',sort=True)
        
        for i in range(len(tickers)):
            
            df_ticker = macro_data_ticker[macro_data_ticker['future_ticker'] == tickers_[i]]
            
            if len(df_ticker) > 0:
                df_ticker = df_ticker[['trd_dt','adj_value']]
                df_ticker.columns = ['trd_dt',factors[i]]
            else:
                continue
            
            if len(df_data) < 1:
                df_data = df_ticker.copy()
            else:
                df_data = pd.merge(df_data,df_ticker,on='trd_dt',how='outer',sort=True)
        
        df_data['trd_dt'] = pd.to_datetime(df_data['trd_dt'])      
        df_data = df_data.set_index('trd_dt') 
        
        logger.info(msg="[PASS] built column based macro data") 
        return df_data
        
    
    def get_factor_value(self, factor_val: pd.Series, factor_name: str) -> None:
        """_summary_
        assign factor value to factors_lst, a class attribute 
        Args:
            factor_val (pd.Series): calculated factor value 
            factor_name (str): factor name
        """
        factor = pd.DataFrame(factor_val.copy())
        factor.insert(0,'factor',[factor_name] * len(factor))
        factor = factor.reset_index()
        factor.columns = ['trd_dt','factor','value']
        self.factors_lst.append(factor)
    
    def calc_style_factors(
        self, 
        factors = ["momentum", "value", "growth", "smallcap", "lowvol"]
    ) -> None: 
        """_summary_
        calculate equity style factors from raw macro data.
        Args:
            factors (list, optional): Defaults to ["momentum","value","growth","smallcap","lowvol"].
        """
        for i in range(len(factors)):
            
            self.factor = excess_performance(
            self.data[factors[i]].dropna(), self.data.developedequity.dropna()
            )
            self.factor.name = factors[i]
            self.get_factor_value(self.factor,factors[i])
        
        logger.info(msg="[PASS] style factor update") 
        
    def calc_macro_factors(self) -> None:
        """_summary_
        
        calculate macro factors from raw macro data.
        calculate 9 macro factors (rate, equity, credit, commodity, emerging, localinflation, localequity,shortvol, currency) by default.       
        
        raw data: equity, rate
        risk_weighted_performance: credit, emerging
        residualization: commodity, localinflation, localequity, shortvol, currency
        
        """
        self.rate = self.data.rate.dropna()
        self.get_factor_value(self.rate,"rate")
        
        self.equity = self.data.equity.dropna()
        self.get_factor_value(self.equity,"equity")
    
        credit_performance = risk_weighted_performance(
            self.data[["uscredit", "eucredit", "usjunk", "eujunk"]].dropna()
        )
        
        self.credit = self.residualization(
            price=credit_performance,
            price_factor=pd.concat([self.rate, self.equity], axis=1).dropna(),
        ).squeeze()
        self.credit.name = "credit"
        self.get_factor_value(self.credit,"credit")
    
        self.commodity = self.residualization(
            price=self.data.commodity,
            price_factor=pd.concat([self.rate, self.equity], axis=1).dropna(),
        ) 
        self.commodity.name = "commodity"
        self.get_factor_value(self.commodity,"commodity")
        
        core_macro = pd.concat(
            objs=[self.rate, self.equity, self.credit, self.commodity], axis=1
        ).dropna()

        em_equity = excess_performance(
            self.data.emergingequity.dropna(), self.data.developedequity.dropna()
        )

        em_bond = excess_performance(
            self.data.emergingbond.dropna(), self.data.developedbond.dropna()
        )

        emerging = (
            risk_weighted_performance(pd.concat([em_equity, em_bond], axis=1).dropna())
            .dropna()
            .loc[core_macro.index[0] :]
        )

        self.emerging = self.residualization(
            
            price=emerging, price_factor=core_macro
        ).squeeze()
        self.emerging.name = "emerging"
        self.get_factor_value(self.emerging,"emerging")
        
        self.localinflation = self.residualization(
            price=self.data.localinflation.dropna().loc[core_macro.index[0] :],
            price_factor=core_macro,
        )

        self.localinflation.name = "localinflation"
        self.get_factor_value(self.localinflation,"localinflation")
        
        self.localequity = self.residualization(
            price=self.data.localequity.dropna().loc[self.equity.dropna().index[0] :],
            price_factor=self.equity.to_frame(),
        )
        self.localequity.name = "localequity"
        self.get_factor_value(self.localequity,"localequity")
        
        self.shortvol = self.residualization(
            price=self.data.shortvol.dropna().loc[self.equity.index[0] :],
            price_factor=self.equity.to_frame(),
        )
        self.shortvol.name = "shortvol"
        self.get_factor_value(self.shortvol,"shortvol")
        
        ccy = self.data.currency.resample("D").last().loc[core_macro.index]

        self.currency = self.residualization(price=ccy, price_factor=core_macro)
        self.currency.name = "currency"
        self.get_factor_value(self.currency,"currency")
        
        logger.info(msg="[PASS] macro factor update")
            
    def process(self) -> pd.DataFrame():
        """_summary_
       
        call calc_style_factors function and calc_macro_factors faunction
        build a factor dataframe using a class attribute, factors_lst, and insert the calculated factor data to TB_Lens table.
        
        Returns: df_factors_update (pd.DataFrame): 14-column macro factor
        
        """
        self.calc_style_factors(factors = ["momentum","value","growth","smallcap","lowvol"])
        self.calc_macro_factors()
        
        df_factors = pd.DataFrame()
        df_factors = df_factors._append(self.factors_lst)

        df_factors_update = pd.DataFrame()
        factor_lens = sorted(df_factors['factor'].unique())
        for i in range(len(factor_lens)):
            
            df_factor = df_factors[df_factors['factor'] == factor_lens[i]]
            df_factor = df_factor[df_factor['trd_dt'] > self.factor_last_date[factor_lens[i]]]
            df_factors_update = df_factors_update._append(df_factor)
        
        if args.database == "true":
            db.TbLens.insert(df_factors_update)
        
        return df_factors_update
    

    @staticmethod
    def residualization(
        price: pd.Series,
        price_factor: pd.DataFrame,
        window: int = 252 * 3,
        smoothing_window: int = 5,
        **kwargs,
    ) -> pd.Series:

        """_summary_
        calculate core factor removed factor cumulative return
        
        Args:
            price (pd.Series): Non-core factor price data
            price_factor (pd.DataFrame): core factor price data
            window (int): rolling window for exposure function
            smoothing_window (int): rolling window for mean beta calculation
            
        Returns: excess performance (pd.Series): cumulative return of non-core factor multiplied by the beta
             
        """
        itx = price.index.intersection(price_factor.index)

        price = price.loc[itx]
        price_factor = price_factor.loc[itx]

        pri_return_1 = price.pct_change().fillna(0)
        pri_return_2 = price_factor.pct_change().fillna(0)

        betas = exposure(
            dependents=pri_return_1, independents=pri_return_2, window=window, **kwargs
        )

        if smoothing_window is not None:
            betas = betas.rolling(window=smoothing_window, min_periods=0).mean()

        perf_exposure = exposures_implied_performance(
            exposures=betas, price_factor=price_factor
        )

        return excess_performance(price, perf_exposure)

if __name__ == "__main__":
    
    factorlens = FactorLens()
