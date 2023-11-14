import os
import sys
from typing import Optional
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn import linear_model
sys.path.insert(0, os.path.join(os.path.abspath(__file__), "../.."))
from hive import db
# from core.analytics import metrics
# from core.analytics.metrics import Utility
from core.factor_analysis import exposure, excess_performance,excess_performance2, risk_weighted_performance, exposures_implied_performance
from datetime import datetime, date

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
    Returns:
        _type_: _description_
    """
    asofdate: Optional[date] = datetime.strptime('20000101','%Y%m%d')
    factors = []
    tickers = []
    
    factor_lens = FactorLensSettings().dict()
    for name, ticker in factor_lens.items():

        factors.append(name)
        tickers.append(ticker)
    
    macro_data = db.get_macro_data(asofdate)
    macro_id = db.get_macro_id()
    
    dic_id = {}
    for i in range(len(factors)):
        dic_id[factors[i]] = macro_id[macro_id['factor'] == factors[i]]['macro_id'].values[0]
    
    macro_name_lst = []
    macro_id_lst = []
    for name, id in dic_id.items():

        macro_name_lst.append(name)
        macro_id_lst.append(id)
        
    df_data = pd.DataFrame()
    for i in range(len(macro_id_lst)):
        
        df_ticker = macro_data[macro_data['macro_id'] == macro_id_lst[i]]
        df_ticker = df_ticker[['trd_dt','adj_value']]
        df_ticker.columns = ['trd_dt',macro_name_lst[i]]
        
        if i == 0:
            df_data = df_ticker.copy()
        else:
            df_data = pd.merge(df_data,df_ticker,on='trd_dt',how='outer',sort=True)
    
    df_data = df_data.set_index('trd_dt') 
    data = df_data.copy()
    # print(df_data)
    
    df_factors = pd.DataFrame()
    factors_lst = []
    
    def __init__(self,) -> None: 
        
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
        self.process()

    def get_factor_value(self, factor_val: pd.Series, factor_name: str) -> None:
        """_summary_
        assign factor value to factors_lst, a class attribute 
        Args:
            factor_val (pd.Series): calculated factor value 
            factor_name (str): factor name
        """
        factor = pd.DataFrame(factor_val.copy())
        factor.insert(0,'lens_id',[factor_name] * len(factor))
        factor = factor.reset_index()
        factor.columns = ['trd_dt','lens_id','value']
        self.factors_lst.append(factor)
    
    def calc_style_factors(self,factors = ["momentum","value","growth","smallcap","lowvol"]) -> None: 
        """_summary_
        calculate equity style factors from raw macro data.
        Args:
            factors (list, optional): _description_. Defaults to ["momentum","value","growth","smallcap","lowvol"].
        """
        for i in range(len(factors)):
            
            self.factor = excess_performance(
            self.data[factors[i]].dropna(), self.data.developedequity.dropna()
            )
            self.factor.name = factors[i]
            self.get_factor_value(self.factor,factors[i])
            
    def calc_macro_factors(self,) -> None:
        """_summary_
        calculate macro factors from raw macro data.
        calculate 9 macro factors (rate, equity, credit, commodity, emerging, localinflation, localequity,shortvol, currency) by default.       
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
            
    def process(self, factors = ['rate', 'equity','credit','commodity','emerging', 'currency','localequity','localinflation',
    'shortvol','momentum','value','growth','smallcap','lowvol']) -> pd.DataFrame():
        """_summary_
        call calc_style_factors function and calc_macro_factors faunction
        build a factor dataframe using a class attribute, factors_lst, and insert the calculated factor data to TB_Lens table.
        Args:
            factors (list, optional): _description_. Defaults to ['rate', 'equity','credit','commodity','emerging', 'currency','localequity','localinflation', 'shortvol','momentum','value','growth','smallcap','lowvol'].

        Returns:
            _type_: _description_
        """
        self.calc_style_factors(factors = ["momentum","value","growth","smallcap","lowvol"])
        self.calc_macro_factors()
        
        df_factors = pd.DataFrame()
        df_factors = df_factors.append(self.factors_lst)
        df_factors = df_factors.sort_values(by=['trd_dt'],ascending=False)
                 
        try:
            db.TbLens.insert(df_factors)
        except:
            db.TbLens.update(df_factors)
        
        return df_factors
    

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
        Returns:
            _type_: pd.Series 
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
