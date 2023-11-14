import os
import sys
import logging
import numpy as np
import pandas as pd

from dateutil import parser
from sklearn import mixture
from typing import Union, Any, Optional
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from hive import db
from config import get_args

logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)


class GMM:
    
    """
    Gaussian Mixture Model\n
    : Detect a dangerous economic regime, which should liquidate risky assets and secure getting enough liquidity, with daily returns of assets
    """

    def __init__(
        self,
        rolling: int = 5,
        period: int = 1,
        term: int = 12,
        start: Optional[date] = datetime.strptime('20090601','%Y%m%d'),
        end: Optional[date] = TODAY,
    ) -> None:
        """
        Initiallization

        Args:
            rolling (int, optional): the term of rolling. Defaults to 5.
            period (int, optional): the period of making return data with daily price data(e.g., weekly: m=5, monthly: m=21, quarterly: m=63, annaully: m=252). Defaults to 1.
            term (int, optional): the term to annualize the daily price data(e.g., daily: m=252, weekly: m=52, monthly: m=12, quarterly: m=4, annaully: m=1). Defaults to 12.
            start (Optional[date], optional): the start date of getting data from DB for this algorithm. Defaults to datetime.strptime('20090601','%Y%m%d').
            end (Optional[date], optional): the end date of getting data from DB for this algorithm. Defaults to TODAY.
        """
        self.rolling = rolling
        self.data = db.query.get_GMM_data(start, end, rolling)
        self.rs(gross_rtn=self.data, n=rolling, m=period)
        self.st_weights(r=self.ret, term=term)


    def rs(self, gross_rtn: pd.DataFrame, n: int = 5, m: int = 1):
        """
        This function gets rate of return data to make excess return data sets\n
        : The excess return data sets will be the input of GMM Algorithm.

        Args:
            gross_rtn (pd.DataFrame): TimeSeries of daily `gross_rtn` of all the assets we know(from TB_META of DB)
            n (int): the term of rolling
            m (int): the period of making return data with daily price data(e.g., weekly: m=5, monthly: m=21, quarterly: m=63, annaully: m=252)

        Returns:
            ret (pd.DataFrame): the general rate of return by rolling of `n`
            ret_t1 (pd.DataFrame): the general rate of return by rolling of `n` and the return data set which consist of the rate of return lagged 1-day, 1-week and 1-month
            ret_t2 (pd.DataFrame): the general rate of return by rolling of `n` and the return data set which consist of the rate of return lagged 1-day, 2-day, 3-day and 4-day
        """
        ret = gross_rtn.rolling(n).mean().interpolate()[::m].dropna()
        for i in range(1, m):
            ret = pd.concat([ret, gross_rtn.rolling(n).mean().interpolate()[i::m].dropna()])
        ret.sort_index(inplace=True)
        ret_1=ret.shift(1)
        ret_2=ret.shift(2)
        ret_3=ret.shift(3)
        ret_4=ret.shift(4)
        ret_5=ret.shift(5)
        ret_21=ret.shift(21)
        ret_1.rename(columns = lambda x: x + "_1", inplace = True)
        ret_2.rename(columns = lambda x: x + "_2", inplace = True)
        ret_3.rename(columns = lambda x: x + "_3", inplace = True)
        ret_4.rename(columns = lambda x: x + "_4", inplace = True)
        ret_5.rename(columns = lambda x: x + "_5", inplace = True)
        ret_21.rename(columns = lambda x: x + "_21", inplace = True)
        ret_t1=pd.concat([ret, ret_1, ret_5, ret_21], axis=1).dropna()
        ret_t2=pd.concat([ret, ret_1, ret_2, ret_3, ret_4], axis=1).dropna()

        # excess return
        self.ret = ret-ret.mean() 
        self.ret_t1 = ret_t1-ret_t1.mean() 
        self.ret_t2 = ret_t2-ret_t2.mean()
    
    def add_states(self, arg: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """assign the state column to the pandas"""
        states = self.weights.iloc[:,1].resample("D").last().ffill().reindex(arg.index).ffill()
        return arg.assign(states=states)
    
    def expected_returns_by_states(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """calculate expected return by states"""
        price_df = price_df.sort_index().resample("d").last().ffill()
        weights_index = self.weights.reset_index()["trd_dt"].apply(lambda x: x.date().strftime("%Y-%m-%d"))
        data = price_df.reindex(weights_index)
        fwd_return = self.add_states(data.pct_change().shift(-1)).dropna()
        grouped = fwd_return.groupby(by="states").mean() * 12
        return grouped

    def get_state(self, date: Any) -> str:
        """get the current market regime measured by GMM algorithm"""
        self.weights.index = pd.to_datetime(self.weights.index)
        return self.weights.iloc[:,1].loc[:pd.to_datetime(date)].resample("d").last().ffill().iloc[-1]
    
    def st_weights(self, r: pd.DataFrame, term: int):
        """
        This function gets the current market regime and the weight when each market regime comes\n
        Get the each `label` meaning the market regime by getting the input `r`(rate of return of each asset)\n
        Amalgamize the `label` and `r` so that we can figure out the features of the `r` following each regime.\n
        According to the features, we decide whether the current market regime is `normal` or `risky` and determine the weights.

        Args:
            r (pd.DataFrame): the rate of return data

        Returns:
            weights (pd.DataFrame): the weights this strategy suggests for the current market regime
            weights_bm (pd.DataFrame): the benchmark weights
            labels (pd.DataFrame): the labels of the rate of return data
            st (pd.DataFrame): the market regime according to each label
        """
        labels = mixture.GaussianMixture(n_components=2, random_state=2).fit(r.iloc[:,:]).predict(r.iloc[:,:])
        labels =pd.DataFrame(labels, index=self.data.trd_dt[:len(r)], columns=['label'])
        grouped = pd.concat([r,labels],axis=1).groupby(by=["label"])
        excess_return = grouped.mean() * term
        states =  excess_return.copy()
        
        states.loc[(states.index == 0), 'state'] ='risky'
        states.loc[(states.index == 1), 'state'] ='normal'

        # 비중 정하기
        st=states['state']        
        wei = pd.merge(left= labels, right= st, how = "inner", left_on='label', right_index=True)
        wei.value_counts()
        wei['equity'] = 0.6
        wei.loc[wei['state']=='risky', 'equity'] = 0.1
        wei.loc[wei['state']=='normal', 'equity'] = 0.6
        wei.value_counts()
        wei['bond'] = 1 - wei['equity']
        wei.value_counts()
        weights= wei.sort_index()

        self.weights = weights
        self.labels = labels
        self.states = st
        
        
def run_regime_gmm(regime: str = "GMM", today: datetime.date = date.today()) -> None:
    """
    GMM regime detection

    Args:
        regime (str, optional): the regime module name. Defaults to "GMM".
        today (datetime.date, optional): the exact date of runnging this function. Defaults to date.today().

    Returns:
        pd.DataFrame: `df_state`
    """
    extra = dict(user=args.user, activity="regime detection", category="script")
    
    if today != db.get_start_trading_date(market="KR", asofdate=today):
        logger.info(msg=f"[SKIP] {regime} Regime. {today:%Y-%m-%d}", extra=extra)
        return

    logger.info(msg=f"[PASS] Start {regime} Regime. {today:%Y-%m-%d}", extra=extra)
    
    regime_model = GMM(end=today)
    
    state = regime_model.get_state(today.strftime("%Y-%m-%d"))
        
    if state == "risky":
        equity = 0
        fixed_income = 0
        alternative = 0.1
        liquidity = 0.9
        
    else:
        equity = 0.7
        fixed_income = 0.2
        alternative = 0.1
        liquidity = 0
        
    df_state = pd.DataFrame(
        {"trd_dt": [today],
         "module": [f"{regime}"],
         "regime": [state],
         "equity": [equity],
         "fixed_income": [fixed_income],
         "alternative": [alternative],
         "liquidity": [liquidity],
        }
    )
    
    print(df_state)
    
    from core.model.regime.base import run_abl_regime_allocation
    run_abl_regime_allocation(regime, today)
    
    logger.info(msg=f"[PASS] End {regime} Regime. {today:%Y-%m-%d}", extra=extra)
    
    return df_state