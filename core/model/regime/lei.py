import os
import sys
from typing import Union, Any
import logging
import numpy as np
import pandas as pd
from datetime import date
from dateutil import parser
from statsmodels.api import tsa
import pandas_datareader as pdr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from hive import db
from config import get_args

logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)

class USLEIHP:
    
    """
    US leading economic indicator signal
    """

    def __init__(
        self,
        lamb: int = 0,
        min_periods: int = 12,
        months_offsets: int = 1,
        resample_by: str = "M",
        asofdate: date = date.today(),
    ) -> None:
        """
        Initiallization

        Args:
            lamb (int, optional): `lambda` , the smoothing degree. Defaults to 0.
            min_periods (int, optional): the minimum period of the LEI data. Defaults to 12 (month).
            months_offsets (int, optional): applicate the lagging of the LEI data. Defaults to 1 (month).
            resample_by (str, optional): set the resampling term of the data to be used generally. Defaults to "M" (monthly).
            asofdate (date, optional): the end date to get LEI data. Defaults to date.today().
        """
        self.data = db.get_lei(asofdate=asofdate)
        self.data.index = self.data.index + pd.DateOffset(months=1)
        self.months_offset = months_offsets
        self.lamb = lamb
        self.min_periods = min_periods
        self.resample_by = resample_by
        self.signals: Union[pd.Series, pd.DataFrame] = pd.DataFrame(dtype=float)
        self.states: Union[pd.Series, pd.DataFrame] = pd.DataFrame(dtype=str)
        self.process()

    def add_states(self, arg: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """assign the state column to the pandas"""
        states = self.states.resample("D").last().ffill().reindex(arg.index).ffill()
        return arg.assign(states=states)

    def expected_returns_by_states(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """calculate expected return by states"""
        price_df = price_df.sort_index()
        data = price_df.resample("d").last().ffill().reindex(self.states.index)
        fwd_return = self.add_states(data.pct_change().shift(-1)).dropna()
        grouped = fwd_return.groupby(by="states").mean() * 12
        return grouped

    def get_state(self, date: Any) -> str:
        """get the current market regime measured by LEI algorithm"""
        return self.states.resample("D").last().ffill().loc[:date].iloc[-1]

    @staticmethod
    def leading_economic_indicator() -> pd.DataFrame:
        """Get raw data for leading economic indicator regime"""
        tickers = dict(USALOLITONOSTSAM="USLEI")
        data = pdr.DataReader(list(tickers.keys()), "fred", start="1900-01-01") - 100
        data = data.rename(columns=tickers)
        return data[["USLEI"]]

    def process(self) -> None:
        """process signals"""
        # process data for processing.
        if self.resample_by:
            self.data = self.data.resample(self.resample_by).last().dropna()

        result = []

        for idx, date in enumerate(self.data.index):
            if idx < self.min_periods:
                continue
            _, trend = tsa.filters.hpfilter(
                x=self.data.loc[:date].values, lamb=self.lamb
            )
            result.append(
                dict(date=date, level=trend[-1], direction=np.diff(trend)[-1])
            )
        self.signals = pd.DataFrame(result).set_index(keys="date")

        def _mapper(row: pd.Series) -> str:
            if row.level >= 0 and row.direction >= 0:
                return "expansion"
            if row.level < 0 and row.direction >= 0:
                return "recovery"
            if row.level < 0 and row.direction < 0:
                return "contraction"
            if row.level >= 0 and row.direction < 0:
                return "slowdown"
            raise ValueError("???")

        self.states = self.signals.apply(_mapper, axis=1)
        

def run_regime_lei(regime: str = "lei", today: date = date.today()) -> None:
    """
    LEI regime detection

    Args:
        regime (str, optional): the regime module name. Defaults to "lei".
        today (date, optional): the exact date of runnging this function. Defaults to date.today().

    Returns:
        pd.DataFrame: `df_state`
    """
    extra = dict(user=args.user, activity="regime detection", category="script")
    
    if today != db.get_start_trading_date(market="KR", asofdate=today):
        logger.info(msg=f"[SKIP] LEI Regime. {today:%Y-%m-%d}", extra=extra)
        return

    logger.info(msg=f"[PASS] Start LEI Regime. {today:%Y-%m-%d}", extra=extra)
    
    regime_model = USLEIHP(asofdate=today)
    
    state = regime_model.get_state(today.strftime("%Y-%m-%d"))
    
    if state == "expansion":  
        equity = 0.8
        fixed_income = 0.1
        alternative = 0.1
        liquidity = 0

    elif state == "recession":
        equity = 0
        fixed_income = 0.2
        alternative = 0.1
        liquidity = 0.7
        
    else:
        equity = 0.6
        fixed_income = 0.2
        alternative = 0.1
        liquidity = 0.1
        
    df_state = pd.DataFrame(
        {"trd_dt": [today],
         "module": ["LEI"],
         "regime": [state],
         "equity": [equity],
         "fixed_income": [fixed_income],
         "alternative": [alternative],
         "liquidity": [liquidity],
        }
    )
    
    print(df_state)
    
    logger.info(msg=f"[PASS] End LEI Regime. {today:%Y-%m-%d}", extra=extra)
    
    return df_state