import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Union, Any
from dateutil import parser
from datetime import date, datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from hive import db
from config import get_args

parent_folder = os.path.dirname(os.path.abspath(__file__))
parent_parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if parent_folder not in sys.path:
    sys.path.append(parent_folder)
if parent_parent_folder not in sys.path:
    sys.path.append(parent_parent_folder)

from core.model.regime.base import BaseRegime, run_abl_regime_allocation

logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)


class IML(BaseRegime):

    """
    Inflation Matching Line (IML) regime signal class.\n
    Internally know as Inflation Matching Line (IML).\n
    Made with Inflation Expectation Rate(5yF5y), US Generic Gov Yield(10Y), and Forward Term Premium(10Y)
    """
    
    def __init__(
        self, 
        start: str = datetime(2010,1,4), 
        end: str = TODAY,
        short_yield_range: tuple = (-1.5, 0.5),
        inflation_range: tuple = (1, 3),
        bins: int = 5
    ) -> None:
        """
        Initialization
        
        Args:
            start (None, optional): start date of the raw data fetch. Defaults to None
            end (None, optional): end date of the raw data fetch. Defaults to None
            short_yield_range (tuple, optional): the bound of the short_yield values. Defaults to (-1.5, 0.5)
            inflation_range (tuple, optional): the bound of the inflation values. Defaults to (1, 3)
            bins (int, optional): number of bins for grid. Defaults to 5
        """
        self.raw_data = db.query.get_IML_data(start, end)
        self.inflation_bin_proxy = ["AA", "A", "B", "C", "D", "DD"]
        self.short_yield_bin = np.append(np.linspace(short_yield_range[0],
                                                     short_yield_range[1],
                                                     bins),
                                         np.inf)
        self.inflation_bin = np.append(np.linspace(inflation_range[0],
                                                   inflation_range[1],
                                                   bins),
                                       np.inf)
        self.fit()

    def add_states(self, arg: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """assign the state column to the pandas"""
        states = self.states.resample("D").last().ffill().reindex(arg.index).ffill()
        return arg.assign(states=states)

    def expected_returns_by_states(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """calculate expected return by states"""
        data = price_df.sort_index().resample("d").last().ffill().reindex(self.states.index)
        fwd_return = self.add_states(data.pct_change().shift(-1)).dropna()
        grouped = fwd_return.groupby(by="states").mean() * 12
        return grouped

    def get_state(self, date: Any) -> str:
        """get the current market regime measured by IML algorithm"""
        self.states.index = pd.to_datetime(self.states.index)
        return self.states.resample("D").last().ffill().loc[:date].iloc[-1]

    def fit(self, short_yield_window:int=5, inflation_window:int=5):
        """
        Fit the raw data into different inflation regimes.

        Args:
            short_yield_window (int, optional): short_yield's window to use (default: 63days(3months))
            inflation_window (int, optional): inflation's window to use (default: 63days(3months))

        Returns:
            instance: self
        """
        # Transform raw_data into moving averages.
        moving_average = self.raw_data.copy()
        moving_average["short_yield"] = moving_average["short_yield"].rolling(short_yield_window).mean()
        moving_average["inflation"] = moving_average["inflation"].rolling(inflation_window).mean()
        moving_average = moving_average.dropna()

        # Apply regime Labeling.
        self.states = moving_average.dropna().apply(self.regime_labeling, axis=1)

        return self

    def regime_labeling(self, current_signal: pd.Series):
        """
        Helper function to label regime state based on inflation and short-yeild.

        Args:
            current_signal (pd.Series): the current inflation and short-yield numbers.

        Returns:
            str: the regime state.
        """
        inf_id, sht_id = None, None
        for idx, inf_bin in enumerate(self.inflation_bin):
            if current_signal.inflation < inf_bin:
                inf_id = idx
                break

        for idx, sty_bin in enumerate(self.short_yield_bin, 1):
            if current_signal.short_yield < sty_bin:
                sht_id = idx
                break

        self.signal = self.inflation_bin_proxy[inf_id] + str(sht_id)

        return self.signal


def run_regime_iml(regime: str = "IML", today: datetime.date = date.today()) -> None:
    """
    IML regime detection

    Args:
        regime (str, optional): the regime module name. Defaults to "IML".
        today (datetime.date, optional): the exact date of runnging this function. Defaults to date.today().

    Returns:
        pd.DataFrame: `df_state`
    """
    extra = dict(user=args.user, activity="regime detection", category="script")
    
    if today != db.get_start_trading_date(market="KR", asofdate=today):
        logger.info(msg=f"[SKIP] {regime} Regime. {today:%Y-%m-%d}", extra=extra)
        return

    logger.info(msg=f"[PASS] Start {regime} Regime. {today:%Y-%m-%d}", extra=extra)
    
    regime_model = IML(end=today)
    
    state = regime_model.get_state(today.strftime("%Y-%m-%d"))
        
    equity_focus_regime = ["AA6", "A5", "B4", "C3", "D2", "DD1"]
    fixed_income_focus_regime= ["A6", "B6", "C6", "D6", "DD6" \
                                      "B5", "C5", "D5", "DD5" \
                                            "C4", "D4", "DD4" \
                                                  "D3", "DD3" \
                                                        "DD2" ]
    
    if state in fixed_income_focus_regime:
        state_focus = "Bond Overweight"
        equity = 0.4
        fixed_income = 0.5
        alternative = 0.1
        liquidity = 0
        
    elif state in equity_focus_regime:
        state_focus = "Equity Overweight"
        equity = 0.8
        fixed_income = 0.1
        alternative = 0.1
        liquidity = 0
        
    else:
        state_focus = "Neutral"
        equity = 0.6
        fixed_income = 0.3
        alternative = 0.1
        liquidity = 0
        
    df_state = pd.DataFrame(
        {"trd_dt": [today],
         "module": [f"{regime}"],
         "regime": [state_focus],
         "equity": [equity],
         "fixed_income": [fixed_income],
         "alternative": [alternative],
         "liquidity": [liquidity],
        }
    )
    
    print(df_state)
    
    run_abl_regime_allocation(regime, today)
    
    logger.info(msg=f"[PASS] End {regime} Regime. {today:%Y-%m-%d}", extra=extra)
    
    return df_state