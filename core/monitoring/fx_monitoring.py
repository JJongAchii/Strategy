import os
import sys
import logging
import numpy as np
import pandas as pd
from dateutil import parser
from datetime import timedelta


sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import get_args
from hive import db

args = get_args()
logger = logging.getLogger("sqlite")
extra = dict(user=args.user, activity="Bloomberg_Index_Monitoring", category="script")
extra2 = dict(user=args.user, activity="Bloomberg_Index_Monitoring", category="monitoring")

TODAY = parser.parse(args.date)
YESTERDAY = TODAY - timedelta(days=1)
LASTTRADINGDATE = YESTERDAY - timedelta(days=1)
if LASTTRADINGDATE.weekday() >= 5:
    while LASTTRADINGDATE.weekday() >= 5:
        LASTTRADINGDATE -= timedelta(days=1)

##############################################################################################################
def fx_monitoring():
    """
    US only
    """
    if TODAY.date().weekday() == 6 or TODAY.date().weekday() == 0:
        logger.info(msg=f"[SKIP] FX Monitoring. {TODAY:%Y-%m-%d}", extra=extra)
        return        
    
    logger.info(msg=f"[PASS] start FX Monitoring. {TODAY:%Y-%m-%d}", extra=extra)

    fx_data = db.get_fx().reset_index()
    fx_data.trd_dt = pd.to_datetime(fx_data.trd_dt)
    today_fx_data = fx_data[fx_data.trd_dt == YESTERDAY]
    lasttradingdate_fx_data = fx_data[fx_data.trd_dt == LASTTRADINGDATE]
    
    no_price, excess_rtn = [], []
    with db.session_local() as session:
        for currency in fx_data.currency.unique().tolist():
            each_currency_today_fx_data = today_fx_data[today_fx_data.currency == currency]
            each_currency_lasttradingdate_fx_data = lasttradingdate_fx_data[lasttradingdate_fx_data.currency == currency]
            
            if each_currency_today_fx_data.empty:
                no_price.append(currency)
            else:
                today_fx = float(np.array2string(each_currency_today_fx_data.close_prc.values, separator=',')[1:-1])
                lasttradingdate_fx = float(np.array2string(each_currency_lasttradingdate_fx_data.close_prc.values, separator=',')[1:-1])
                rtn = (today_fx-lasttradingdate_fx)/lasttradingdate_fx
                if abs(rtn) > 0.05:
                    excess_rtn.append(currency)
                else:
                    continue
                
    if (len(no_price)!=0) | (len(excess_rtn)!=0):
        logger.warning(
            msg=f"[WARN] FX Monitoring | {TODAY:%Y-%m-%d}\n\
                -No value: {no_price}\n\
                -Excessive absolute return: {excess_rtn}",
            extra=extra2)

    logger.info(msg=f"[PASS] End FX Monitoring.", extra=extra)