import os
import sys
import logging
import numpy as np
import pandas as pd
from dateutil import parser
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import get_args
from hive import db

logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)

extra = dict(user=args.user, activity="Bloomberg_Index_Monitoring", category="script")
extra2 = dict(user=args.user, activity="Bloomberg_Index_Monitoring", category="monitoring")

##############################################################################################################
def index_monitoring():
    """
    US only
    """
    if TODAY.date().weekday() == 6:
        logger.info(msg=f"[SKIP] Index Monitoring. {TODAY:%Y-%m-%d}", extra=extra)
        return
        
    else:
        logger.info(msg=f"[PASS] Start Index Monitoring. {TODAY:%Y-%m-%d}", extra=extra)

    today_daily_data = db.get_macro_data_by_created_date(TODAY)
    today_daily_data.created_date = today_daily_data.created_date.apply(lambda x: x.date())
    today_daily_data = today_daily_data[today_daily_data.created_date == today_daily_data.created_date.min()]
    print("today_daily_data:\n", today_daily_data, "\n", "-"*100)
    
    yesterday_daily_data = db.get_last_trading_date_index_by_created_date(TODAY)
    yesterday_daily_data = yesterday_daily_data[yesterday_daily_data.created_date == yesterday_daily_data.created_date.min()]
    print("yesterday_daily_data:\n", yesterday_daily_data, "\n", "-"*100)
    
    no_price, excess_rtn = [],[] 
    with db.session_local() as session:
        for macro in db.TbMacro.query().all():
            macro_id = macro.macro_id

            today_value = today_daily_data[today_daily_data["macro_id"]==macro_id].adj_value
            yesterday_value = yesterday_daily_data[yesterday_daily_data["macro_id"]==macro_id].adj_value

            if today_value.empty:
                no_price.append(macro.ticker)
            else:
                today_value = float(np.array2string(today_value.values, separator=',')[1:-1])
                yesterday_value = float(np.array2string(yesterday_value.values, separator=',')[1:-1])
                rtn = (today_value-yesterday_value)/yesterday_value
                if abs(rtn) > 0.05:
                    excess_rtn.append(macro.ticker)
                else:
                    continue

    if (len(no_price)!=0) | (len(excess_rtn)!=0):
        logger.warning(
            msg=f"[WARN] Index Monitoring | {TODAY:%Y-%m-%d}\n\
                -No value: {no_price}\n\
                -Excessive absolute return: {excess_rtn}",
            extra=extra2)

    logger.info(msg=f"[PASS] End Index Monitoring.", extra=extra)