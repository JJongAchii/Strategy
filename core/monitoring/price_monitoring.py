import os
import sys
import logging
import numpy as np
import pandas as pd
from dateutil import parser
from datetime import timedelta,datetime


sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import get_args
from hive import db

logger = logging.getLogger("sqlite")
##############################################################################################################
args = get_args()
TODAY = parser.parse(args.date)
YESTERDAY = TODAY - timedelta(days=1)

universe = db.load_universe()[["strategy","stk_id"]].reset_index()
##############################################################################################################
def kr_price_monitoring():
    """
    KR price monitoring function
    
    Monitor whether the data inserted today is accurate or not.\n
    
    There are 5 things to check below:\n
    1) No price: the price data for the ticker hasn't been inserted into DB.\n
       It can be caused by the ticker changed or delisted.\n
    2) No volume, but price changed: the volome data for the ticker is zero, but the price has changed. \n
       Therefore, we should check if there's any things like stock split or dividend.\n
    3) No volume no change: the price has not changed due to lack of its trading volume.\n
       It will be certain that there is NO PROBLEM. No brainer.\n
    4) Excess return: the absolute price change percentage is more than 5%.\n
       Check if the price is the price of another asset or doesn't make sense.\n
    5) Same price: the prices of today and yesterday are the same. \n
       Check if today's price is yesterday's one or not.
    """
    global universe
    extra = dict(user=args.user, activity="KR_daily_price_uploaded", category="script")
    extra2 = dict(user=args.user, activity="KR_daily_price_uploaded", category="monitoring")
    
    if db.query.check_weekend_holiday_date("KR", TODAY):
        logger.info(msg=f"[SKIP] KR_price monitoring. {TODAY:%Y-%m-%d}", extra=extra)
        return
    
    logger.info(msg=f"[PASS] start KR_price monitoring. {TODAY:%Y-%m-%d}", extra=extra)
    
    today_daily_data = db.TbDailyBar.query_df(trd_dt=TODAY)[
        ["trd_dt", "stk_id", "close_prc", "gross_rtn", "adj_value"]
    ] 
    kr_daily_data = today_daily_data[(today_daily_data.stk_id < 2215)&(today_daily_data.stk_id > 1490)]
    kr_adj_value_df_1d = db.query.get_last_trading_date_price(TODAY, "KR")
    
    today_volume = db.TbMetaUpdat.query_df(trd_dt=TODAY)[["trd_dt","stk_id","trd_volume"]]
    today_volume_zero = today_volume[today_volume.trd_volume == 0]
    
    no_price = [] 
    no_volume_but_price_changed  = [] 
    no_volume_no_change  = [] 
    yes_volume_with_excess_rtn = [] 
    yes_volume_but_price_changed = []
    with db.session_local() as session:
        for meta in db.TbMeta.query().all():
            stk_id = meta.stk_id

            if (meta.iso_code == "KR") & (meta.status is None):
                if stk_id in universe.stk_id.tolist():
                    today_close = kr_daily_data[kr_daily_data["stk_id"]==stk_id].close_prc
                    if today_close.empty:
                        no_price.append(meta.ticker)
                    else:
                        today_close = float(np.array2string(today_close.values, separator=',')[1:-1])
                        yesterday_close = kr_adj_value_df_1d[kr_adj_value_df_1d["stk_id"]==meta.stk_id].close_prc.values
                        yesterday_close = float(np.array2string(yesterday_close, separator=',')[1:-1])
                        rtn = (today_close-yesterday_close)/yesterday_close

                        if stk_id in today_volume_zero.stk_id.tolist():
                            if abs(rtn) != 0:
                                no_volume_but_price_changed.append(meta.ticker)
                            elif abs(rtn) == 0:
                                no_volume_no_change.append(meta.ticker)
                        else:
                            if abs(rtn) > 0.05:
                                yes_volume_with_excess_rtn.append(meta.ticker)
                            elif abs(rtn) == 0:
                                yes_volume_but_price_changed.append(meta.ticker)
            else:
                continue

        if len(no_price) == 0:
            no_price = " Nothing "
        if len(no_volume_but_price_changed) == 0:
            no_volume_but_price_changed = " Nothing "
        if len(no_volume_no_change) == 0:
            no_volume_no_change = " Nothing "
        if len(yes_volume_with_excess_rtn) == 0:
            yes_volume_with_excess_rtn = " Nothing "
        if len(yes_volume_but_price_changed) == 0:
            yes_volume_but_price_changed = " Nothing "
        
        if (len(no_price)!=0) | (len(no_volume_but_price_changed)!=0) | (len(no_volume_no_change)!=0) | (len(yes_volume_with_excess_rtn)!=0) | (len(yes_volume_but_price_changed)!=0):
            logger.warning(
                msg=f"[PASS] KR price monitoring | {TODAY:%Y-%m-%d}\n\
                    -No price:{no_price}\n\
                    -No volume, but price changed:{no_volume_but_price_changed}\n\
                    -No volume, No price change:{no_volume_no_change}\n\
                    -Excessive absolute return:{yes_volume_with_excess_rtn}\n\
                    -Same price:{yes_volume_but_price_changed}",
                extra=extra2)
        
        logger.info(msg=f"[PASS] End KR_price monitoring. {TODAY:%Y-%m-%d}", extra=extra)

##############################################################################################################
def us_price_monitoring():
    """
    US price monitoring function
    
    Monitor whether the data inserted today is accurate or not.\n
    
    There are 5 things to check below:\n
    1) No price: the price data for the ticker hasn't been inserted into DB.\n
       It can be caused by the ticker changed or delisted.\n
    2) No volume, but price changed: the volome data for the ticker is zero, but the price has changed. \n
       Therefore, we should check if there's any things like stock split or dividend.\n
    3) No volume no change: the price has not changed due to lack of its trading volume.\n
       It will be certain that there is NO PROBLEM. No brainer.\n
    4) Excess return: the absolute price change percentage is more than 5%.\n
       Check if the price is the price of another asset or doesn't make sense.\n
    5) Same price: the prices of today and yesterday are the same. \n
       Check if today's price is yesterday's one or not.
    """
    global universe
    extra = dict(user=args.user, activity="US_daily_price_uploaded", category="script")
    extra2 = dict(user=args.user, activity="US_daily_price_uploaded", category="monitoring")
    
    if db.query.check_weekend_holiday_date("US", TODAY):
        logger.info(msg=f"[SKIP] US_price monitoring. {TODAY:%Y-%m-%d}", extra=extra)
        return
    
    logger.info(msg=f"[PASS] start US_price monitoring. {TODAY:%Y-%m-%d}", extra=extra)
    
    yesterday_daily_data = db.TbDailyBar.query_df(trd_dt=YESTERDAY)[
        ["trd_dt", "stk_id", "close_prc", "gross_rtn", "adj_value"]
    ] 
    us_daily_data = pd.concat([yesterday_daily_data[yesterday_daily_data.stk_id > 2214], yesterday_daily_data[yesterday_daily_data.stk_id<1491]])
    us_adj_value_df_1d = db.query.get_last_trading_date_price(YESTERDAY, "US")
    
    yesterday_volume = db.TbMetaUpdat.query_df(trd_dt=YESTERDAY)[["trd_dt","stk_id","trd_volume"]]
    yesterday_volume_zero = yesterday_volume[yesterday_volume.trd_volume == 0]
    
    no_price = [] 
    no_volume_but_price_changed  = [] 
    no_volume_no_change  = [] 
    yes_volume_with_excess_rtn = [] 
    yes_volume_but_price_changed = []
    with db.session_local() as session:
        for meta in db.TbMeta.query().all():
            stk_id = meta.stk_id

            if (meta.iso_code == "US") & (meta.status is None) & (meta.source != "bloomberg"):
                if stk_id in universe.stk_id.tolist():
                    today_close = us_daily_data[us_daily_data["stk_id"]==stk_id].close_prc
                    if today_close.empty:
                        no_price.append(meta.ticker)
                    else:
                        today_close = float(np.array2string(today_close.values, separator=',')[1:-1])
                        yesterday_close = us_adj_value_df_1d[us_adj_value_df_1d["stk_id"]==stk_id].close_prc.values
                        yesterday_close = float(np.array2string(yesterday_close, separator=',')[1:-1])
                        rtn = (today_close-yesterday_close)/yesterday_close

                        if stk_id in yesterday_volume_zero.stk_id.tolist():
                            if abs(rtn) != 0:
                                no_volume_but_price_changed.append(meta.ticker)
                            elif abs(rtn) == 0:
                                no_volume_no_change.append(meta.ticker)
                        else:
                            if abs(rtn) > 0.05:
                                yes_volume_with_excess_rtn.append(meta.ticker)
                            elif abs(rtn) == 0:
                                yes_volume_but_price_changed.append(meta.ticker)
                else:
                    continue
        
        if len(no_price) == 0:
            no_price = " Nothing "
        if len(no_volume_but_price_changed) == 0:
            no_volume_but_price_changed = " Nothing "
        if len(no_volume_no_change) == 0:
            no_volume_no_change = " Nothing "
        if len(yes_volume_with_excess_rtn) == 0:
            yes_volume_with_excess_rtn = " Nothing "
        if len(yes_volume_but_price_changed) == 0:
            yes_volume_but_price_changed = " Nothing "

        if (len(no_price)!=0) | (len(no_volume_but_price_changed)!=0) | (len(no_volume_no_change)!=0) | (len(yes_volume_with_excess_rtn)!=0) | (len(yes_volume_but_price_changed)!=0):
            logger.warning(
                msg=f"[PASS] US price monitoring | {TODAY:%Y-%m-%d}\n\
                    -No price:{no_price}\n\
                    -No volume, but price changed:{no_volume_but_price_changed}\n\
                    -No volume, No price change:{no_volume_no_change}\n\
                    -Excessive absolute return:{yes_volume_with_excess_rtn}\n\
                    -Same price:{yes_volume_but_price_changed}",
                extra=extra2)
        
        logger.info(msg=f"[PASS] End US_price monitoring. {TODAY:%Y-%m-%d}", extra=extra)