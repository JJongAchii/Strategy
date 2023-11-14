"""
Daily fx Data
"""
import os
import sys
import pandas as pd
import argparse
from datetime import date, timedelta, datetime
from dateutil import parser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from hive import db 
import logging
from config import get_args

logger = logging.getLogger("sqlite")
####################################################################################################
# parse arguments
parse = argparse.ArgumentParser(description="Run index_update Script.")
parse.add_argument("-d", "--date", default=date.today().strftime("%Y-%m-%d"))
args = get_args()

####################################################################################################
# global variables
TODAY   = parser.parse(args.date)
TODAY_  = TODAY.date()
TODAY_8 = TODAY_.strftime("%Y%m%d")

YESTERDAY   = TODAY   - timedelta(days=1)
YESTERDAY_  = TODAY_  - timedelta(days=1)
YESTERDAY_8 = YESTERDAY_.strftime("%Y%m%d")

extra = dict(user=args.user, activity="Daily_fx_inserting_check", category="script")
extra2 = dict(user=args.user, activity="Daily_fx_inserting_check", category="monitoring")

##################################################################################################
def fx_uploader(filedata, fx_ticker):
    """
    Insert/Update the index data into/in DB by checking the ticker of the foreign currency

    Args:
        filedata (pd.DataFrame): The index data fed 
        fx_ticker (str): The legacy ticker of each index

    Returns:
        str: return its ticker if there has been any error while inserting or updating
    """
    if fx_ticker in filedata.ticker.tolist():
        data = filedata[filedata.ticker == fx_ticker][["trd_dt","ticker","value"]].reset_index().drop("index", axis=1)
        data.columns = ["trd_dt","currency","close_prc"]
        quote_currency = data.currency[0][:3]
        base_currency = data.currency[0][3:6]
        if base_currency == "KRW":
            data.currency = quote_currency
        else:
            data.currency = "/".join([quote_currency, base_currency])
            
        if data.empty:
            return fx_ticker
        else:
            try:
                db.TbFX.insert(data)
            except:
                try:
                    db.TbFX.update(data)
                except:
                    return fx_ticker
                
##################################################################################################
def run_update_fx():
    """
    Insert/Update all the fx data from diffrent files fed
    """
    if TODAY.date().weekday() == 6 or TODAY.date().weekday() == 0:
        logger.info(msg=f"[SKIP] FX Upload. {TODAY:%Y-%m-%d}", extra=extra)
        return        
    
    logger.info(msg=f"[PASS] start FX Upload. {TODAY:%Y-%m-%d}", extra=extra)
    
    # Read fed File
    new_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../hive/eai/receive/Index"))
    os.chdir(new_dir)
    sys.path.insert(0, new_dir)
    idx_current_file_path,idx_new_file_path = f'Index_close_{YESTERDAY_8}', f'Index_close_{YESTERDAY_8}.txt'
    try:
        os.rename(idx_current_file_path, idx_new_file_path)
    except:
        pass

    try:
        bb_index3 = pd.read_csv(f"Index_close_{YESTERDAY_8}.txt", sep="|",encoding="euc-kr",names=['market','representative_index','trd_dt','ticker','unit','value','prc_change','gross_rtn'])
        bb_index3 = bb_index3[['trd_dt','ticker','unit','value']]
    except:
        logger.warning(msg=f"[WARN] No file named Index_close_{YESTERDAY_8}", extra=extra2)
        return
    
    # Insert FX data into DB
    failed_tickers = []
    with db.session_local() as session:
        failed_tickers = []
        for macro in db.TbMacro.query().all():
            if macro.memo == "FX":
                try:
                    ticker= fx_uploader(bb_index3, macro.ticker)
                    if ticker is not None:
                        failed_tickers.append(ticker)
                except:
                    pass

    # KRW currency data insert
    krw = pd.DataFrame({"trd_dt": [YESTERDAY_8],"currency": ["KRW"],"close_prc": [1]})
    try:
        db.TbFX.insert(krw)
    except:
        db.TbFX.update(krw)
        
    if failed_tickers:
        logger.warning(msg=f"[WARN] FX Uploading Fail\n-Failed ticker: {failed_tickers}", extra=extra2)
        
    logger.info(msg=f"[PASS] End FX Upload", extra=extra)
