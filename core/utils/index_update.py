"""
Daily Bloomberg Index Data
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
extra = dict(user=args.user, activity="Daily_data_inserting_check", category="script")
extra2 = dict(user=args.user, activity="Daily_data_inserting_check", category="monitoring")
####################################################################################################

if TODAY.date().weekday() == 6:
    pass
    
else:
    # World-wide Bond indeces#####################################################################
    new_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../hive/eai/receive/bloomberg"))
    os.chdir(new_dir)
    sys.path.insert(0, new_dir)
    try:
        bb_index1 = pd.read_csv(f"Bloomberg_Bond_Index{YESTERDAY_8}.dat")
        bb_index1 = bb_index1[["IndexValue","BBGTicker"]]
        bb_index1['trd_dt'] = YESTERDAY_8
        bb_index1.columns =['value','ticker','trd_dt']
    except:
        print(f"No file named Bloomberg_Bond_Index{YESTERDAY_8}.dat")
        logger.warning(msg=f"[WARN] No file named Bloomberg_Bond_Index{YESTERDAY_8}.dat", extra=extra2)

    # General indeces and Macro indicators#########################################################
    file_name = f"Bloomberg_Index{TODAY_8}"
    try:
        try:
            f = open(file_name+'.dat', 'r').read()
            f = f[f.find("START-OF-DATA")+len("START-OF-DATA")+1:f.find("END-OF-DATA")]
            file = open(file_name+".txt", 'w').write(f)
            bb_index2 = pd.read_csv(file_name+".txt", sep="|", header=None).iloc[:,[0,3,11]].drop_duplicates()
            bb_index2.columns = ["ticker", "value", "trd_dt"]
        except:
            bb_index2 = pd.read_csv(file_name+".txt", sep="|").iloc[:,[0,3,11]].drop_duplicates()
            bb_index2.columns = ["ticker", "value", "trd_dt"]
    except:
        print(f"No file named Bloomberg_Index{TODAY_8}")
        logger.warning(msg=f"[WARN] No file named Bloomberg_Index{TODAY_8}", extra=extra2)


    # Long_term TIPS ##########################################################################
    try:
        bb_US_LT_TIPS = pd.read_csv(f"Bloomberg_Bond_B_Index{YESTERDAY_8}.dat")[["TRI"]]
        bb_US_LT_TIPS["ticker"] = "BCIT5T"
        macro_id = 11
        value = bb_US_LT_TIPS.TRI[0]
        b_list = [macro_id,YESTERDAY_,value,value]
        b_df = pd.DataFrame(b_list).T
        b_df.columns = ['macro_id','trd_dt','value','adj_value']
        db.TbMacroData.insert(b_df)
    except:
        try:
            bb_US_LT_TIPS = pd.read_csv(f"Bloomberg_Bond_B_Index{YESTERDAY_8}.dat")[["TRI"]]
            bb_US_LT_TIPS["ticker"] = "BCIT5T"
            macro_id = 11
            value = bb_US_LT_TIPS.TRI[0]
            b_list = [macro_id,YESTERDAY_,value,value]
            b_df = pd.DataFrame(b_list).T
            b_df.columns = ['macro_id','trd_dt','value','adj_value']
            db.TbMacroData.update(b_df)
        except:
            print("Bloomberg @ Ticker: BCIT5T", "Error Occured | You'd better check it out")
            logger.warning(msg=f"[WARN] No file named Bloomberg_Bond_B_Index{YESTERDAY_8}\n\
                        - Failed Index : BCIT5T", extra=extra2)


    #  Each country's representative Index#########################################################
    new_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../hive/eai/receive/Index"))
    os.chdir(new_dir)
    sys.path.insert(0, new_dir)
    idx_current_file_path,idx_new_file_path = f'Index_close_{YESTERDAY_8}', f'Index_close_{YESTERDAY_8}.txt'
    try:
        os.rename(idx_current_file_path, idx_new_file_path)
    except:
        pass

    try:
        bb_index3 = pd.read_csv(f"Index_close_{YESTERDAY_8}.txt", sep="|",encoding="euc-kr",names=['market','representative_index','trd_dt','ticker','name','value','prc_change','gross_rtn'])
        bb_index3 = bb_index3[['trd_dt','ticker','name','value']]
    except:
        print(f"No file named Index_close_{YESTERDAY_8}")
        logger.warning(msg=f"[WARN] No file named Index_close_{YESTERDAY_8}", extra=extra2)

    ##################################################################################################
    def uploader(filedata, macro_ticker,macro_future,macro_id):
        """
        Insert/Update the index data into/in DB by checking the ticker and macro_id of each index

        Args:
            filedata (pd.DataFrame): The index data fed
            macro_ticker (str): The legacy ticker of each index
            macro_future (str): The future ticker of each index
            macro_id (int): The macro_id of each index

        Returns:
            str: return its ticker if there has been any error while inserting or updating
        """
        if macro_id in [3,8,9,10,13,15,17,18,19,20,21,22,24,25,28,29,35,37,38,40,41,42,43,44,45,46,50,55]:

            if macro_ticker in filedata.ticker.tolist():
                data = filedata[filedata.ticker == macro_ticker][["trd_dt","value","value"]].reset_index().drop("index", axis=1)
                
                if macro_id in [42,43,44,45]:
                    data.trd_dt[0] = str(data.trd_dt[0])
                
                data.trd_dt = pd.to_datetime(data.trd_dt)
                    
                current_day = data.trd_dt[0] - timedelta(days=1)
                if current_day.weekday() >= 5:
                    while current_day.weekday() >= 5:
                        current_day -= timedelta(days=1)
                        
                data['trd_dt'] = current_day
                data['macro_id'] = macro_id
                data.columns = ["trd_dt","value","adj_value","macro_id"]
                data = data[["macro_id","trd_dt","value","adj_value"]]
                if data.empty:
                    pass
                else:
                    try:
                        db.TbMacroData.insert(data)
                    except:
                        try:
                            db.TbMacroData.update(data)
                        except:
                            print("Bloomberg @ Ticker:", macro_ticker, "Error Occured | You'd better check it out")
                            return macro_ticker


            elif macro_future in filedata.ticker.tolist():
                data = filedata[filedata.ticker == macro_future][["trd_dt","value","value"]].reset_index().drop("index", axis=1)
                
                if macro_id in [42,43,44,45]:
                    data.trd_dt[0] = str(data.trd_dt[0])
                
                data.trd_dt = pd.to_datetime(data.trd_dt)
                
                current_day = data.trd_dt[0] - timedelta(days=1)
                if current_day.weekday() >= 5:
                    while current_day.weekday() >= 5:
                        current_day -= timedelta(days=1)
                
                data['trd_dt'] = current_day
                data['macro_id'] = macro_id
                data.columns = ["trd_dt","value","adj_value","macro_id"]
                data = data[["macro_id","trd_dt","value","adj_value"]]
                if data.empty:
                    pass
                else:
                    try:
                        db.TbMacroData.insert(data)
                    except:
                        try:
                            db.TbMacroData.update(data)
                        except:
                            print("Bloomberg @ Ticker:", macro_future, "Error Occured | You'd better check it out")
                            return macro_future
                        
        else:
            if macro_ticker in filedata.ticker.tolist():
                data = filedata[filedata.ticker == macro_ticker][["trd_dt","value","value"]].reset_index().drop("index", axis=1)
                data['macro_id'] = macro_id
                data.columns = ["trd_dt","value","adj_value","macro_id"]
                data = data[["macro_id","trd_dt","value","adj_value"]]
                if data.empty:
                    pass
                else:
                    try:
                        db.TbMacroData.insert(data)
                    except:
                        try:
                            db.TbMacroData.update(data)
                        except:
                            print("Bloomberg @ Ticker:", macro_ticker, "Error Occured | You'd better check it out")
                            return macro_ticker


            elif macro_future in filedata.ticker.tolist():
                data = filedata[filedata.ticker == macro_future][["trd_dt","value","value"]].reset_index().drop("index", axis=1)
                data['macro_id'] = macro_id
                data.columns = ["trd_dt","value","adj_value","macro_id"]
                data = data[["macro_id","trd_dt","value","adj_value"]]
                if data.empty:
                    pass
                else:
                    try:
                        db.TbMacroData.insert(data)
                    except:
                        try:
                            db.TbMacroData.update(data)
                        except:
                            print("Bloomberg @ Ticker:", macro_future, "Error Occured | You'd better check it out")
                            return macro_future
                        
##################################################################################################
def run_update_index():
    """
    Insert/Update all the index data from diffrent files fed
    """
    if TODAY.date().weekday() == 6:
        logger.info(msg=f"[SKIP] Index Upload. {TODAY:%Y-%m-%d}", extra=extra)
        return
        
    else:
        logger.info(msg=f"[PASS] start Index Upload", extra=extra)
    
    with db.session_local() as session:
        failed_tickers = []
        for macro in db.TbMacro.query().all():
            macro_ticker = macro.ticker if macro.factor != "currency" else macro.future_ticker
            macro_future = macro.future_ticker.split(" ")[0] if macro.future_ticker else macro.future_ticker
            macro_id = macro.macro_id
            
            if macro.memo == 'FX':
                continue

            try:
                ticker= uploader(bb_index1, macro_ticker, macro_future, macro_id)
                if ticker is not None:
                    failed_tickers.append(ticker)
            except:
                pass
            
            try:
                ticker= uploader(bb_index2, macro_ticker, macro_future, macro_id)
                if ticker is not None:
                    failed_tickers.append(ticker)
            except:
                pass
            
            try:
                ticker= uploader(bb_index3, macro_ticker, macro_future, macro_id)
                if ticker is not None:
                    failed_tickers.append(ticker)
            except:
                pass
            
        logger.warning(msg=f"[WARN] Index Uploading Fail\nFailed Ticker: {failed_tickers}", extra=extra2)
        logger.info(msg=f"[PASS] Index Upload", extra=extra)