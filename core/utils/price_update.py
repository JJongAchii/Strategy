import os
import sys
import logging
from dateutil import parser
from datetime import date, timedelta, datetime
import numpy as np
import pandas as pd
import sqlalchemy as sa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import get_args
from hive import db

logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)
YESTERDAY = TODAY - timedelta(days=1)


def run_upload_historical_data(end: datetime = datetime.today()) -> None:
    """
    upload the historical data with the data from INFOMAX

    Args:
        end (date, optional): the end date of uploading historical data. Defaults to date.today().
    """
    extra = dict(user=args.user, activity="historical_data_replacing", category="script")
    extra2 = dict(user=args.user, activity="historical_data_replacing_check", category="monitoring")
    logger.info(msg=f"[PASS] start historical data replacing. {TODAY:%Y-%m-%d}", extra=extra)
    
    global meta_data

    new_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../hive/eai/receive"))
    os.chdir(new_dir)
    sys.path.insert(0, new_dir)

    price_data_in_db = db.get_price(ticker)
    price_data_in_db.columns = ["adj_value"]
    price_data_in_db = price_data_in_db[price_data_in_db.index == price_data_in_db.index.max()]
    
    us_data = pd.read_csv("usday.txt", sep="|", names=["trd_dt","EXC","TICKER","close_prc","gross_rtn","volumn","shr_volumn"])
    us_data = us_data.drop(us_data[us_data["close_prc"]==0].index)
    us_data["TICKER"] = us_data["TICKER"].apply(lambda x: str(x).replace(" ", ""))
    us_data["trd_dt"] = pd.to_datetime(us_data["trd_dt"])
    us_data = us_data[(us_data["trd_dt"] > price_data_in_db.trd_dt.max()) & (us_data["trd_dt"] <= end)]

    kr_data = pd.read_csv("stkday.txt", sep="|", names=["trd_dt","TICKER","close_prc","gross_rtn","volumn","shr_volumn","aum"])
    kr_data = kr_data.drop(kr_data[kr_data["close_prc"]==0].index)
    kr_data["TICKER"] = kr_data["TICKER"].apply(lambda x: str(x).zfill(6))
    kr_data["trd_dt"] = pd.to_datetime(kr_data["trd_dt"])
    kr_data = kr_data[(kr_data["trd_dt"] > price_data_in_db.trd_dt.max()) & (kr_data["trd_dt"] <= end)]

    f= open("usmaster_listdate.txt", "r", encoding='utf-8')
    us_text_list = f.read().replace(" ", "").split("\n")
    f= open("stkmaster_listdate.txt", "r", encoding='utf-8')
    kr_text_list = f.read().replace(" ", "").split("\n")

    us_inception_date_table = pd.DataFrame()
    for each_line in us_text_list:
        i_list = []
        each_unit= each_line.split("|")   
        ticker = each_unit[1]
        inception_date = str(each_unit[3])
        i_list.append(ticker)
        i_list.append(inception_date)
        us_inception_date_table = pd.concat([us_inception_date_table, pd.DataFrame(i_list).T])
    us_inception_date_table.columns = ["ticker","inception_date"]

    kr_inception_date_table = pd.DataFrame()
    for each_line in kr_text_list:
        i_list = []
        each_unit= each_line.split("|")   
        ticker = each_unit[0]
        inception_date = str(each_unit[2])
        i_list.append(ticker)
        i_list.append(inception_date)
        kr_inception_date_table = pd.concat([kr_inception_date_table, pd.DataFrame(i_list).T])
    kr_inception_date_table.columns = ["ticker","inception_date"]

    def get_adj_value2(ticker:str, id:int, data:pd.DataFrame, price_data_in_db:pd.DataFrame) -> pd.DataFrame:
        """
        Calculate adjusted value and Adjust the dataframe to insert into DB

        Args:
            ticker (str): the ticker of each security
            id (int): the stk_id of each security
            data (pd.DataFrame): Total historical data of each market

        Returns:
            pd.DataFrame: which consists of the columns of tb_daily_bar and fits for inserting into DB
        """
        meta_data = data[data["TICKER"] == ticker] 
        if meta_data.empty:
            return None
        meta_data = meta_data[["trd_dt","close_prc","gross_rtn"]].reset_index().drop("index", axis=1)
        meta_data["gross_rtn"] = meta_data["gross_rtn"].apply(lambda x: x*0.01)
        meta_stk_id = pd.DataFrame(np.repeat(id, len(meta_data)), columns=["stk_id"])
        meta_data = pd.concat([meta_stk_id, meta_data], axis=1)
        cut_inception = []
        if meta.iso_code == "US":
            if meta.source == "bloomberg":
                pass
            else:
                inception_date = int(us_inception_date_table[us_inception_date_table.ticker == ticker].inception_date.values)
                if meta_data["trd_dt"].min() < inception_date:
                    meta_data = meta_data[meta_data["trd_dt"] >= inception_date]
                    cut_inception.append(meta.ticker)
                else:
                    pass
        if meta.iso_code == "KR":
            inception_date = int(kr_inception_date_table[kr_inception_date_table.ticker == ticker].inception_date.values)
            if meta_data["trd_dt"].min() < inception_date:
                meta_data = meta_data[meta_data["trd_dt"] >= inception_date]
                cut_inception.append(meta.ticker)
            else:
                pass

        meta_data["trd_dt"] = pd.to_datetime(meta_data["trd_dt"], format="%Y%m%d").apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S.%f"))
        meta_data.iloc[-1,3] = 0
        sorted_meta_data = meta_data.sort_values('trd_dt', ascending=True)
        
        meta_data["adj_value"] = 0
        if price_data_in_db.empty:
            first_close_prc = sorted_meta_data.reset_index().drop("index", axis=1).close_prc[0]
            meta_data["adj_value"] = sorted_meta_data["gross_rtn"].apply(lambda x: x + 1).cumprod().apply(lambda x: x*first_close_prc)
        else:
            last_adj_value = price_data_in_db.adj_value
            meta_data["adj_value"] = sorted_meta_data["gross_rtn"].apply(lambda x: x + 1).cumprod().apply(lambda x: x*last_adj_value)
            
        meta_data["adj_value"] = meta_data["adj_value"].apply(lambda x: 0 if abs(x) < 0.0001 else x)

        return meta_data, cut_inception[0]
    
    US_failed_ticker, KR_failed_ticker, us_cut_inception, kr_cut_inception = [], [], [], []
    for meta in db.TbMeta.query().all():
        if meta.iso_code == "US":
            if meta.source == "bloomberg":
                pass
            else:
                try:
                    meta_data,cut_inception = get_adj_value2(ticker=meta.ticker,id=meta.stk_id,data=us_data)
                    db.TbDailyBar.insert(meta_data)
                    us_cut_inception.append(cut_inception)
                except:
                    US_failed_ticker.append(meta.ticker)
                    continue

        if meta.iso_code == "KR":
            try:
                meta_data,cut_inception = get_adj_value2(ticker=meta.ticker,id=meta.stk_id,data=kr_data) 
                db.TbDailyBar.insert(meta_data) 
                kr_cut_inception.append(cut_inception)
            except:
                KR_failed_ticker.append(meta.ticker)
                continue

    # ETFs which were detected as errors
    logger.warning(msg=f"[WARN] Data cut by inception date\n\
                - US: {us_cut_inception}\n\
                - KR: {kr_cut_inception}",
                extra=extra2)
    logger.warning(msg=f"[WARN] Uploading Failed\n\
                - US: {US_failed_ticker}\n\
                - KR: {KR_failed_ticker}",
                extra=extra2)
    logger.info("[PASS] Historical price update complete.")


def data_separator(meta: db.query):
    """
    separator the file into several part and make them DataFrames for DB loading

    Args:
        meta (db.query): the entire information of each asset
        
    Returns:
        pd.DataFrame: daily_bar_df(today's price data to insert into DB),\n
                      daily_bar_1_df(yesterday's price data to update in DB),\n
                      meta_updat_df(today's volume and market cap to insert into DB),\n
                      risk_score_df(the risk level calculated by the data vendor to insert into DB)
    """
    
    def get_adj_value(meta: db.query, data:pd.DataFrame) -> pd.Series:
        """
        Get the adjusted value of the specific asset from the 'data' put in this function

        Args:
            meta (db.query): The entire information of each asset
            data (pd.DataFrame): the price data of all the assets in the long universe

        Returns:
            float: The adjusted value of each asset
        """
        ticker_data = data[data["stk_id"] == meta.stk_id]
        adj_value_df = ticker_data.adj_value.values
        adj_value = np.array2string(adj_value_df, separator=',')[1:-1]
        adj_value = 0 if float(adj_value) < 0.0001 else float(adj_value)
        return adj_value

    # US TODAY(just today; one line data)
    if meta.iso_code == "US":
        if meta.source == "bloomberg":
            pass
        else:
            for each_line in us_text_list:
                each_unit= each_line.split("|")   
                if meta.ticker == each_unit[2]: 
                    date = int(each_unit[0])
                    close = each_unit[3]
                    gross_return = each_unit[4]
                    close_1 = each_unit[5]
                    gross_return_1 = each_unit[6]
                
                    adj_value_today = (1+float(gross_return)*0.01) * get_adj_value(meta=meta, data=us_adj_value_df_1d)
                    adj_value_yesterday = (1+float(gross_return_1)*0.01) * get_adj_value(meta=meta, data=us_adj_value_df_2d)
                    daily_bar_today_list = [str(meta.stk_id), str(date), close, float(gross_return)*0.01, adj_value_today]
                    daily_bar_yesterday_list = [str(meta.stk_id), str(us_adj_value_df_1d.trd_dt[0]), close_1, float(gross_return_1)*0.01, adj_value_yesterday]
                
                    risk_score = each_unit[7]

                    risk_score_list = [str(meta.stk_id), str(date), risk_score]
                
                    trd_volume = each_unit[8]
                    trd_amount = each_unit[9]
                    aum = each_unit[10]
                
                    meta_updat_list = [str(date), str(meta.stk_id), aum, trd_volume, trd_amount]
                
                    daily_bar_df = pd.DataFrame(daily_bar_today_list).T
                    daily_bar_df.columns = ["stk_id","trd_dt","close_prc","gross_rtn", "adj_value"]
                    daily_bar_1_df = pd.DataFrame(daily_bar_yesterday_list).T
                    daily_bar_1_df.columns = ["stk_id","trd_dt","close_prc","gross_rtn", "adj_value"]
                    meta_updat_df = pd.DataFrame(meta_updat_list).T
                    meta_updat_df.columns = ["trd_dt","stk_id","aum","trd_volume","trd_amount"]
                    risk_score_df = pd.DataFrame(risk_score_list).T
                    risk_score_df.columns = ["stk_id","trd_dt","risk_score"]
                    return daily_bar_df, daily_bar_1_df, meta_updat_df, risk_score_df
                else:
                    continue
            
    # KR TODAY(just today; one line data)
    if meta.iso_code == "KR":
        for each_line in kr_text_list:
            each_unit = each_line.split("|")   
            if meta.ticker == each_unit[1]: 
                date = int(each_unit[0])
                close = each_unit[2]
                gross_return = each_unit[3]
                close_1 = each_unit[4]
                gross_return_1 = each_unit[5]

                adj_value_today = (1+float(gross_return)*0.01) * get_adj_value(meta=meta, data=kr_adj_value_df_1d)
                adj_value_yesterday = (1+float(gross_return_1)*0.01) * get_adj_value(meta=meta, data=kr_adj_value_df_2d)                
                daily_bar_today_list = [str(meta.stk_id), str(date), close, float(gross_return)*0.01, adj_value_today]
                daily_bar_yesterday_list = [str(meta.stk_id), str(kr_adj_value_df_1d.trd_dt[0]), close_1, float(gross_return_1)*0.01, adj_value_yesterday]
                
                trd_volume = each_unit[6]
                trd_amount = each_unit[7]
                aum = each_unit[8]
                
                meta_updat_list = [str(date), str(meta.stk_id), aum, trd_volume, trd_amount]

                daily_bar_df = pd.DataFrame(daily_bar_today_list).T
                daily_bar_df.columns = ["stk_id","trd_dt","close_prc","gross_rtn", "adj_value"]
                daily_bar_1_df = pd.DataFrame(daily_bar_yesterday_list).T
                daily_bar_1_df.columns = ["stk_id","trd_dt","close_prc","gross_rtn", "adj_value"]
                meta_updat_df = pd.DataFrame(meta_updat_list).T
                meta_updat_df.columns = ["trd_dt","stk_id","aum","trd_volume","trd_amount"]
                return daily_bar_df, daily_bar_1_df, meta_updat_df  
            else:
                continue

def run_update_daily_KR():
    """insert the daily KR data to DB tables"""
    extra = dict(user=args.user, activity="Daily_data_inserting", category="script")
    extra2 = dict(user=args.user, activity="Daily_data_inserting_check", category="monitoring")
    
    if db.query.check_weekend_holiday_date("KR", TODAY):
        logger.info(msg=f"[SKIP] KR daily data inserting. {TODAY:%Y-%m-%d}", extra=extra)
        return
    
    logger.info(msg=f"[PASS] start KR daily data inserting. {TODAY:%Y-%m-%d}", extra=extra)

    global kr_adj_value_df_1d, kr_adj_value_df_2d,kr_text_list

    TODAY_ = TODAY.date()
    TODAY_8 = TODAY_.strftime("%Y%m%d")

    kr_adj_value_df_1d, kr_adj_value_df_2d = db.query.get_last_two_trading_dates_price(TODAY, "KR")

    new_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../hive/eai/receive"))
    os.chdir(new_dir)
    sys.path.insert(0, new_dir)

    kr_current_file_path,kr_new_file_path = f'stk_close.{TODAY_8}', f'stk_close_{TODAY_8}.txt'

    try:
        os.rename(kr_current_file_path, kr_new_file_path)
    except:
        pass
    try:
        f= open(f"stk_close_{TODAY_8}.txt", "r")
        kr_text_list = f.read().split("\n")
    except Exception as e:
        logger.warning(msg=f"[WARN] KR - {e}", extra=extra2)
    
    with db.session_local() as session:
        for meta in db.TbMeta.query().all():
            gross_return_date = (
                session.query(sa.func.max(db.TbDailyBar.trd_dt))
                .filter(
                    db.TbDailyBar.gross_rtn.isnot(None),
                    db.TbDailyBar.stk_id == meta.stk_id,
                )
                .scalar()
            )
            close_date = (
                session.query(sa.func.max(db.TbDailyBar.trd_dt))
                .filter(
                    db.TbDailyBar.close_prc.isnot(None),
                    db.TbDailyBar.stk_id == meta.stk_id,
                )
                .scalar()
            )
            if not close_date is None and not gross_return_date is None:
                if close_date >= TODAY_ and gross_return_date >= TODAY_:
                    continue    

            daily_bar_today_df, daily_bar_lastday_df, meta_updat_df, risk_score_df = None, None, None, None
            if (meta.iso_code == "KR") & (meta.status is None):
                try:
                    daily_bar_today_df,daily_bar_lastday_df,meta_updat_df = data_separator(meta=meta)
                except:
                    print("KR |",meta.ticker, "(", meta.stk_id,")", "| This must be delisted. You better check it out")
                    logger.warning(msg=f"[WARN] KR - {meta.ticker} - Uploading Failed", extra=extra2)
                    continue

            if daily_bar_today_df is not None:
                if not daily_bar_today_df.empty:
                    try:
                        db.TbDailyBar.insert(daily_bar_today_df)
                    except:
                        try:
                            db.TbDailyBar.update(daily_bar_today_df)
                        except:
                            logger.warning(msg=f"[WARN] {meta.iso_code}-{meta.ticker}-TbDailyBar Uploading Failed", extra=extra2)
                            pass
            if daily_bar_lastday_df is not None:
                if not daily_bar_lastday_df.empty:
                    try:
                        db.TbDailyBar.update(daily_bar_lastday_df)
                    except:
                        logger.warning(msg=f"[WARN] {meta.iso_code}-{meta.ticker}-TbDailyBar Updating Failed", extra=extra2)
            if meta_updat_df is not None:
                if not meta_updat_df.empty:
                    try:
                        db.TbMetaUpdat.insert(meta_updat_df)
                    except:
                        try:
                            db.TbMetaUpdat.update(meta_updat_df)
                        except:
                            logger.warning(msg=f"[WARN] {meta.iso_code}-{meta.ticker}-TbMetaUpdat Uploading Failed", extra=extra2)
                            pass
            # if risk_score_df is not None:
            #     if not risk_score_df.empty:
            #         try:
            #            db.TbRiskScore.insert(risk_score_df)
            #         except:
            #             try:
            #                 db.TbRiskScore.update(risk_score_df)
            #             except:
            #                 logger.warning(msg=f"[WARN] {meta.iso_code}-{meta.ticker}-TbRiskScore Uploading Failed", extra=extra2)
            #                 pass
            else:
                continue

    logger.info("[PASS] KR price update complete.")    

def run_update_daily_US():
    """insert the daily US data to DB tables"""
    extra = dict(user=args.user, activity="Daily_data_inserting", category="script")
    extra2 = dict(user=args.user, activity="Daily_data_inserting_check", category="monitoring")
    
    if db.query.check_weekend_holiday_date("US", TODAY):
        logger.info(msg=f"[SKIP] US daily data inserting. {TODAY:%Y-%m-%d}", extra=extra)
        return
    
    logger.info(msg=f"[PASS] start US daily data inserting. {TODAY:%Y-%m-%d}", extra=extra)

    global us_adj_value_df_1d,us_adj_value_df_2d,us_text_list

    TODAY_ = TODAY.date()
    YESTERDAY_ = TODAY_ - timedelta(days=1)
    YESTERDAY_8 = YESTERDAY_.strftime("%Y%m%d")

    us_adj_value_df_1d, us_adj_value_df_2d = db.query.get_last_two_trading_dates_price(YESTERDAY, "US")

    new_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../hive/eai/receive"))
    os.chdir(new_dir)
    sys.path.insert(0, new_dir)

    us_current_file_path,us_new_file_path = f'gst_close.{YESTERDAY_8}', f'gst_close_{YESTERDAY_8}.txt'

    try:
        os.rename(us_current_file_path, us_new_file_path)
    except:
        pass
    try:
        f= open(f"gst_close_{YESTERDAY_8}.txt", "r")
        us_text_list = f.read().replace(" ", "").split("\n")
    except Exception as e:
        logger.warning(msg=f"[WARN] US - {e}", extra=extra2)

    with db.session_local() as session:
        for meta in db.TbMeta.query().all():
            gross_return_date = (
                session.query(sa.func.max(db.TbDailyBar.trd_dt))
                .filter(
                    db.TbDailyBar.gross_rtn.isnot(None),
                    db.TbDailyBar.stk_id == meta.stk_id,
                )
                .scalar()
            )
            close_date = (
                session.query(sa.func.max(db.TbDailyBar.trd_dt))
                .filter(
                    db.TbDailyBar.close_prc.isnot(None),
                    db.TbDailyBar.stk_id == meta.stk_id,
                )
                .scalar()
            )
            if not close_date is None and not gross_return_date is None:
                if close_date >= YESTERDAY_ and gross_return_date >= YESTERDAY_:
                    continue    
            
            daily_bar_today_df,daily_bar_lastday_df,meta_updat_df,risk_score_df = None,None,None,None
            if (meta.iso_code == "US") & (meta.status is None):
                if meta.source == "bloomberg":
                    pass
                else:
                    try:    
                        daily_bar_today_df,daily_bar_lastday_df,meta_updat_df,risk_score_df = data_separator(meta=meta)
                    except:
                        print("US |",meta.ticker,"(", meta.stk_id,")", "| This must be delisted. You better check it out")
                        logger.warning(msg=f"[WARN] US - {meta.ticker} - Uploading Failed", extra=extra2)
                        continue

            if daily_bar_today_df is not None:
                if not daily_bar_today_df.empty:
                    try:
                        db.TbDailyBar.insert(daily_bar_today_df)
                    except:
                        try:
                            db.TbDailyBar.update(daily_bar_today_df)
                        except:
                            logger.warning(msg=f"[WARN] {meta.iso_code}-{meta.ticker}-TbDailyBar Uploading Failed", extra=extra2)
                            pass
            if daily_bar_lastday_df is not None:
                if not daily_bar_lastday_df.empty:
                    try:
                        db.TbDailyBar.update(daily_bar_lastday_df)
                    except:
                        logger.warning(msg=f"[WARN] {meta.iso_code}-{meta.ticker}-TbDailyBar Updating Failed", extra=extra2)

            if meta_updat_df is not None:
                if not meta_updat_df.empty:
                    try:
                        db.TbMetaUpdat.insert(meta_updat_df)
                    except:
                        try:
                            db.TbMetaUpdat.update(meta_updat_df)
                        except:
                            logger.warning(msg=f"[WARN] {meta.iso_code}-{meta.ticker}-TbMetaUpdat Uploading Failed", extra=extra2)
                            pass
            # if risk_score_df is not None:
            #     if not risk_score_df.empty:
            #         try:
            #            db.TbRiskScore.update(risk_score_df)
            #         except:
            #             try:
            #                 db.TbRiskScore.update(meta_updat_df)
            #             except:
            #                 logger.warning(msg=f"[WARN] {meta.iso_code}-{meta.ticker}-TbRiskScore Uploading Failed", extra=extra2)
            #                 pass
            else:
                continue

    logger.info("[PASS] US price update complete.")