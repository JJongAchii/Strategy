import pandas as pd
from datetime import datetime
from hive import db
import sqlalchemy as sa


def data_separator(meta: db.query, filename: str = "usmaster.txt"):
    def get_adj_value(meta: db.query, filename1: str = "usday.txt") -> pd.Series:
        global us_data, kr_data
        if "us" in filename1:

            # data = pd.read_csv(filename1, sep="|", names=["TRD_DT","EXC","TICKER","CLOSE_PRC","GROSS_RTN","VOLUMN","SHR_VOLUMN"])
            # for i in range(len(data["TICKER"])):
            #     data["TICKER"][i] = data["TICKER"][i].replace(" ", "")
            ticker_data = us_data[us_data["TICKER"] == meta.ticker]
            min_date = ticker_data.TRD_DT.min()
            first_day = ticker_data[ticker_data["TRD_DT"] == min_date]
            first_close_prc = first_day.CLOSE_PRC.values
            ticker_data["TRD_DT"] = pd.to_datetime(ticker_data["TRD_DT"])
            df = ticker_data.sort_values('TRD_DT', ascending=True)
            cummaltive_return = df["GROSS_RTN"].add(1).cumprod().sort_index()
            adj_value = cummaltive_return * first_close_prc

        else:
            # data = pd.read_csv(filename1, sep="|", names=["TRD_DT","TICKER","CLOSE_PRC","GROSS_RTN","VOLUMN","SHR_VOLUMN","AUM"])
            # for i in range(len(data["TICKER"])):
            #     data["TICKER"][i] = data["TICKER"][i].zfill(6)
            ticker_data = kr_data[kr_data["TICKER"] == meta.ticker]
            min_date = ticker_data.TRD_DT.min()
            first_day = ticker_data[ticker_data["TRD_DT"] == min_date]
            first_close_prc = first_day.CLOSE_PRC.values
            ticker_data["TRD_DT"] = pd.to_datetime(ticker_data["TRD_DT"])
            df = ticker_data.sort_values('TRD_DT', ascending=True)
            cummaltive_return = df["GROSS_RTN"].add(1).cumprod().sort_index()
            adj_value = cummaltive_return * first_close_prc

        return adj_value

    f = open(filename, "r")
    text_list = f.read().split("\n")

    # US TODAY(just today; one line data)
    if filename == "usmaster.txt":
        for i in text_list:
            i = i.replace(" ", "")
            ii = i.split("|")
            if meta.ticker == ii[2]:

                # tb_daily_bar
                date = int(ii[0])
                close = ii[3]
                gross_return = ii[4]
                close_1 = ii[5]
                gross_return_1 = ii[6]

                # adj_value from "usday.txt"
                adj_value = get_adj_value(meta=meta, filename1="usday.txt")

                daily_bar_today_list = [meta.stk_id, str(date), close, gross_return, adj_value[0]]
                daily_bar_yesterday_list = [meta.stk_id, str(date - 1), close_1, gross_return_1, adj_value[1]]

                # tb_meta_updat
                volumn = ii[8]
                shr_volumn = ii[9]
                aum = ii[10]

                meta_updat_list = [str(date), meta.stk_id, aum, volumn, shr_volumn]

                # pd.DataFrame -ing
                daily_bar_df = pd.DataFrame(daily_bar_today_list).T
                daily_bar_df.columns = ["STK_ID", "TRD_DT", "CLOSE_PRC", "GROSS_RTN", "ADJ_VALUE"]
                daily_bar_1_df = pd.DataFrame(daily_bar_yesterday_list).T
                daily_bar_1_df.columns = ["STK_ID", "TRD_DT", "CLOSE_PRC", "GROSS_RTN", "ADJ_VALUE"]
                meta_updat_df = pd.DataFrame(meta_updat_list).T
                meta_updat_df.columns = ["TRD_DT", "STK_ID", "AUM", "VOLUMN", "SHR_VOLUMN"]

                return daily_bar_df, daily_bar_1_df, meta_updat_df

            else:
                continue

    # KR TODAY(just today; one line data)
    if filename == "stkmaster.txt":
        for i in text_list:
            ii = i.split("|")
            if meta.ticker == ii[1]:

                # tb_daily_bar
                date = ii[0]
                close = ii[2]
                gross_return = ii[3]
                close_1 = ii[4]
                gross_return_1 = ii[5]

                # adj_value from "usday.txt"
                adj_value = get_adj_value(meta=meta, filename1="stkday.txt")

                daily_bar_today_list = [meta.stk_id, str(date), close, gross_return, adj_value[0]]
                daily_bar_yesterday_list = [meta.stk_id, str(date - 1), close_1, gross_return_1, adj_value[1]]

                # tb_meta_updat
                volumn = ii[6]
                shr_volumn = ii[7]
                aum = ii[8]

                meta_updat_list = [str(date), meta.stk_id, aum, volumn, shr_volumn]

                # pd.DataFrame -ing
                daily_bar_df = pd.DataFrame(daily_bar_today_list).T
                daily_bar_df.columns = ["STK_ID", "TRD_DT", "CLOSE_PRC", "GROSS_RTN", "ADJ_VALUE"]
                daily_bar_1_df = pd.DataFrame(daily_bar_yesterday_list).T
                daily_bar_1_df.columns = ["STK_ID", "TRD_DT", "CLOSE_PRC", "GROSS_RTN", "ADJ_VALUE"]
                meta_updat_df = pd.DataFrame(meta_updat_list).T
                meta_updat_df.columns = ["TRD_DT", "STK_ID", "AUM", "VOLUMN", "SHR_VOLUMN"]

                return daily_bar_df, daily_bar_1_df, meta_updat_df

            else:
                continue


def run_update_price2():
    global us_data, kr_data
    us_data = pd.read_csv("usday.txt", sep="|",
                          names=["TRD_DT", "EXC", "TICKER", "CLOSE_PRC", "GROSS_RTN", "VOLUMN", "SHR_VOLUMN"])
    for i in range(len(us_data["TICKER"])):
        us_data["TICKER"][i] = us_data["TICKER"][i].replace(" ", "")

    kr_data = pd.read_csv("stkday.txt", sep="|",
                          names=["TRD_DT", "TICKER", "CLOSE_PRC", "GROSS_RTN", "VOLUMN", "SHR_VOLUMN", "AUM"])
    for i in range(len(kr_data["TICKER"])):
        kr_data["TICKER"][i] = kr_data["TICKER"][i].zfill(6)

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

            # Check if close_data, gross_return_date are already updated with resent data
            # if not close_date is None and not gross_return_date is None:
            #     continue
            # if close_date >= YESTERDAY_ and gross_return_date >= YESTERDAY_:
            # continue    # if yes twice -> continue to Next Ticker

            if close_date:
                close_date = datetime.combine(close_date, datetime.min.time())

            if gross_return_date:
                gross_return_date = datetime.combine(
                    gross_return_date, datetime.min.time()
                )

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n @ ticker: ", meta.ticker, "@ ISO_CODE: ", meta.iso_code)
            if meta.iso_code == "US":
                try:
                    df1, df2, df3 = data_separator(meta=meta, filename="usmaster.txt")
                    print("\n\n *** TB_DAILY_BAR TODAY *** \n", df1)
                    print("\n *** TB_DAILY_BAR YESTERDAY *** \n", df2)
                    print("\n *** TB_META_UPDAT *** \n", df3)
                    print("###################################################################")
                except:
                    continue

            elif meta.iso_code == "KR":
                try:
                    df1, df2, df3 = data_separator(meta=meta, filename="stkmaster.txt")
                    print("\n\n *** TB_DAILY_BAR TODAY *** \n", df1)
                    print("\n *** TB_DAILY_BAR YESTERDAY *** \n", df2)
                    print("\n *** TB_META_UPDAT *** \n", df3)
                    print("###################################################################")
                except:
                    continue
            else:
                df = None


run_update_price2()
