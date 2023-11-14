import os
import sys

import win32com.client
import pythoncom
import time
import datetime
import sqlite3
import pandas as pd
from pandas import DataFrame, Series


path = os.path.join(os.getcwd(), 'reposit', 'nasdaq.csv')
tick_df = pd.read_csv(path, low_memory=True)[::-1].reset_index()


buy_sign = True
sell_sign = False
buy_cost = 0
sell_cost = 0
open_cost = 0
buy_date = ''
sell_date = ''
buy_time = ''
sell_time = ''
df_result = []
position = ''
save_rsi = 0
count = 1

gap = 50
half = False






for row in range(1, len(tick_df)):

    if tick_df.loc[row, 'time'] == '17:01:00':
        if count == 2:
            half = True
        open_cost = tick_df.loc[row, 'open']
        count = 1



    if count == 1 and (position == '' or position == 'long') and tick_df.loc[row, 'high'] >= open_cost + gap:


        sell_cost = open_cost + gap
        profit = 2 * (sell_cost - buy_cost)

        if half is True:
            profit = sell_cost - buy_cost


        sell_date = tick_df.loc[row , 'date']
        sell_time = tick_df.loc[row, 'time']
        df_list = [buy_date, buy_time, sell_date, sell_time, profit, buy_cost, sell_cost, position, count]
        df_result.append(df_list)

        count = 1
        position = 'short'
        half = False

    elif count == 1 and (position == '' or position == 'short') and tick_df.loc[row, 'low'] <= open_cost - gap:

        buy_cost = open_cost - gap
        profit = 2 * (sell_cost - buy_cost)
        if half is True:
            profit = sell_cost - buy_cost

        buy_date = tick_df.loc[row , 'date']
        buy_time = tick_df.loc[row, 'time']
        df_list = [sell_date, sell_time, buy_date, buy_time, profit, sell_cost, buy_cost, position, count]
        df_result.append(df_list)

        count = 1
        position = 'long'
        half = False


    elif count == 1 and position == 'long' and tick_df.loc[row , 'low'] <= open_cost - 2 * gap:


        sell_cost = open_cost - 2 * gap
        profit = 2 * (sell_cost - buy_cost)
        if half is True:
            profit = sell_cost - buy_cost

        sell_date = tick_df.loc[row, 'date']
        sell_time = tick_df.loc[row, 'time']
        df_list = [buy_date, buy_time, sell_date, sell_time, profit, buy_cost, sell_cost, position, count]
        df_result.append(df_list)

        count = 2
        position = 'short'

    elif count == 1 and position == 'short' and tick_df.loc[row, 'high'] >= open_cost + 2 * gap:


        buy_cost = open_cost + 2 * gap
        profit = 2 * (sell_cost - buy_cost)
        if half is True:
            profit = sell_cost - buy_cost

        buy_date = tick_df.loc[row, 'date']
        buy_time = tick_df.loc[row, 'time']
        df_list = [sell_date, sell_time, buy_date, buy_time, profit, sell_cost, buy_cost, position, count]
        df_result.append(df_list)

        count = 2
        position = 'long'

    elif count == 2 and position == 'long' and tick_df.loc[row, 'low'] <= open_cost + gap:


        sell_cost = open_cost + gap
        profit = sell_cost - buy_cost

        sell_date = tick_df.loc[row, 'date']
        sell_time = tick_df.loc[row, 'time']
        df_list = [buy_date, buy_time, sell_date, sell_time, profit, buy_cost, sell_cost, position, count]
        df_result.append(df_list)

        count = 1
        position = 'short'
        half = False


    elif count == 2 and position == 'short' and tick_df.loc[row, 'high'] >= open_cost - gap:


        buy_cost = open_cost - gap
        profit = sell_cost - buy_cost

        buy_date = tick_df.loc[row, 'date']
        buy_time = tick_df.loc[row, 'time']
        df_list = [sell_date, sell_time, buy_date, buy_time, profit, sell_cost, buy_cost, position, count]
        df_result.append(df_list)

        count = 1
        position = 'long'
        half = False







df = DataFrame(data=df_result, columns=['진입날짜', '진입시간', '청산날짜', '청산시간', '수익률', '매수단가', '매도단가', 'position', 'count'])
df.to_csv('reposit/nas_'+str(gap)+'.csv', index=False, encoding='euc-kr')

