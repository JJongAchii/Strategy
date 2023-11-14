import numpy as np
import pandas as pd
from pandas import DataFrame
import math
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#Raw Data 끌어오기
close_df = pd.read_csv('C:/Users/user03003123/Desktop/Strategy/Backdata/20210401/수정종가.csv', low_memory=True).dropna(subset=['DATE']).drop(columns=['DATE']).astype('float32')
rlclose_df = pd.read_csv('C:/Users/user03003123/Desktop/Strategy/Backdata/20210401/종가.csv', low_memory=True).dropna(subset=['DATE']).drop(columns=['DATE']).astype('float32')
open_df = pd.read_csv('C:/Users/user03003123/Desktop/Strategy/Backdata/20210401/수정시가.csv', low_memory=True).dropna(subset=['DATE'])
high_df = pd.read_csv('C:/Users/user03003123/Desktop/Strategy/Backdata/20210401/수정고가.csv', low_memory=True).dropna(subset=['DATE']).drop(columns=['DATE']).astype('float32')
low_df = pd.read_csv('C:/Users/user03003123/Desktop/Strategy/Backdata/20210401/수정저가.csv', low_memory=True).dropna(subset=['DATE']).drop(columns=['DATE']).astype('float32')
market_df = pd.read_csv('C:/Users/user03003123/Desktop/Strategy/Backdata/20210401/시장구분.csv', low_memory=True).dropna(subset=['DATE'])
trprice_df = pd.read_csv('C:/Users/user03003123/Desktop/Strategy/Backdata/20210401/거래대금.csv', low_memory=True).dropna(subset=['DATE']).drop(columns=['DATE']).astype('float32')
RSI_df = pd.read_csv('C:/Users/user03003123/Desktop/Strategy/Backdata/20210401/RSI(2일).csv', low_memory=True)
RSI14_df = pd.read_csv('C:/Users/user03003123/Desktop/Strategy/Backdata/20210401/RSI.csv', low_memory=True)


#종가 이평선 Dataframe 생성
MA200_close_df = close_df.rolling(200).mean()
MA5_close_df = close_df.rolling(5).mean()

#종가와 5일 이평 이격도 Dataframe
far_df = (close_df - MA5_close_df) / MA5_close_df


#매매결과에 대한 Dataframe
result_df = DataFrame(columns=['날짜', '년', '월', '전략매수종목수', '실제매수종목수', '매도종목수', '일일수익률', '전략수익률', '승률', '총이익', '누적이익', '최소보유일', '최대보유일', '주문종목', '종목코드', '실제 순자산증가', '장종료후 순자산(현금+평가금액)', '보유종목수', '보유종목코드', '실제수익률', '실제수익금', '수익률변화율', 'MDD금액', 'MDD'])
result_df['날짜'] = open_df['DATE']
result_df.fillna(0, inplace=True)




#Dataframe 내 칼럼순서에 맞게 리스트 생성
buy_sign = [True for _ in close_df.columns]
buy_con = [False for _ in close_df.columns]
sell_sign = [False for _ in close_df.columns]
buy_cost = [0 for _ in close_df.columns]
sell_cost = [0 for _ in close_df.columns]
save_row = [0 for _ in close_df.columns]
cost_forstock = [0 for _ in close_df.columns]
stock_num = [0 for _ in close_df.columns]
have_stock = []
result_list = []
pre_havestock = 0
first_buy = False
first_row = 0
capital = 50000000


#호가 규칙 적용
def cost_rule(cost, market):
    if cost < 1000:  # 1원

        cost = int(cost)

    elif cost >= 1000 and cost < 5000:  # 5원

        if cost % 10 < 5:
            cost = int(cost - cost % 10)
        else:
            cost = int(cost - cost % 10 + 5)

    elif cost >= 5000 and cost < 10000:  # 10원

        cost = int(cost - cost % 10)

    elif cost >= 10000 and cost < 50000:

        if cost % 100 < 50:
            cost = int(cost - cost % 100)
        else:
            cost = int(cost - cost % 100 + 50)

    elif cost >= 50000 and cost < 100000:

        cost = int(cost - cost % 100)

    elif cost >= 100000 and cost < 500000 and market == 'KS':

        if cost % 1000 < 500:
            cost = int(cost - cost % 1000)
        else:
            cost = int(cost - cost % 1000 + 500)

    elif cost >= 100000 and cost < 500000 and market == 'KQ':

        cost = int(cost - cost % 100)

    elif cost >= 500000:

        cost = int(cost - cost % 1000)

    return cost


for row in range(1, len(trprice_df)):
    print(row)
    buy_count = 0
    col_count = 0
    sell_count = 0
    order_list = ''
    stock_list = ''
    have_list = ''

#종가와 5일 이평 이격도 오름차순 정렬
    far_df = far_df.sort_values(by=row-1, axis=1)

    price = 50000000 / 25


    for col in far_df.columns:
        col_num = trprice_df.columns.get_loc(col)
        if buy_sign[col_num] == True and close_df.loc[row - 1, col] > MA200_close_df.loc[row - 1, col] and close_df.loc[row - 1, col] < MA5_close_df.loc[row - 1, col] and RSI14_df.loc[row - 1, col] < 50 and RSI_df.loc[row - 2, col] > 5 and RSI_df.loc[row - 1, col] < 5 and trprice_df.loc[row - 1, col] > 5000000000 and pd.isnull(trprice_df.loc[row, col]) == False and trprice_df.loc[row, col] != 0 and col_count < 25 - pre_havestock:
            col_count += 1

            mod = close_df.loc[row, col] / rlclose_df.loc[row, col]
            buy_cost[col_num] = close_df.loc[row - 1, col] * 0.97
            order_list = order_list + col + ', '

            if buy_cost[col_num] >= open_df.loc[row, col]:

                buy_count += 1

                stock_num[col_num] = int(price / cost_rule(open_df.loc[row, col] / mod, market_df.loc[row, col])) / mod
                buy_cost[col_num] = cost_rule(open_df.loc[row, col] / mod, market_df.loc[row, col]) * mod

                capital = capital - (buy_cost[col_num] * stock_num[col_num] * 1.001)

                sell_sign[col_num] = True
                save_row[col_num] = row
                buy_sign[col_num] = False
                have_stock.append(col)
                stock_list = stock_list + col + ', '


            elif buy_cost[col_num] < open_df.loc[row, col] and buy_cost[col_num] >= low_df.loc[row, col]:

                buy_count += 1

                stock_num[col_num] = int(price / cost_rule(close_df.loc[row - 1, col] * 0.97 / mod, market_df.loc[row, col])) / mod
                buy_cost[col_num] = cost_rule(close_df.loc[row - 1, col] * 0.97 / mod, market_df.loc[row, col]) * mod


                capital = capital - (buy_cost[col_num] * stock_num[col_num] * 1.001)

                sell_sign[col_num] = True
                save_row[col_num] = row
                buy_sign[col_num] = False
                have_stock.append(col)
                stock_list = stock_list + col + ', '

            if buy_count > 0 and first_buy == False:

                first_row = row
                first_buy = True



        elif sell_sign[col_num] == True and ((RSI_df.loc[row - 1, col] > 5) or row - save_row[col_num] == 5) and pd.isnull(trprice_df.loc[row, col]) == False and trprice_df.loc[row, col] != 0:

            sell_count += 1

            hold = row - save_row[col_num]

            mod = close_df.loc[row, col] / rlclose_df.loc[row, col]
            sell_cost[col_num] = close_df.loc[row - 1, col]

            if sell_cost[col_num] <= open_df.loc[row, col]:

                sell_cost[col_num] = cost_rule(open_df.loc[row, col] / mod, market_df.loc[row, col]) * mod

            elif sell_cost[col_num] > open_df.loc[row, col] and sell_cost[col_num] <= high_df.loc[row, col]:

                sell_cost[col_num] = cost_rule(close_df.loc[row - 1, col] / mod, market_df.loc[row, col]) * mod

            elif sell_cost[col_num] > high_df.loc[row, col]:

                sell_cost[col_num] = cost_rule(close_df.loc[row, col] / mod, market_df.loc[row, col]) * mod



            profit_rate = (sell_cost[col_num] * 0.9965 - buy_cost[col_num] * 1.001) / (buy_cost[col_num] * 1.001)
            profit = (sell_cost[col_num] * 0.9965 - buy_cost[col_num] * 1.001) * stock_num[col_num]

            info_list = [open_df.loc[save_row[col_num], 'DATE'], col, profit_rate, profit,
                         RSI_df.loc[save_row[col_num] - 1, col],
                         RSI14_df.loc[save_row[col_num] - 1, col], close_df.loc[save_row[col_num] - 1, col],
                         MA5_close_df.loc[save_row[col_num] - 1, col], MA200_close_df.loc[save_row[col_num] - 1, col],
                         open_df.loc[save_row[col_num] - 1, col], high_df.loc[save_row[col_num] - 1, col],
                         low_df.loc[save_row[col_num] - 1, col],
                         trprice_df.loc[save_row[col_num] - 1, col], hold]

            result_list.append(info_list)



            result_df.loc[save_row[col_num], '총이익'] = result_df.loc[save_row[col_num], '총이익'] + (
                    sell_cost[col_num] * 0.9965 - buy_cost[col_num] * 1.001) * stock_num[col_num]
            result_df.loc[save_row[col_num], '일일수익률'] = result_df.loc[save_row[col_num], '일일수익률'] + (
                    sell_cost[col_num] * 0.9965 - buy_cost[col_num] * 1.001) / (buy_cost[col_num] * 1.001)


            if sell_cost[col_num] * 0.9965 > buy_cost[col_num] * 1.001:
                result_df.loc[save_row[col_num], '승률'] = result_df.loc[save_row[col_num], '승률'] + 1

            if hold > result_df.loc[save_row[col_num], '최대보유일']:
                result_df.loc[save_row[col_num], '최대보유일'] = hold

            if result_df.loc[save_row[col_num], '최소보유일'] == 0 or hold < result_df.loc[save_row[col_num], '최소보유일']:
                result_df.loc[save_row[col_num], '최소보유일'] = hold

            capital = capital + (sell_cost[col_num] * stock_num[col_num] * 0.9965)

            sell_sign[col_num] = False
            buy_sign[col_num] = True
            have_stock.remove(col)


        elif sell_sign[col_num] == True and trprice_df.loc[row - 1, col] == 0 and trprice_df.loc[row, col] != 0 and pd.isnull(trprice_df.loc[row, col]) == False:

            sell_count += 1

            hold = row - save_row[col_num]

            mod = close_df.loc[row, col] / rlclose_df.loc[row, col]
            sell_cost[col_num] = cost_rule(open_df.loc[row, col] / mod, market_df.loc[row, col]) * mod



            profit_rate = (sell_cost[col_num] * 0.9965 - buy_cost[col_num] * 1.001) / (buy_cost[col_num] * 1.001)
            profit = (sell_cost[col_num] * 0.9965 - buy_cost[col_num] * 1.001) * stock_num[col_num]

            info_list = [open_df.loc[save_row[col_num], 'DATE'], col, profit_rate, profit,
                         RSI_df.loc[save_row[col_num] - 1, col],
                         RSI14_df.loc[save_row[col_num] - 1, col], close_df.loc[save_row[col_num] - 1, col],
                         MA5_close_df.loc[save_row[col_num] - 1, col], MA200_close_df.loc[save_row[col_num] - 1, col],
                         open_df.loc[save_row[col_num] - 1, col], high_df.loc[save_row[col_num] - 1, col],
                         low_df.loc[save_row[col_num] - 1, col], 
                         trprice_df.loc[save_row[col_num] - 1, col], hold]

            result_list.append(info_list)



            result_df.loc[save_row[col_num], '총이익'] = result_df.loc[save_row[col_num], '총이익'] + (
                    sell_cost[col_num] * 0.9965 - buy_cost[col_num] * 1.001) * stock_num[col_num]
            result_df.loc[save_row[col_num], '일일수익률'] = result_df.loc[save_row[col_num], '일일수익률'] + (
                    sell_cost[col_num] * 0.9965 - buy_cost[col_num] * 1.001) / (buy_cost[col_num] * 1.001)

            if sell_cost[col_num] * 0.9965 > buy_cost[col_num] * 1.001:
                result_df.loc[save_row[col_num], '승률'] = result_df.loc[save_row[col_num], '승률'] + 1

            if hold > result_df.loc[save_row[col_num], '최대보유일']:
                result_df.loc[save_row[col_num], '최대보유일'] = hold

            if result_df.loc[save_row[col_num], '최소보유일'] == 0 or hold < result_df.loc[save_row[col_num], '최소보유일']:
                result_df.loc[save_row[col_num], '최소보유일'] = hold


            capital = capital + (sell_cost[col_num] * stock_num[col_num] * 0.9965)

            sell_sign[col_num] = False
            buy_sign[col_num] = True
            have_stock.remove(col)




        elif sell_sign[col_num] == True and pd.isnull(trprice_df.loc[row, col]) == True:
            hold = -1

            sell_cost[col_num] = 0

            result_df.loc[save_row[col_num], '총이익'] = result_df.loc[save_row[col_num], '총이익'] + (
                    sell_cost[col_num] * 0.9965 - buy_cost[col_num] * 1.001) * stock_num[col_num]
            result_df.loc[save_row[col_num], '일일수익률'] = result_df.loc[save_row[col_num], '일일수익률'] + (
                    sell_cost[col_num] * 0.9965 - buy_cost[col_num] * 1.001) / (buy_cost[col_num] * 1.001)

            profit_rate = (sell_cost[col_num] * 0.9965 - buy_cost[col_num] * 1.001) / (buy_cost[col_num] * 1.001)
            profit = (sell_cost[col_num] * 0.9965 - buy_cost[col_num] * 1.001) * stock_num[col_num]



            info_list = [open_df.loc[save_row[col_num], 'DATE'], col, profit_rate, profit,
                         RSI_df.loc[save_row[col_num] - 1, col],
                         RSI14_df.loc[save_row[col_num] - 1, col], close_df.loc[save_row[col_num] - 1, col],
                         MA5_close_df.loc[save_row[col_num] - 1, col], MA200_close_df.loc[save_row[col_num] - 1, col],
                         open_df.loc[save_row[col_num] - 1, col], high_df.loc[save_row[col_num] - 1, col],
                         low_df.loc[save_row[col_num] - 1, col], 
                         trprice_df.loc[save_row[col_num] - 1, col], hold]

            result_list.append(info_list)



            if sell_cost[col_num] * 0.9965 > buy_cost[col_num] * 1.001:
                result_df.loc[save_row[col_num], '승률'] = result_df.loc[save_row[col_num], '승률'] + 1

            if hold > result_df.loc[save_row[col_num], '최대보유일']:
                result_df.loc[save_row[col_num], '최대보유일'] = hold

            if result_df.loc[save_row[col_num], '최소보유일'] == 0 or hold < result_df.loc[save_row[col_num], '최소보유일']:
                result_df.loc[save_row[col_num], '최소보유일'] = hold

            sell_sign[col_num] = False
            buy_sign[col_num] = False
            have_stock.remove(col)

        if col in have_stock:
            result_df.loc[row, '장종료후 순자산(현금+평가금액)'] = result_df.loc[row, '장종료후 순자산(현금+평가금액)'] + (close_df.loc[row, col] * stock_num[col_num] * 0.9965)
            have_list = have_list + col + ', '


    result_df.loc[row, '전략매수종목수'] = col_count
    result_df.loc[row, '실제매수종목수'] = buy_count
    result_df.loc[row, '매도종목수'] = sell_count
    result_df.loc[row, '주문종목'] = order_list
    result_df.loc[row, '종목코드'] = stock_list
    result_df.loc[row, '장종료후 순자산(현금+평가금액)'] = result_df.loc[row, '장종료후 순자산(현금+평가금액)'] + capital
    result_df.loc[row, '보유종목수'] = len(have_stock)
    result_df.loc[row, '보유종목코드'] = have_list
    pre_havestock = len(have_stock)


result_df['년'] = pd.DatetimeIndex(result_df['날짜']).year
result_df['월'] = pd.DatetimeIndex(result_df['날짜']).month
result_df['일일수익률'] = (result_df['일일수익률'] / result_df['실제매수종목수']).astype(float).round(4)
result_df['전략수익률'] = (result_df['총이익'] / 50000000).astype(float).round(4)
result_df['승률'] = (result_df['승률'] / result_df['실제매수종목수']).astype(float).round(4)
result_df['누적이익'] = result_df['총이익'].rolling(window=len(result_df), min_periods=1).sum()
result_df.loc[0, '장종료후 순자산(현금+평가금액)'] = result_df.loc[1, '장종료후 순자산(현금+평가금액)']
result_df['실제 순자산증가'] = result_df['장종료후 순자산(현금+평가금액)'].diff()
result_df['실제수익금'] = result_df['장종료후 순자산(현금+평가금액)'] - 50000000
result_df['실제수익률'] = (result_df['실제수익금'] / 50000000).astype(float).round(4)
result_df['수익률변화율'] = (1 + result_df['실제수익률']) / (1 + result_df['실제수익률'].shift(1))
result_df['MDD금액'] = result_df['실제수익금'] - result_df['실제수익금'].cummax()
result_df['MDD'] = (result_df['MDD금액'] / 50000000).astype(float).round(4)


result_df = result_df.loc[first_row:,:]

result_df.to_csv('C:/Users/user03003123/Desktop/Strategy/Backdata/20210401/RSI_역추세돌파매도(25종목)_20210401.csv', index=False, encoding='euc-kr')
list_df = DataFrame(data=result_list, columns=['날짜', '종목코드', '수익율', '수익금', 'RSI2', 'RSI14', '종가', '5이평', '200일이평', '시가', '고가', '저가', '시가총액', '거래대금', '보유일'])
list_df.to_csv('C:/Users/user03003123/Desktop/Strategy/Backdata/20210401/RSI_역추세돌파매도(25종목)_20210401_매매데이터.csv', index=False, encoding='euc-kr')


#매매결과에 대한 최종 HTML 생성
result_df['일'] = pd.DatetimeIndex(result_df['날짜']).day

print(result_df)
print(result_df['일'])
print(result_df['월'])
print(result_df['년'])

#year1 = result_df[(result_df.loc[:,'년'] == result_df.loc[result_df.index[-1], '년'] - 1) & (result_df.loc[:,'월'] == result_df.loc[result_df.index[-1], '월']) & (result_df.loc[:,'일'] > result_df.loc[result_df.index[-1], '일'])].index[0]
#year3 = result_df[(result_df.loc[:,'년'] == result_df.loc[result_df.index[-1], '년'] - 3) & (result_df.loc[:,'월'] == result_df.loc[result_df.index[-1], '월']) & (result_df.loc[:,'일'] > result_df.loc[result_df.index[-1], '일'])].index[0]
#year5 = result_df[(result_df.loc[:,'년'] == result_df.loc[result_df.index[-1], '년'] - 5) & (result_df.loc[:,'월'] == result_df.loc[result_df.index[-1], '월']) & (result_df.loc[:,'일'] > result_df.loc[result_df.index[-1], '일'])].index[0]

year1 = result_df[(result_df.loc[:,'년'] == result_df.loc[result_df.index[-1], '년']) & (result_df.loc[:,'월'] == 1)].index[0]
year3 = result_df[(result_df.loc[:,'년'] == result_df.loc[result_df.index[-1], '년'] - 2) & (result_df.loc[:,'월'] == 1)].index[0]
year5 = result_df[(result_df.loc[:,'년'] == result_df.loc[result_df.index[-1], '년'] - 4) & (result_df.loc[:,'월'] == 1)].index[0]

year1_df = result_df.loc[year1:,:]
year3_df = result_df.loc[year3:,:]
year5_df = result_df.loc[year5:,:]

year1_df['MDD금액'] = year1_df['실제수익금'] - year1_df['실제수익금'].cummax()
year1_df['MDD'] = (year1_df['MDD금액'] / 50000000).astype(float).round(4)

year3_df['MDD금액'] = year3_df['실제수익금'] - year3_df['실제수익금'].cummax()
year3_df['MDD'] = (year3_df['MDD금액'] / 50000000).astype(float).round(4)

year5_df['MDD금액'] = year5_df['실제수익금'] - year5_df['실제수익금'].cummax()
year5_df['MDD'] = (year5_df['MDD금액'] / 50000000).astype(float).round(4)

cagr = (result_df.loc[result_df.index[-1], '실제수익률']) / (len(result_df) / 250)
stdev = result_df['수익률변화율'].std() * 250 ** 0.5
sharp = round((cagr / stdev), 2)
MDD = str(round(result_df['MDD'].min() * 100, 2)) + "%"


win_rate = round((list_df.loc[list_df.loc[:, '수익금'] > 0, '수익금'].count() / len(list_df)) * 100, 2)
pro_rate = round((list_df.loc[list_df.loc[:, '수익금'] > 0, '수익금'].sum() / (0 - list_df.loc[list_df.loc[:, '수익금'] < 0, '수익금'].sum())), 2)
exp_val = round((pro_rate * win_rate / 100) - (1 - (win_rate / 100)), 2)


year_df = (result_df.groupby(['년']).agg({'실제 순자산증가' : 'sum'}) / 50000000 * 100).astype(float).round(2).reset_index()

month_df = DataFrame(columns=year_df['년'])

month_graph = (result_df.groupby(['년', '월']).agg({'실제 순자산증가' : 'sum'}) / 50000000 * 100).astype(float).round(2).reset_index()
month_graph['Rolling'] = month_graph['실제 순자산증가'].rolling(12).sum()

month_graph['년-월'] = month_graph['년'].astype(str) + '-' + month_graph['월'].astype(str)



for row in range(12):
    for col in month_df.columns:
        month_df.loc[row, col] = ((result_df.loc[(result_df.loc[:,'년'] == col) & (result_df.loc[:,'월'] == row + 1), '실제 순자산증가'].sum() / 50000000 * 100).astype(float).round(2))

month_df['월 평균'] = month_df.mean(axis=1).round(2)
month_df = month_df.astype(str) + "%"
month_df.insert(0, '월', [1,2,3,4,5,6,7,8,9,10,11,12])
month_df = month_df.T

print(month_df)


fig = make_subplots(
    rows=10, cols=1,
    vertical_spacing=0.03,
    specs=[[{"type": "table"}],
           [{"type": "scatter"}],
           [{"type": "scatter"}],
           [{"type": "table"}],
           [{"type": "scatter"}],
           [{"type": "table"}],
           [{"type": "scatter"}],
           [{"type": "table"}],
           [{"type": "table"}],
           [{"type": "table"}]],
    subplot_titles=('', '수익률 차트', 'MDD 차트', '', '월별 수익률 차트', '월간수익률', '1년 Rolling 차트',  '', '',''),
)


fig.add_trace(
    go.Table(
        header=dict(
            values=["<b>전체기간 결과표</b>", ""],
            font=dict(size=20),
            align="left",
            height = 50,
        ),
        cells=dict(
            values=[["CAGR", "Stdev", "Sharp", "MDD", "", "승률", "손익비", "기대값"], [str(round(cagr * 100, 2)) + "%", str(round(stdev * 100, 2)) + "%", sharp, MDD, "", str(win_rate) + "%", pro_rate, exp_val]],
            align = "left",
            height = 30
        ),
    ),
    row=1, col=1
)





fig.add_trace(
    go.Scatter(
        x=result_df['날짜'],
        y=(result_df['실제수익률'] * 100).round(2),
        mode="lines",
        name='수익률'
    ),
    row=2, col=1
)


fig.add_trace(
    go.Scatter(
        x=result_df['날짜'],
        y=(result_df['MDD'] * 100).round(2),
        mode="lines",
        name='MDD'
    ),
    row=3, col=1
)


fig.add_trace(
    go.Table(
        header=dict(
            values=["<b>연도</b>", "<b>연간수익률</b>"],
            font=dict(size=20),
            align="left",
            height = 50
        ),
        cells=dict(
            values=[year_df['년'], year_df['실제 순자산증가'].astype(str) + "%"],
            align = "left")
    ),
    row=4, col=1
)


fig.add_trace(
    go.Bar(
        x=month_graph['년-월'],
        y=(month_graph['실제 순자산증가']),
    ),
    row=5, col=1
)


fig.add_trace(
    go.Table(
        header=dict(
            values= ['<b>월 / 연도</b>'] + year_df['년'].tolist() + ['월 평균'],
            font=dict(size=15),
            align="left",
            height=50
        ),
        cells=dict(
            values= month_df,
            font_color= ['red' if val < 0 else 'black' for val in month_df],
            align = "left")
    ),
    row=6, col=1
)


fig.add_trace(
    go.Scatter(
        x=month_graph['년-월'],
        y=(month_graph['Rolling']),
        mode="lines",
        name='1년 Rolling 차트'
    ),
    row=7, col=1
)




cagr1 = (year1_df.loc[year1_df.index[-1], '실제수익률'] - year1_df.loc[year1_df.index[0], '실제수익률']) / 1
stdev1 = year1_df.loc[1:,'수익률변화율'].std() * len(year1_df) ** 0.5
sharp1 = round((cagr1 / stdev1), 2)
MDD1 = str(round(year1_df['MDD'].min() * 100, 2)) + "%"

fig.add_trace(
    go.Table(
        header=dict(
            values=["<b>최근 1년 결과표</b>", ""],
            font=dict(size=20),
            align="left",
            height = 50,
        ),
        cells=dict(
            values=[["CAGR", "Stdev", "Sharp", "MDD"], [str(round(cagr1 * 100, 2)) + "%", str(round(stdev1 * 100, 2)) + "%", sharp1, MDD1]],
            align = "left",
            height = 30
        ),
    ),
    row=8, col=1
)

cagr3 = (year3_df.loc[year3_df.index[-1], '실제수익률'] - year3_df.loc[year3_df.index[0], '실제수익률']) / 3
stdev3 = year3_df.loc[1:,'수익률변화율'].std() * (len(year3_df) / 3) ** 0.5
sharp3 = round((cagr3 / stdev3), 2)
MDD3 = str(round(year3_df['MDD'].min() * 100, 2)) + "%"

fig.add_trace(
    go.Table(
        header=dict(
            values=["<b>최근 3년 결과표</b>", ""],
            font=dict(size=20),
            align="left",
            height = 50,
        ),
        cells=dict(
            values=[["CAGR", "Stdev", "Sharp", "MDD"], [str(round(cagr3 * 100, 2)) + "%", str(round(stdev3 * 100, 2)) + "%", sharp3, MDD3]],
            align = "left",
            height = 30
        ),
    ),
    row=9, col=1
)

cagr5 = (year5_df.loc[year5_df.index[-1], '실제수익률'] - year5_df.loc[year5_df.index[0], '실제수익률']) / 5
stdev5 = year5_df.loc[1:,'수익률변화율'].std() * (len(year5_df) / 5) ** 0.5
sharp5 = round((cagr5 / stdev5), 2)
MDD5 = str(round(year5_df['MDD'].min() * 100, 2)) + "%"

fig.add_trace(
    go.Table(
        header=dict(
            values=["<b>최근 5년 결과표</b>", ""],
            font=dict(size=20),
            align="left",
            height = 50,
        ),
        cells=dict(
            values=[["CAGR", "Stdev", "Sharp", "MDD"], [str(round(cagr5 * 100, 2)) + "%", str(round(stdev5 * 100, 2)) + "%", sharp5, MDD5]],
            align = "left",
            height = 30
        ),
    ),
    row=10, col=1
)


fig.update_layout(
    height=5000,
    showlegend=False,
    title_text="BACKTEST REPORT",
    titlefont=dict(size=30)
)



plotly.offline.plot(fig, filename='C:/Users/user03003123/Desktop/Strategy/Backdata/20210401/Backtest_Report_20210401.html')