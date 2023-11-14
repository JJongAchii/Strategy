import pandas as pd
import numpy as np
from hive import db
from datetime import date
from shgo import shgo
import matplotlib.pyplot as plt


def get_port_nav_daily_returns():
    """
    Datadase 에서 portfolio nav를 가지고 와서 각 daily return을 계산한다.
    현재는 MLP portfolio 만 가지고 오고 있다.

    Returns: 
        pd.DataFrame: daily returns of the portfolio nav
    """
    port_id = db.get_mlp_port_id().sort_values('port_id')
    port_nav = db.get_port_value(port_id['port_id'])
    port_nav.columns = port_id['portfolio']
    port_nav = port_nav.T.sort_index().T
    daily_returns = port_nav.pct_change().dropna()
    return daily_returns

def get_port_rebal_dt(daily_returns):
    """
    daily_returns 이 있는 기간의 rebalancing day를 구한다.

    Args:
        daily_returns (pd.DataFrame): daily returns of the portfolio nav

    Returns: 
        list: rebalancing day list
    """
    rebal_dt=[]
    for year in range(min(daily_returns.index).year,max(daily_returns.index).year + 1):
        for month in range(1,13):
            first_day_of_month = date(year, month, 1)
            start_trading_date=db.get_start_trading_date(market="KR", asofdate=first_day_of_month)
            if start_trading_date >= min(daily_returns.index.date) and start_trading_date <= max(daily_returns.index.date):
                rebal_dt.append(start_trading_date)
    return rebal_dt
    
def drawdown(series, returns=False):
    """
    Args:
        series (pd.DataFrame): time series of asset price
        returns (Boolean): True 이면 series 에 returns(수익율) 데이터가 들어가고, False 이면 가격 데이터가 들어간다.

    Returns:
        pd.DataFrame: the percentage drawdown
    """
    if returns:
        prices = (1 + series).cumprod()
    else:
        prices = series

    previous_peaks = prices.cummax()
    drawdowns = (prices - previous_peaks)/previous_peaks
    return drawdowns


def max_drawdown(series, returns=False):
    """
    Args:
        series  (pd.DataFrame): time series of asset price
        returns (Boolean): True 이면 series 에 returns(수익율) 데이터가 들어가고, False 이면 가격 데이터가 들어간다.

    Returns: 
        float: the percentage max drawdown
    """
    return drawdown(series, returns).min()

def investment_simulation(start_dt, end_dt, target_rate, port_returns, rebal_dt, uskr, returns_boundary):

    """Investment simulation with certain period of time, following the rules of changing portfolio.
    
    Rules:
    
    1. With the target rate and retruns boundarys of the portfolio, decide proper portfolio
    
    2. prices are updated every day, following the prices history of the portfolio. 
    
    3. Check whether or not reach the goal. If it reaches the goal, it's suceed.  
    
    4. Whenever one path is at the end, measure MDD and Volatility.

    Args:
        start_dt (DatetimeIndex): start date of the simulation
        end_dt (DatetimeIndex): end date of the simulation
        target_rate (float): annualized target rate  
        port_returns (pd.DataFrame): portfolio returns
        rebal_dt (list): list of rebalancing days(datetime)
        uskr (string): Market (ex. "US", "KR")
        returns_boundary (list): return boundaries of the portfolios.

    Returns:

        tuple containing

        - suceed (boolean): True when the simulation reaches the goal False when it does not

        - price (float): last price of the portfoio

        - vol (float): volatility of the portfolio

        - mdd (float): max draw down of the simulation path

        - time_taken (int): taken time(days)
    """
    # port_returns: 실제 시뮬레이션에 활용될 데이터
    duration = float((end_dt - start_dt).days)/365.0
    target_price = (1+target_rate)**(duration)
    # 처음의 port_id 는 target_rate 에 따라서 결정됨
    portfolio = get_portfolio(target_rate, uskr, returns_boundary)
    # 한달에 한번 port_id 정하고, 매일 매일 평가금액 갱신하면서, 타겟 프라이스에 도달했는지 체크 
    # 도달했으면 달성 그 때까지의 변동성 및 mdd 측정
    price = 1
    prices=[]
    prices.append(price)
    returns=[]
    for index, row in port_returns.iterrows():
        if index == end_dt:
            continue
        price = price * (1+row[portfolio])
        prices.append(price)
        returns.append(row[portfolio])
        d=float((end_dt - index).days)/365.0
        # 변동성, MDD, 달성율
        if index.date in rebal_dt: # 계산하는 날이 리밸런싱 데이에 해당이 되면.
            required_return_rate = (target_price/price)**(1/d)-1
            portfolio = get_portfolio(required_return_rate, uskr, returns_boundary)
        if price >= target_price:
            suceed=True
            time_taken= float((index-start_dt).days)
            df_prices=pd.DataFrame(prices, columns=['price'])
            mdd = max_drawdown(df_prices)
            df_returns=pd.DataFrame(returns, columns=['return'])
            vol = df_returns.std()*(252)**(0.5)
            return suceed, price, vol, mdd, time_taken
    suceed=False
    df_returns=pd.DataFrame(returns, columns=['return'])
    vol = df_returns.std()*(252)**(0.5)
    df_prices=pd.DataFrame(prices, columns=['price'])
    mdd = max_drawdown(df_prices)
    time_taken = (end_dt - start_dt).days
    return suceed, price, vol, mdd, time_taken

def investment_simulation2(start_dt, end_dt, target_rate, port_returns, rebal_dt, uskr, returns_boundary):
    """Investment simulation with certain period of time, following the rules of changing portfolio.
    This function is simple form of function investment_simulation.
    
    Rules:
    
    1. With the target rate and retruns boundarys of the portfolio, decide proper portfolio
    
    2. prices are updated every day, following the prices history of the portfolio. 
    
    3. Check whether or not reach the goal. If it reaches the goal, it's suceed.  

    Args:
        start_dt (DatetimeIndex): start date of the simulation
        end_dt (DatetimeIndex): end date of the simulation
        target_rate (float): annualized target rate  
        port_returns (pd.DataFrame): portfolio returns
        rebal_dt (list): list of rebalancing days(datetime)
        uskr (string): Market (ex. "US", "KR")
        returns_boundary (list): return boundaries of the portfolios.

    Returns:
        boolean: True when the simulation reaches the goal False when it does not
    """
    duration = float((end_dt - start_dt).days)/365.0
    target_price = (1+target_rate)**(duration)
    portfolio = get_portfolio(target_rate, uskr, returns_boundary)
    price = 1

    for index, row in port_returns.iterrows():
        if index == end_dt:
            continue
        d=float((end_dt - index).days)/365.0
        price = price * (1+row[portfolio])
        # 변동성, MDD, 달성율
        if index.date in rebal_dt: # 계산하는 날이 리밸런싱 데이에 해당이 되면.
            required_return_rate = (target_price/price)**(1/d)-1
            portfolio = get_portfolio(required_return_rate, uskr, returns_boundary)
        if price >= target_price:
            suceed=True
            return suceed
    suceed=False
    return suceed


def get_portfolio(target_rate, uskr, returns_boundary):
    """With the target rate, market, and retruns boundarys of the portfolio, decide proper portfolio

    Args:
        target_rate (float): annualized target rate  
        uskr (string): Market (ex. "US", "KR")
        returns_boundary (list): return boundaries of the portfolios.

    Returns:
        string: portfolio
    """
    if uskr == 'KR':
        if (target_rate < returns_boundary[0]):
            return 'MLP_KR_1'
        elif (target_rate >= returns_boundary[0]) and  (target_rate < returns_boundary[1]):
            return 'MLP_KR_2'
        elif (target_rate >= returns_boundary[1]) and  (target_rate < returns_boundary[2]):
            return 'MLP_KR_3'
        elif (target_rate >= returns_boundary[2]) and  (target_rate < returns_boundary[3]):
            return 'MLP_KR_4'
        elif (target_rate >= returns_boundary[3]):
            return 'MLP_KR_5'
        
    elif uskr == 'US':
        if (target_rate < returns_boundary[0]):
            return 'MLP_US_1'
        elif (target_rate >= returns_boundary[0]) and  (target_rate < returns_boundary[1]):
            return 'MLP_US_2'
        elif (target_rate >= returns_boundary[1]) and  (target_rate < returns_boundary[2]):
            return 'MLP_US_3'
        elif (target_rate >= returns_boundary[2]) and  (target_rate < returns_boundary[3]):
            return 'MLP_US_4'
        elif (target_rate >= returns_boundary[3]):
            return 'MLP_US_5'


# 예상 달성율(최적화를 위해서 부호 반대로)
def achivability(returns_boundary, daily_returns, rebal_dt, uskr, target_rate, simulating_years): 
    """This function get the probability of suceed to reach the goal.

    Args:
        returns_boundary (list): return boundaries of the portfolios.
        daily_returns (pd.DataFrame): total portfolio returns data
        rebal_dt (list): list of rebalancing days(datetime)
        uskr (string): Market (ex. "US", "KR")
        target_rate (float): annualized target rate  
        simulating_years (int): lenth of the simulation period

    Returns:
        float: probability of suceed to reach the goal
    """
    suceed_list=[] 
    simulating_days=simulating_years*252
    for i in range(len(daily_returns)-simulating_days):
        port_returns = daily_returns.iloc[i:(i+simulating_days)]
        start_dt=port_returns.index[0] 
        end_dt=port_returns.index[-1]
        suceed = investment_simulation2(start_dt, end_dt, target_rate, port_returns, rebal_dt, uskr, returns_boundary)
        suceed_list.append(suceed)
    return -sum(suceed_list)/len(suceed_list)

def achivability_sum(returns_boundary, daily_returns, rebal_dt, market=['US','KR'], target_rate_min=0.03, target_rate_max=0.13, simulating_years=2):
    """Sum of the achivability

    Args:
        returns_boundary (list): return boundaries of the portfolios.
        daily_returns (pd.DataFrame): total portfolio returns data
        rebal_dt (list): list of rebalancing days(datetime)
        market (list, optional): Market. Defaults to ['US','KR'].
        target_rate_min (float, optional): minimum annualized target rate. Defaults to 0.03.
        target_rate_max (float, optional): maximum annualized target rate. Defaults to 0.13.
        simulating_years (int, optional): lenth of the simulation period. Defaults to 2.

    Returns:
        float: Sum of the achivability
    """
    p=0.0
    for uskr in market:
        for target_rate in np.arange(target_rate_min, target_rate_max, 0.01):
            p = p + achivability(returns_boundary, daily_returns, rebal_dt, uskr, target_rate, simulating_years)
    return p

def plot_achivability(returns_boundary, daily_returns, rebal_dt, market=['US','KR'], target_rate_min=0.03, target_rate_max=0.13, simulating_years=2):
    """Achivability plot function

    Args:
        returns_boundary (list): return boundaries of the portfolios.
        daily_returns (pd.DataFrame): total portfolio returns data
        rebal_dt (list): list of rebalancing days(datetime)
        market (list, optional): Market. Defaults to ['US','KR'].
        target_rate_min (float, optional): minimum annualized target rate. Defaults to 0.03.
        target_rate_max (float, optional): maximum annualized target rate. Defaults to 0.13.
        simulating_years (int, optional): lenth of the simulation period. Defaults to 2.
    """
    for uskr in market:
        x = []
        y = []
        for target_rate in np.arange(target_rate_min, target_rate_max, 0.01):  
            x.append(target_rate)
            y.append(-achivability(returns_boundary, daily_returns, rebal_dt, uskr, target_rate, simulating_years))
        fig, ax = plt.subplots()
        ax.plot(x,y)
        ax.set_xlabel('Target rate')
        ax.set_ylabel('Probability')
        ax.set_title(f'Market:{uskr}, Returns Boundary:{returns_boundary}')
        plt.show()


def get_simulation_statistics(daily_returns,target_rate,rebal_dt,uskr,returns_boundary,simulating_years=2):
    """get simulation statistics of MDD and volatility

    Args:
        daily_returns (pd.DataFrame): total portfolio returns data
        target_rate (float): annualized target rate
        rebal_dt (list): list of rebalancing days(datetime)
        uskr (string): Market (ex. "US", "KR")
        returns_boundary (list): return boundaries of the portfolios.
        simulating_years (int, optional): lenth of the simulation period. Defaults to 2.
    """
    suceed_list=[] 
    price_list=[] 
    vol_list=[] 
    mdd_list=[] 
    time_taken_list=[]

    simulating_days=simulating_years*252
    for i in range(len(daily_returns)-simulating_days):
        port_returns = daily_returns.iloc[i:(i+simulating_days)]
        start_dt=port_returns.index[0] 
        end_dt=port_returns.index[-1]
        suceed, price, vol, mdd, time_taken = investment_simulation(start_dt, end_dt, target_rate, port_returns, rebal_dt, uskr, returns_boundary)
        suceed_list.append(suceed)
        price_list.append(price)
        vol_list.append(vol)
        mdd_list.append(mdd)
        time_taken_list.append(time_taken)

    #Statistics
    print("Target Return Rate: %.3f"%(target_rate))
    print("Number of Sample:", len(suceed_list))
    achivability = sum(suceed_list)/len(suceed_list)
    print("achivability:", achivability)
    df_vol=pd.DataFrame(vol_list)
    df_vol.columns = ['Volatility']
    print("Minimum Volatility:", df_vol.min().values)
    print("Maximum Volatility:", df_vol.max().values)
    print("Average Volatility:",  df_vol.mean().values)
    #ax1 = df_vol.plot.hist(bins=20)
    ax1 = df_vol.plot.kde()
    ax1.set_title(f'Market:{uskr}, Target Rate:{target_rate:.3f}')
    plt.show()

    df_mdd=pd.DataFrame(mdd_list)
    df_mdd.columns = ['MDD']
    print("Minimum MDD:", df_mdd.min().values)
    print("Maximum MDD:", df_mdd.max().values)
    print("Average MDD:",  df_mdd.mean().values)
    #ax2 = df_mdd.plot.hist(bins=20)
    ax2 = df_mdd.plot.kde()
    ax2.set_title(f'Market:{uskr}, Target Rate:{target_rate:.3f}')
    plt.show()

# target_rate, uskr 을 하나로 고정해 놓고, 달성확율(achivability)을 극대화하는 returns boundary 를 찾는 과정
def optimize_returns_boundary_for_one_target_rate(daily_returns, rebal_dt, uskr, target_rate, simulating_years=2):
    """
    This function searches returns boundaries of portfolio which maximize achivability of one target goal.

    Args:
        daily_returns (pd.DataFrame): total portfolio returns data
        rebal_dt (list): list of rebalancing days(datetime)
        uskr (string): Market (ex. "US", "KR")
        target_rate (float): annualized target rate
        simulating_years (int, optional): lenth of the simulation period. Defaults to 2.

    Returns:
        object: optimization results
    
    """
    
    def g1(x):	
        return -x[0] + x[1] -0.005 # >=0
    
    def g2(x):	
        return -x[1] + x[2] -0.005 # >=0
    
    def g3(x):	
        return -x[2] + x[3] -0.005 # >=0
    
    cons = ({'type': 'ineq', 'fun': g1},
		{'type': 'ineq', 'fun': g2},
		{'type': 'ineq', 'fun': g3})
    
    bounds = [(0.008, 0.15),]*4
    
    res = shgo(achivability, bounds, args=(daily_returns, rebal_dt, uskr, target_rate, simulating_years), iters=10, sampling_method='sobol', constraints = cons)
	
    return res


# target_rate 을 일정한 범위에서 변하게 하고, 달성확율(achivability)들의 합을 극대화하는 returns boundary 를 찾는 과정
def optimize_returns_boundary_for_general_target_rates(daily_returns, rebal_dt, market=['US','KR'], target_rate_min=0.03, target_rate_max=0.13, simulating_years=2):
    """
    This function searches returns boundaries of portfolio which maximize achivability of broad target goals

    Args:
        daily_returns (pd.DataFrame): total portfolio returns data
        rebal_dt (list): list of rebalancing days(datetime)
        market (list, optional): Market. Defaults to ['US','KR'].
        target_rate_min (float, optional): minimum annualized target rate. Defaults to 0.03.
        target_rate_max (float, optional): maximum annualized target rate. Defaults to 0.13.
        simulating_years (int, optional): lenth of the simulation period. Defaults to 2.

    Returns:
        object: optimization results
    
    """

    def g1(x):
        return -x[0] + x[1] -0.005 # >=0

    def g2(x):
        return -x[1] + x[2] -0.005 # >=0

    def g3(x):
        return -x[2] + x[3] -0.005 # >=0

    cons = ({'type': 'ineq', 'fun': g1},
        {'type': 'ineq', 'fun': g2},
        {'type': 'ineq', 'fun': g3})
    
    bounds = [(0.008, 0.15),]*4

    res = shgo(achivability_sum, bounds, args=(daily_returns, rebal_dt, market, target_rate_min, target_rate_max, simulating_years), iters=10, sampling_method='sobol', constraints = cons)
    return res