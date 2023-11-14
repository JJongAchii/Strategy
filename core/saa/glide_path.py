import numpy as np
import pandas as pd
from pandas_datareader import data as wb
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt
#from scipy.stats import norm, gmean, cauchy
import seaborn as sns
from datetime import datetime
import pylab
from scipy.optimize import curve_fit
from datetime import datetime, date, timedelta
from hive import db

def bootstrap_yearly_return(data, years, iterations, frequency, weighted=False):
    """지수의 daily 시계열 데이터를 받아서 weekly return을 추출해서 시뮬레이션 횟수 만큼, yearly prices set를 반환, bootstraping 방식

    Args:
        data (pd.DataFrame): timeseries asset price data
        years (int): remaining years (maturity)
        iterations (int): number of simulation
        frequency (str): characteristics of the data('d' means daily data, 'w' means weekly data, and 'm' means monthly data)
        weighted (bool, optional): If true, recent time series data is more weighted. Defaults to False.

    Returns:
        NDArray[float64]: yearly prices array with shape of (years, iterations, number of asset)
    """
    if frequency == 'd':
        lr = np.log(1+data.interpolate()[::5].pct_change())[1:]
        for i in range(1,5):
            lr = pd.concat([lr,np.log(1+data.interpolate()[i::5].pct_change())[1:]])
        lr.sort_index(inplace=True)
        nf = 52 # number of frequency 일년에 반복되는 횟수, 52주
    elif frequency == 'w':
        lr = np.log(1+data.pct_change())[1:]
        nf = 52
    elif frequency == 'm':
        lr = np.log(1+data.pct_change())[1:]
        nf =12 # number of frequency 일년에 반복되는 횟수, 12달

    lr = lr.to_numpy()
    
    if weighted == True:
        # 최근을 선형 가중해서 sampling
        i = (np.random.triangular(0, len(lr), len(lr), nf*years*iterations)).astype(int)

    else:
        # uniform sampling
        i = np.random.choice(range(len(lr)),nf*years*iterations)
    p_returns =  lr[i]
    y_returns= np.ones((years*iterations,np.size(p_returns,1)))
    for j in range(len(y_returns)):
        for k in range(nf):
            y_returns[j] = y_returns[j] * np.exp(p_returns[nf*j+k])
    return np.reshape(y_returns,(years,iterations,-1))


def gbm_yearly_return(data, years, iterations, frequency):
    """지수의 weekly 시계열 데이터를 받아서 시뮬레이션 횟수 만큼, yearly prices set를 반환, GBM 방식

    Args:
        data (pd.DataFrame): timeseries asset price data
        years (int): remaining years (maturity)
        iterations (int): number of simulation
        frequency (str): characteristics of the data('d' means daily data, 'w' means weekly data, and 'm' means monthly data)

    Returns:
        NDArray[float64]: yearly prices array with shape of (years, iterations, number of asset)
    """
    if frequency == 'd':
        lr = np.log(1+data.interpolate()[::5].pct_change())[1:]
        for i in range(1,5):
            lr = pd.concat([lr,np.log(1+data.interpolate()[i::5].pct_change())[1:]])
        lr.sort_index(inplace=True)
        nf = 52 # number of frequency 일년에 반복되는 횟수, 52주
    elif frequency == 'w':
        lr = np.log(1+data.pct_change())[1:]
        nf = 52
    elif frequency == 'm':
        lr = np.log(1+data.pct_change())[1:]
        nf =12 # number of frequency 일년에 반복되는 횟수, 12달

    cov = np.cov(np.transpose(lr))*nf # weekly or monthly -> yearly
    L = np.linalg.cholesky(cov) # cholesky decomposition
    c = np.size(data,1)
    mu = np.mean(lr, axis=0)*nf # weekly or monthly -> yearly 
    z = np.random.normal(0, 1, size=(c, years*iterations))
    MU = np.full((years*iterations,c), mu) #평균 값들의 배열
    MU = np.transpose(MU) # 지수가 행으로
    x= MU + np.dot(L,z)
    t_cov = np.cov(x)
    x = np.reshape(np.transpose(x),(years,iterations,-1))
    # 수익율을 exponentila 위에 올린 형태(price 차원으로 return)
    return np.exp(x)


def import_stock_data(tickers, start = '2010-1-1', end = datetime.today().strftime('%Y-%m-%d')): 
    """Import data using pandas_datareader (daily data)

    Args:
        tickers (str or list of str): tickers
        start (str, optional): start date of the data. Defaults to '2010-1-1'.
        end (str, optional): end date of the data.. Defaults to datetime.today().strftime('%Y-%m-%d').
    """
    data = pd.DataFrame()
    if len([tickers]) ==1:
        data[tickers] = wb.DataReader(tickers, data_source='yahoo', start = start)['Adj Close']
        data = pd.DataFrame(data)
    else:
        for t in tickers:
            data[t] = wb.DataReader(t, data_source='yahoo', start = start)['Adj Close']
    return(data)


def log_returns(data): 
    """the logarithmic returns

    Args:
        data (pd.DataFrame): timeseries asset price data

    Returns:
        NDArray: timeseries asset log return data
    """
    return (np.log(1+data.pct_change()))


def simple_returns(data):
    """the simple returns

    Args:
        data (pd.DataFrame): timeseries asset price data

    Returns:
        pd.DataFrame: timeseries asset return data
    """
    return ((data/data.shift(1))-1)


def drift_sigma_calc(data, return_type='log'): 
    """Compute the Drift and the volatility

    Args:
        data (pd.DataFrame): timeseries asset price data
        return_type (str, optional): return type. 'log' means log return, 'simple' means simple return. Defaults to 'log'.

    Returns:
        tuple containing

        - drift (ndarray): drifts for the assets

        - sigma (ndarray): sigmas for the assets

        - u (ndarray): mean of log returns of the assets
    """
    if return_type=='log':
        lr = log_returns(data)
    elif return_type=='simple':
        lr = simple_returns(data)
    u = lr.mean()
    var = lr.var()
    drift = u-(0.5*var)
    sigma = lr.std()
    try:
        return drift.values, sigma.values, u.values
    except:
        return drift, sigma, u

def yearly_returns(ft, stv, years, iterations):
    """월단위의 수익율, 표준편차를 받아서, 년 수익율에 해당하는 시뮬레이션을 년 단위로 배열 생성

    Args:
        ft (float): 월 단위의 수익율
        stv (float): 월 단위의 표준편차
        years (int): remaining years (maturity)
        iterations (int): number of simulation

    Returns:
        ndarray: yearly prices array with shape of (years,iterations,number of asset)
    """
    yr = np.exp(np.random.normal(ft*12, stv*(12**0.5), size=(years, iterations)))
    return yr


def yearly_returns2(ft): 
    """월 단위 수익율을 받아서 년 단위 수익율로 변환

    Args:
        ft (float): 월 단위의 수익율

    Returns:
        float: yearly price
    """
    yr = np.exp(ft*12)
    return yr

def plot_price_df(price_list,plot=False):
    """plot price distribution function. One is 'Price Probability Density Function', the other is 'Price Cumulative Distribution Function'.

    Args:
        price_list (list): price distribution data
        plot (bool, optional): IF True, plot it. Defaults to False.
    """
    if plot == True:
        x = pd.DataFrame(price_list).iloc[-1]
        pdf=sns.displot(x, kind="kde").set(title="Price Probability Density Function")
        pdf.set_axis_labels("Price", "Probability Density")
        plt.show()
        cdf=sns.displot(x, kind="ecdf").set(title="Price Cumulative Distribution Function")
        cdf.set_axis_labels("Price", "Probability Density")    
        plt.show()


def monte_carlo_asset_(mu, years, plot=False):
    """목표 적립금(target value) 함수 (deterministic 한 함수)

    Args:
        mu (float): monthly return
        years (int): remaining years (maturity)
        plot (bool, optional): IF True, plot it. Defaults to False.

    Returns:
        NDArray[float64]: price time series array
    """
    returns = yearly_returns2(mu)
    price_list = np.zeros(years+1)
    price_list[0] = 0
    for t in range(0,years):
        price_list[t+1] = (price_list[t]+1.02**t)*returns

    # Plot Option
    plot_price_df(price_list,plot)
    #return np.mean(price_list, axis = 1)
    return price_list


def monte_carlo_asset_var(returns_e, years, iterations, percentile, plot=False):
    """percentile VaR 시뮬레이션

    Args:
        returns_e (list): 위험 자산의 수익율 시뮬레이션 배열
        years (int): remaining years (maturity)
        iterations (int): number of simulation
        percentile (int): percentile
        plot (bool, optional): IF True, plot it. Defaults to False.

    Returns:
        scalar or ndarray: np.percentile(price_list[years], percentile)
    """

    # 자산 가격을 저장할 빈 배열 생성
    price_list = np.zeros((years+1, iterations))
    price_list[0] = 0 # 현재 시작 시점의 자산은 없다.
    # 각 해의 자산 계산
    # 매해 정기 기여금 1 로 설정
    for t in range(0,years):
        price_list[t+1] = (price_list[t]+1.02**t)*returns_e[t]
    # Plot Option
    plot_price_df(price_list,plot)

    return np.percentile(price_list[years], percentile)


def monte_carlo_asset_prob(returns_e, returns_b, theta, years, iterations, floor, plot=False):
    """원금 손실 확율을 계산하는 simulation

    Args:
        returns_e (list): 위험 자산의 수익율 시뮬레이션 배열
        returns_b (list): 안전 자산의 수익율 시뮬레이션 배열
        theta (list): theta[t]는 t 시점의 위험자산 비율
        years (int): remaining years (maturity)
        iterations (int): number of simulation
        floor (float): 만기에서의 원금(principle)의 증가율
        plot (bool, optional): IF True, plot it. Defaults to False.

    Returns:
        float: 원금 손실 확율
    """

    # price_list : 자산 가격을 저장할 빈 배열 생성
    price_list = np.zeros((years+1, iterations))
    principal = np.zeros((years+1)) # 납입 원금
    price_list[0] = 0 # 현재, 시작 시점의 자산은 없다.
    principal[0] = 0  
    # 각 해의 자산 계산
    # 매해 정기 기여금 1 로 설정
    
    for t in range(0,years):
        price_list[t+1] = (price_list[t]+1.02**t)*((returns_e[t]**theta[t])*(returns_b[t]**(1-theta[t]))) # 두 자산의 1년 합산 수익율
        principal[t+1] = (principal[t]+1.02**t)
    # Plot Option
    plot_price_df(price_list,plot)
    if plot == True:
        print("principal:", principal[years]*(1+floor))
        print("price:", np.mean(price_list, axis = 1))
        print("probablity:",(price_list[years] < principal[years]*(1+floor)).sum()/iterations)
    #return np.mean(price_list, axis = 1), (price_list[years] < years).sum()/iterations
    #return np.mean(np.where(price_list[years] < var, 0, price_list), axis=1) # 값이 var 이하면 값을 0으로

    return (price_list[years] < principal[years]*(1+floor)).sum()/iterations


def monte_carlo_asset_keep_principal(returns_e, returns_b, theta, years, iterations, floor, plot=False):
    """실제 적립금(Actual fund level) 시뮬레이션을 통해, 원금 손실 확율을 계산

    Args:
        returns_e (list): 위험 자산의 수익율 시뮬레이션 배열
        returns_b (list): 안전 자산의 수익율 시뮬레이션 배열
        theta (list): theta[t]는 t 시점의 위험자산 비율
        years (int): remaining years (maturity)
        iterations (int): number of simulation
        floor (float): 만기에서의 원금(principle)의 증가율
        plot (bool, optional): IF True, plot it. Defaults to False.

    Returns:
        ndarray: 원금 손실 확율
    """

    # price_list : 자산 가격을 저장할 빈 배열 생성
    price_list = np.zeros((years+1, iterations))
    principal = np.zeros((years+1)) # 납입 원금
    price_list[0] = 0 # 현재, 시작 시점의 자산은 없다.
    principal[0] = 0  
    # 각 해의 자산 계산
    # 매해 정기 기여금 1 로 설정
    
    for t in range(0,years):
        price_list[t+1] = (price_list[t]+1.02**t)*((returns_e[t]**theta[t])*(returns_b[t]**(1-theta[t]))) # 두 자산의 1년 합산 수익율
        principal[t+1] = (principal[t]+1.02**t)
    # Plot Option
    plot_price_df(price_list,plot)
    if plot == True:
        print("principal:", principal[years]*(1+floor))
        print("price:", np.mean(price_list, axis = 1))
        print("probablity:",(price_list[years] < principal[years]*(1+floor)).sum()/iterations)
    # return np.mean(price_list, axis = 1), (price_list[years] < years).sum()/iterations
    return np.mean(np.where(price_list[years] < principal[years]*(1+floor), 0, price_list), axis=1) # 값이 납입금 이하면 값을 0으로


def monte_carlo_asset(returns, theta, years, iterations, percentile, floor, plot=False):
    """실제 적립금(Actual fund level) 시뮬레이션 함수

    Args:
        returns (list): (years X iteration X 지수종류) 차원의 배열
        theta (list): 자산들의 비중 배열, 여러 자산의 시간에 따르는 비중을 1차원 배열에 담았다.
        years (int): remaining years (maturity)
        iterations (int): number of simulation
        percentile (int): percentile
        floor (float): 만기에서의 원금(principle)의 증가율
        plot (bool, optional): IF True, plot it. Defaults to False.

    Returns:
        tuple containing

        - price_list (ndarray): assets price array with time. price_list[t]

        - penalty여부 (bool): 원금 * (1+수익율) 만도 못한 수익율의 확율 비중이 유의수준을 넘으면 True 반환
    """

    # price_list : 자산 가격을 저장할 빈 배열 생성
    price_list = np.zeros((years+1, iterations))
    principal = np.zeros((years+1)) # 납입 원금
    price_list[0] = 0 # 현재, 시작 시점의 자산은 없다.
    principal[0] = 0 

    c = np.size(returns,2)
    for t in range(0,years):
        r = np.ones(iterations)
        theta_ = 0
        for j in range(c-1):
            r = r*(returns[t,:,j]**theta[t*(c-1)+j])
            theta_ = theta_ + theta[t*(c-1)+j]
        r = r*(returns[t,:,c-1]**(1-theta_))
        price_list[t+1] = (price_list[t]+1.02**t) * r                           
        principal[t+1] = (principal[t]+1.02**t)
    
    # Plot Option
    plot_price_df(price_list,plot)
    if plot == True:
        print("principal:", principal*(1+floor))
        print("price:", np.mean(price_list, axis = 1))
        print("probablity:",(price_list[years] < principal[years]*(1+floor)).sum()/iterations) # 만기에 원금 이하 확율

    return np.mean(price_list, axis = 1),  ((price_list[years] < principal[years]*(1+floor)).sum()/iterations  > float(percentile/100.0)) # 원금 * (1+수익율) 만도 못한 비율이 유의수준을 넘으면 True 반환



def monte_carlo_asset_stat(returns, theta, years, iterations, percentile, floor, plot=False): # monte_carlo_asset_bootstrap 에서 상속 받도록
    """실제 적립금(Actual fund level) 시뮬레이션 함수, statistics를 보려할 때 이용하는 함수

    Args:
        returns (list): (years X iteration X 지수종류) 차원의 배열
        theta (list): 자산들의 비중 배열, 여러 자산의 시간에 따르는 비중을 1차원 배열에 담았다.
        years (int): remaining years (maturity)
        iterations (int): number of simulation
        percentile (int): percentile
        floor (float): 만기에서의 원금(principle)의 증가율
        plot (bool, optional): IF True, plot it. Defaults to False.

    Returns:
        tuple containing

        - price_list (ndarray): assets price array with time. price_list[t]

        - price_list_std (ndarray): assets price standard deviation array with time. std[t]

        - np.percentile(price_list, 1, axis = 1) (ndarray): percentile 1% price array with time.

        - np.percentile(price_list, 99, axis = 1) (ndarray): percentile 99% price array with time.

        - principal (list): principal array with time.

        - sf_pro (list): short fall probability array with time.
    """

    # price_list : 자산 가격을 저장할 빈 배열 생성
    price_list = np.zeros((years+1, iterations))
    principal = np.zeros((years+1)) # 납입 원금
    sf_pro = np.zeros((years+1)) # 납입원금 이하 확율
    price_list[0] = 0 # 현재, 시작 시점의 자산은 없다.
    principal[0] = 0 
    sf_pro[0] =0 
    # 각 해의 자산 계산
    # 매해 정기 기여금 1 X (1+인플레이션) 으로 설정
    # r: 포트폴리오의 1년 합산 수익율
    # 지수의 종류가 c 이면 theta의 합이 1 이라는 조건때문에 실제로 구해야 하는  theta는 c-1 개 이다. theta_ = 0~c-2 번째까지의 theta의 합. 때문에 1- thrta_는 마지막 자산의 비중.

    c = np.size(returns,2)
    for t in range(years):
        r = np.ones(iterations)
        theta_ = 0
        for j in range(c-1):
            r = r*(returns[t,:,j]**theta[t*(c-1)+j])
            theta_ = theta_ + theta[t*(c-1)+j]
        r = r*(returns[t,:,c-1]**(1-theta_))
        price_list[t+1] = (price_list[t]+1.02**t) * r                           
        principal[t+1] = (principal[t]+1.02**t)
        sf_pro[t+1] = (price_list[t+1] < principal[t+1]).sum()/iterations

    # Plot Option
    plot_price_df(price_list,plot)

    if plot == True:
        print("principal:", principal*(1+floor))
        print("price:", np.mean(price_list, axis = 1))
        print("probablity:",sf_pro)


    return np.mean(price_list, axis = 1), np.std(price_list, axis = 1), np.percentile(price_list, 1, axis = 1), np.percentile(price_list, 99, axis = 1), principal, sf_pro
    # return np.mean(np.where(price_list[years] < var, 0, price_list), axis=1) # 값이 var 이하면 그 열의 값들을 0으로 만듦. 패널티를 주는 것과 동일
    # price_list[years] = np.where(price_list[years] < np.percentile(price_list[years], percentile) , 0, price_list[years]) # 꼬리 분포의 값들을 0으로 만들어주는 효과
    '''
    #포트폴리오 자산 평균
    np.mean(price_list, axis = 1)
    #표준편차
    np.std(price_list, axis = 1)
    #만기에서의 원금손실 확율: 예들 들어 만기에서 원금 손실은 안된다는 조건
    (price_list[years] < principal[years]*(1+floor)).sum()/iterations)
    #만기에서의 원금 손실값
    price_list[np.where(price_list[years] < principal[years]*(1+floor))].mean()
    Penalty funtion: np.heaviside((price_list[years] < principal[years]*(1+floor)).sum()/iterations -percentile/100, 1) *1000
    '''
    

def get_utility(A, T, lambda0, delta, year, nu1, nu2): 
    """Prospect theory utility function

    Args:
        A (float): t차 년도 말 시점의 DC 실제 적립금(actual fund level)
        T (float): t차년도 말 시점의 A의 목표 적립금(pre-defined target value)
        lambda0 (float): Initial loss aversion to shortfall
        delta (float): Loss aversion property to shortfall
        year (int): utility의 해당 시점
        nu1 (float): Curvature parameter for surplus
        nu2 (float): Curvature parameter for shortfall

    Returns:
        float: Prospect theory utility function
    """
    if A-T > 0 :
        return ((A-T)**nu1)/nu1
    elif A-T < 0 :
        return  -(lambda0 + delta*year )*((T-A)**nu2)/nu2
    else :
        return 0

def sigmoid(x, x0, k, L, L0):
    """sigmoid function

    Args:
        x (float): independent variable
        x0 (float): S자형의 중간점의 x값
        k (float): 로지스틱 성장율 또는 곡선의 가파른 정도
        L (float): 곡선의 최대 값
        L0 (float): dependent variable (y) 의 중간 값

    Returns:
        float: sigmoid function
    """
    y = L / (1+np.exp(k*(x-x0))) + L0
    return y


class GlidePath:
    """Glide Path class
    """

    def run_glidepath(self, tickers, year = 20, years = 40, iterations = 1000, floor = 0, percentile = 1 ):
        """running glide path function. This function returns timseries weights of assets after optimization. 

        Args:
            tickers (list): tickers
            year (int, optional): 과거 데이터 사용기간(단위 년, windows 크기). Defaults to 20.
            years (int, optional): remaining years (maturity). Defaults to 40.
            iterations (int, optional): number of simulation. Defaults to 1000.
            floor (int, optional): 만기에서의 원금(principle)의 증가율. Defaults to 0.
            percentile (int, optional): percentile. Defaults to 1.

        Returns:
            tuple containing

            - model (object): output_dict (a dictionary including the best set of variables found and the value of the given function associated to it)

            - returns (ndarray): yearly prices array with shape of (years, iterations, number of asset)
        """

        macro_data = db.get_macro_data_from_ticker(tickers, datetime.now()-timedelta(days=int(year*365.25))).sort_index()
        returns =  gbm_yearly_return(macro_data, years, iterations, frequency='d') # GBM 방식, exp(r) 형태의 return
        #returns =  bootstrap_yearly_return(data, years, iterations, frequency='d', weighted=False) # bootstrap 방식, frequency는 엑셀에서 읽어오는 daily data를 쓰면 'd', weekly data를 쓰면 'w', monthly data를 쓰면 'm' 



        def V(theta, years, simulation_trials, lambda0, delta,  nu1, nu2, discount): 
            """CPI(Control Performance Index)

            Args:
                theta (list): 자산들의 비중 배열, 여러 자산의 시간에 따르는 비중을 1차원 배열에 담았다.
                years (int): remaining years (maturity).
                simulation_trials (int): number of simulation
                lambda0 (float): Initial loss aversion to shortfall
                delta (float): Loss aversion property to shortfall
                nu1 (float): Curvature parameter for surplus
                nu2 (float): Curvature parameter for shortfall
                discount (float): discount factor

            Returns:
                float: CPI(Control Performance Index)
            """
            A, penalty = monte_carlo_asset(returns, theta, years, simulation_trials, percentile, floor, plot=False) # 실제 적립금(Actual fund level) 배열
            T = monte_carlo_asset_(0.002, years= years, plot=False) # 목표 적립금(target value) 배열
            Vt= np.zeros(years+1) # CPI 배열 초기화
            for i in range(1, years+1): # CPI 배열 계산
                Vt[i]= Vt[i-1] + (discount**i)*get_utility(A[i], T[i], lambda0, delta, i, nu1, nu2)

            #패널티 펑션 p를 이용해서 비중 theta의 합이 1을 넘지 않는 제약 조건을 만듦    
            c = np.size(returns, 2)
            theta_p=np.zeros(years)
            p = 0
            for t in range(0,years):
                for j in range(c-1):
                    theta_p[t] = theta_p[t] + theta[t*(c-1)+j]
                p = p + np.where(theta_p[t]>1,1000,0)
            return Vt[years] -p 
        


    
        def f(theta):
            """최적화 목적 함수, 최적화를 통해서 구해야 하는 비중 배열(theta)을 제외한 나머지 파라미터들은 고정시킨다.
            최적화는 minimum 값에 해당하는 값을 찾아내므로, -V에 해당하는 f를 정의해서 f의 minimum에 해당하는 theta를 구하는 방식으로 진행.

            Args:
                theta (list): 자산들의 비중 배열, 여러 자산의 시간에 따르는 비중을 1차원 배열에 담았다.

            Returns:
                float: - CPI(Control Performance Index)
            """
            ### Model parameter(except discount rate) setting
            lambda0 =4.5
            delta = 10
            years = 40
            nu1 = 0.44 
            nu2 = 0.88
            discount = 0.98
            iterations = 1000
            return -V(theta, years, iterations, lambda0, delta,  nu1, nu2, discount)

        asset_number=len(tickers)
        dim = years*(asset_number-1)

        varbound=np.array([[0,1]]*dim) # theta 배열의 boundary
        model=ga(function=f,dimension=dim,variable_type='real',variable_boundaries=varbound)
        model.run()

        return model, returns

    def plot_glidepath(self, model, years, tickers):
        """glide path plot function

        Args:
            model (object): optimization output
            years (int): remaining years (maturity).
            tickers (list): tickers

        Returns:
            list: curve_fit parameters of each asset
        """
        glide_path = model.output_dict['variable']
        g = np.reshape(glide_path,(years,-1))
        asset_number = len(tickers)
        c = asset_number
        g_c=np.zeros(years)

        '''
        b : blue.
        g : green.
        r : red.
        c : cyan.
        m : magenta.
        y : yellow.
        k : black.
        w : white.
        '''
        color=['b','g','r','c','m','y','k']

        for t in range(0,years):
            for j in range(c-1):
                g_c[t] = g_c[t] + g[t,j]
            g_c[t] = 1- g_c[t]

        xdata = range(1,len(g)+1)

        popt_a = [] 
        pcov_a = []
        for j in range(c-1):
            popt, pcov = curve_fit(sigmoid, xdata, g[:,j], bounds=[[years/3,-0.2,0,0],[years,0.2,1,1]] ) 
            popt_a.append(popt)
            pcov_a.append(pcov)
        popt, pcov = curve_fit(sigmoid, xdata, g_c, bounds=[[years/3,-0.2,0,0],[years,0.2,1,1]] )
        popt_a.append(popt)
        pcov_a.append(pcov)

        x = np.linspace(0, years, 101)
        for j in range(c-1):
            y = sigmoid(x, *popt_a[j])
            pylab.plot(xdata, g[:,j], 'o', color=color[j])
            pylab.plot(x,y, '-', color=color[j])
        y = sigmoid(x, *popt_a[c-1])
        pylab.plot(xdata, g_c, 'o', color=color[j+1])
        pylab.plot(x,y, '-', color=color[j+1])

        pylab.xlabel('years')
        pylab.ylabel('weight')
        pylab.show()

        return popt_a

    def plot_glidepath_stat(self, popt_a, returns, years, floor):
        """plotting function for glide path statistics

        Args:
            popt_a (list): curve_fit parameters of each asset
            returns (ndarray): yearly prices array with shape of (years, iterations, number of asset)
            years (int): remaining years (maturity).
            floor (float): 만기에서의 원금(principle)의 증가율.
        """
        asset_number = np.size(returns,2)
        dim = years*(asset_number-1)
        theta_f= np.zeros(dim)
        for i in range(asset_number-1):
            for j in range(years):
                theta_f[j*(asset_number-1)+i] = sigmoid(j+1, *popt_a[i])

        m, s , v, v_, p, sf = monte_carlo_asset_stat(returns, theta_f, years, 1000, 1, floor, plot=True)
        i = range(0,years+1)

        pylab.plot(i,m, label='Mean Price')
        pylab.plot(i,s, label='Standard Deviation')
        pylab.plot(i,v, label='1% Percentile')
        pylab.plot(i,v_, label='99% Percentile')
        pylab.plot(i,p, label='Principal Value')
        pylab.legend(loc='best')
        pylab.xlabel('years')
        pylab.ylabel('Value')
        pylab.show()

        pylab.plot(i[1:],sf[1:], label='Principal Loss Probability')
        pylab.legend(loc='best')
        pylab.xlabel('years')
        pylab.ylabel('Probability')
        pylab.show()