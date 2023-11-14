import os
import sys
import logging
import sqlalchemy as sa
import numpy as np
import pandas as pd
import random
import cvxpy as cp
from datetime import datetime,  timedelta
from cvxpy import Variable, Parameter, quad_form, reshape, Problem, Maximize, sqrt
from typing import Union, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from hive import db


class ActualPortfolio:
    def __init__(self,) -> None:
        pass

    def opt(self, w, p, portfolio_value=3000000, min_shares=0):
        self.w = w.values
        self.p = p.values
        self.n = len(self.w)
        self.portfolio_value = portfolio_value
        self.min_shares = min_shares 

        shares = cp.Variable(self.n, integer=True)
        cash = self.portfolio_value - self.p @ shares

        u = cp.Variable(self.n)
        eta = self.w * self.portfolio_value - cp.multiply(shares, self.p)

        _obj = cp.sum(u) + cash

        _cons = [
            eta <= u,
            eta >= -u,
            cash >= 0,
            shares >= self.min_shares,
        ]

        _opt = cp.Problem(cp.Minimize(_obj), _cons)

        _opt.solve(verbose=False)

        return shares.value



def is_start_of_half_year(TODAY: datetime) -> bool:
    """check if today is the start of half year"""
    MONTH = TODAY.month
    
    if MONTH not in [1, 7]:
        return False
    return TODAY.day == db.get_start_trading_date(market="KR", asofdate=TODAY).day


def get_ml_model_training_date(asofdate: datetime) -> datetime:
    """get the supposed date for model training based on the given the asofdate"""
    half = datetime(asofdate.year, 6, 30)
    if min(half, asofdate) == half:
        return half
    return datetime(asofdate.year - 1, 12, 31)


def optimal_portfolio(
        returns: pd.DataFrame, 
        num_sample: int = 100,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
) -> pd.DataFrame:
    """Optimization process to maximize the object function, risk-adjusted return\n
    Get the weights of assets following gammas after assigning the gammas in the same log scale.
    Object Function: risk-adjusted return -> (ret - gamma*risk).
    
    Constraints: 
        (The sum of the weights is 1) : sum(w) == 1,\n
        (all the weights is over 0)   : w >= 0.0,   \n
        (lower bound of the weights)  : w >= lb,    \n
        (upper bound of the weights)  : w <= ub     \n
        
    Args:
        returns (pd.DataFrame): Return time-series data    \n
        lower_bound (float): lower bound of an asset weight\n
        upper_bound (float): upper bound of an asset weight\n
        
    Returns:
        pd.DataFrame: \n
            weight: asset weights (gamma 값과 자산의 종류에 따르는 Dataframe)    \n
            ret_data: return DataFrame depeding on the gamma value             \n
            risk_data: standard deviation DataFrame depeding on the gamma value\n
    """
    n = len(returns.columns)
    w = Variable(n)
    mu = returns.mean()*365
    Sigma = returns.cov()*365
    gamma = Parameter(nonneg=True)
    ret = mu.values.T*w
    risk = quad_form(w, Sigma.values)
    lb = reshape(pd.DataFrame(np.repeat(lower_bound,n)).T.values.T, (n, ))
    ub = reshape(pd.DataFrame(np.repeat(upper_bound,n)).T.values.T, (n, ))

    prob = Problem(Maximize(ret - gamma*risk), [sum(w) == 1, w >= 0.0, w >= lb, w <= ub]) #, aineq.values >= bineq.values*w])
    risk_data = np.zeros(num_sample)
    ret_data = np.zeros(num_sample)

    gamma_vals = np.logspace(-2, 3, num=num_sample)
    weights = []

    for i in range(num_sample):
        gamma.value = gamma_vals[i]
        prob.solve()
        risk_data[i] = sqrt(risk).value
        ret_data[i] = ret.value
        weights.append(np.squeeze(np.asarray(w.value)))
    
    weight = pd.DataFrame(data = weights, columns = returns.columns)
    return weight, ret_data, risk_data


def data_resampler(
        er: pd.DataFrame, 
        cov: pd.DataFrame, 
        sims: int = 100, 
        period: int = 3650,
)->list: 
    """
    Draw Sample data, with multivariate normal distribution function parameters.

    Args:
        er (pd.DataFrame): mean values of returns.
        cov (pd.DataFrame): covariance matrix of returns.
        sims (int): number of simulations.
        period (int): length of the period (days).
    
    Returns:
        list: \n
        data (list): Return's time series data. The list is composed of pd.DataFrame data.
    """
    #create date index
    dates = pd.date_range(start='2013-09-05', periods = period, freq='D')
    
    data = []
    # generate returns data,simulation 수(sims) 만큼
    # 한 simulation의 길이는 period 만큼
    # 원하는 만큼 data를 resampling 하는 과정
    for i in range(0,sims):
        data.append(pd.DataFrame(columns = cov.columns, index = dates, data = np.random.multivariate_normal(er.values, cov.values, period)))
    
    return data


def mean_variance_optimizer(
        data: list,
        rf: float = 0.03,
        sims: int = 100, 
        # lb: pd.DataFrame = pd.DataFrame(np.repeat(0,10)),
        # ub: pd.DataFrame = pd.DataFrame(np.repeat(1,10)),
        # aineq: Optional[float] = 0.2, 
        # bineq: Optional[pd.Series],
) -> pd.Series:
    """
    Mean Variance Optimizer from Markowitz optimizes the weights of portfolios under the specific constraints:    
        In this function, we optimize the weights at the efficient frontier's tangent spot.
        The risk-free rate is US 10Y T-Bond yield in the most cases. Defaults to 0.03

    Args:
        data (list): A list which consists of plenty of DataFrames generated by `data_resampling`
        rf (float, optional): The risk-free rate. Mostly, US 10Y T-Bond yield is used. Defaults to 0.03.
        sims (int, optional): The number of the simulation repetition. Defaults to 100.

    Returns:
        pd.Series: \n
            opt_weights.loc[idx]: The optimal weights of the portfolio assets at the tangent point of the efficient frontier\n
            ef['sigma'].loc[idx]: The standart deviations of the portfolio assets at the tangent point of the efficient frontier\n
            ef['return'].loc[idx]: The expected return of the portfolio assets at the tangent point of the efficient frontier\n
    """
    weights =[]
    stdev = []
    exp_ret = []
    for i in range(0,sims):
        #optimize over every simulation
        w, r, std = optimal_portfolio(data[i], 100) #, lb, ub) #, aineq, bineq)
        weights.append(w)
        stdev.append(std)
        exp_ret.append(r)

    opt_weights = pd.DataFrame(np.mean(weights, axis=0), columns = data[0].columns)
    opt_stdev = pd.DataFrame(np.mean(stdev, axis=0))
    opt_exp_ret = pd.DataFrame(np.mean(exp_ret, axis=0))

    ef= pd.merge(left=opt_stdev, right=opt_exp_ret, left_index=True, right_index=True, how = 'inner')
    ef.columns=['sigma','return']
    ef['tangent']=(ef['return']-rf)/ef['sigma']
    idx= ef['tangent'].idxmax()

    return opt_weights.loc[idx], ef['sigma'].loc[idx], ef['return'].loc[idx]


def calculate_shares_GA(w: pd.Series, p: pd.Series, portfolio_value: float = 3_000_000) -> pd.Series:
    """
    Genetic Algorithm to calculate the shares of the actual portfolio:
        Genetic Algorithm can be used to solve both constrained and unconstrained optimization problems that are based on natural selection.
        In this function, we are looking for an optimized portfolio which meets the constraint that the difference between MP's weights and AP's weights.
    
    The followings are the steps of Genetic Algorithm:
        1. Create initial population
        2. Score and scale population
        3. Retain elite
        4. Select parents
        5. Product crossover and mutation children
    
    Args:
        w (pd.Series): the weights of the portfolio assets
        p (pd.Series): the prices of the portfolio assets
        portfolio_value (float, optional): the principle value invested in the portfolio. Defaults to 3_000_000(won).

    Returns:
        pd.Series: the optimized number of the shares of the portfolio assets
    """
    global value1, best_var
    value1 = 100
    best_var = []
    shares = ((portfolio_value * w) / p).astype(int).reset_index().drop(columns="index").values    # len(shares) = 10

    row_num = len(w)    # row_num = 10
    bnds = tuple((max(1,shares[x]-3),shares[x]+3) for x in range(row_num))  # Boundary allowed for adjusting the num of shares
    var_cnt = len(bnds)     # var_cnt = 10
    
    mutation_rate = 0.3
    target_gap = 0.2  # 괴리율의 합이 20%미만이 목표임

    # 목표비중과 실잔고비중의 차이의 절대값 합 / used to sort the chromosomes in increasing order
    def min_gap(수량):
        row_num = len(수량)
        수량 = np.array(수량)
        평가금액 = p * 수량
        if  portfolio_value > np.sum(평가금액):                         # check if there's cash in the portfolio
            CashAmount = (portfolio_value - np.sum(평가금액))/portfolio_value     # Cash Amount(Actual)
            gap = np.sum(abs(평가금액/portfolio_value - w))+CashAmount  # gap = Sum of the differences
        else:
            gap = 2
        return gap
    
    # 십진수로 변환해서 목적함수에 전달
    def call_func(binary):
        global value1, best_var
        var = []
        for ii in range(var_cnt):
            binary0 = binary[ii*8:ii*8+8]
            decimal = int(binary0,2) + bnds[ii][0]
            if  decimal > bnds[ii][1]: # 변수값 범위를 벗어나는 경우 오류로 뱉어 냄
                return 100
            var.append(decimal)
        result = min_gap(var)   # the difference of a portfolio
        
        if  value1 > result:
            value1 = result
            best_var = var
        return result  
    
    # Generating populations
    equal_cnt_k = 0
    min_val_k = value1

    for k in range(10000):      # 100 populations
        chromosome_list = []
        for i in range(32):     # A population(A "chromosome_list") of 32 portfolios("chromosomes") filled with 10 securities("genes")
            chromosome = ''
            for ii in range(var_cnt):
                decimal = random.randint(0, bnds[ii][1] - bnds[ii][0])
                gene = format(decimal,'b').zfill(8)     # 숫자가 아닌 텍스트로 전환 (e.g., 00000110)
                chromosome = chromosome + gene # 주욱 이어붙임 (length = 8 *10 = 80)
            chromosome_list.append(chromosome)

        # Elitism
        equal_cnt_i = 0
        min_val = value1
        for i in range(100):
            chromosome_list0 = sorted(chromosome_list, key=call_func)

            if  value1 > target_gap:    # Check if "value1" is bigger than "target_gap"
                if  min_val > value1:   # check whether "value1" has decreased than the previous one("min_val")
                    min_val = value1    # if YES, replace "min_val" to "value1"
                    equal_cnt_i = 0
                else:
                    equal_cnt_i += 1    # count if "value1" hasn't decreased
                if  equal_cnt_i >= 30:  # 더 이상 최저값이 갱신되지 않으면 종료한다.
                    break
            else:
                break
                
            chromosome_list0[31] = chromosome_list0[0] # 1등 염색체를 꼴찌 염색체에 Replace. 꼴찌는 사라지고 1등은 생존 확률이 높아진다

            # crossover
            X = random.randint(4, ((var_cnt) * 8) - 4)
            for ii in range(15):
                temp = chromosome_list0[ii][X:]
                chromosome_list0[ii] = chromosome_list0[ii][:X] + chromosome_list0[29-ii][X:]
                chromosome_list0[29-ii]= chromosome_list0[29-ii][:X] + temp
            temp = chromosome_list0[31][X:]  # Do like this (Due to the replacement done before this crossover above)
            chromosome_list0[31] = chromosome_list0[31][:X] + chromosome_list0[30][X:]  
            chromosome_list0[30]= chromosome_list0[30][:X] + temp

            # mutation
            rand = random.random()
            if rand < mutation_rate:
                for j in range(4):  # 1/8(4 of 32 chromosomes)에 돌연변이 염색체를 반영한다.
                    chromosome1 = ''
                    for ii in range(var_cnt):
                        decimal = random.randint(0, bnds[ii][1] - bnds[ii][0])
                        gene = format(decimal,'b').zfill(8) # 숫자가 아닌 텍스트로 전환
                        chromosome1 = chromosome1 + gene # 주욱 이어붙임
                    rand_num = random.randint(1, 30)
                    chromosome_list0[rand_num] = chromosome1
            chromosome_list = chromosome_list0      # replace the "chromosome_list" to a new one("chromosome_list0")
      
        # Conditional Forced Loop Quit
        if  min_val_k > min_val:
            min_val_k = min_val
            equal_cnt_k = 0
        else:
            equal_cnt_k += 1
        if  equal_cnt_k >= 2000:  # 더 이상 최고값이 갱신되지 않으면 종료한다.
            break

    shares = best_var  
   
    return pd.Series(index=w.index, data=shares, name="shares")