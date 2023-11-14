import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil import parser
from scipy import optimize 
from core.strategy.base import BaseStrategy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import get_args
from core.strategy.base import BaseStrategy
from hive import db


logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)
YESTERDAY = TODAY - timedelta(days=1)
    

class DwsStrategy(BaseStrategy):
    """DwsStrategy

    Args:
        BaseStrategy (class): BaseStrategy class is an algorithmic trading strategy that sequentially allocates capital among
    group of assets based on pre-defined allocatioin method.
    """
    
    def __init__(
        self,
        w_max: float = 0.6,
        w_min: float = 0.02,
    ) -> None:
        """initialize class

        Args:
            w_max (float, optional): maximum of the weights. Defaults to 0.6.
            w_min (float, optional): minimum of the weights. Defaults to 0.02.
        """
        self.w_max = w_max
        self.w_min = w_min
    
    def calculate_weights(risk_level: int, investor_style: int, l: float, m: float, asset: pd.DataFrame, cov1, cov2):
        """
        run_dws_allocation call calculate_weights when it tilts the original MLP portfolio weights.
        With object function of Sharpe ratio and weights distance between MLP port and DWS port, this fuction optimizes DWS port.
        During the optimization, so many constraints such as weights sum, upper bounds, lower bounds of the weights are involved.

        Args:
            risk_level (int): risk level
            investor_style (int): investor style
            l (float): Importance of distance between MLP port and DWS port comparing Sharpe ratio in the object function
            m (float): Importance of short term data comparing long term data in the object function
            asset (DataFrame): MLP portfolio allocation information including ticker, stock id, weights, asset class, annual returns
            cov1 (DataFrame): covariance matrix of the port (short term period of data)
            cov2 (DataFrame): covariance matrix of the port (long term period of data)
        
        Returns:
            opt : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination.
        """
        MeanReturns1 = asset['anual_return1']
        MeanReturns2 = asset['anual_return2']

        if risk_level == 1:
            rlb=1.0
            rub=1.2
        elif risk_level == 2:
            rlb=1.2
            rub=2.2
        elif risk_level == 3:
            rlb=2.2
            rub=3.2
        elif risk_level == 4:
            rlb=3.2
            rub=4.2
        elif risk_level == 5:
            rlb=4.2
            rub=5.0 


        weights = asset['weights']
        PortfolioSize = len(asset)
        RISK_SCORE = asset['risk_score']
        asset_class = asset['strg_asset_class']


        # 변화가 큰 pair 와 작은 pair 로 분리
        big_step1=0.8
        small_step1=0.4

        big_step2=0.4
        small_step2=0.2

        # 액티브 투자자는 비중이 높은 자산(equity, liquidity), 패시브 투자자는 자산비중이 낮은 자산(alternative, fixed income) 쪽으로 tilting
        # 변화가 큰 pair 와 작은 pair 로 분리

        if (investor_style ==2) or (investor_style ==4):
            asset['lb']=[-small_step1 if (((investor_style ==4)) and ((c == 'equity') or (c== 'liquidity'))) or (((investor_style == 2)) and ((c == 'alternative') or (c== 'fixedincome'))) else 0 for c in asset['strg_asset_class']]
            asset['ub']=[0 if (((investor_style ==4)) and ((c == 'equity') or (c== 'liquidity'))) or (((investor_style == 2)) and ((c == 'alternative') or (c== 'fixedincome'))) else big_step1 for c in asset['strg_asset_class']]
        elif (investor_style ==5) or (investor_style ==3):
            asset['lb']=[-small_step2 if (((investor_style ==5)) and ((c == 'equity') or (c== 'liquidity'))) or (((investor_style == 3)) and ((c == 'alternative') or (c== 'fixedincome'))) else 0 for c in asset['strg_asset_class']]
            asset['ub']=[0 if (((investor_style ==5)) and ((c == 'equity') or (c== 'liquidity'))) or (((investor_style == 3)) and ((c == 'alternative') or (c== 'fixedincome'))) else big_step2 for c in asset['strg_asset_class']]

        alb=asset['lb']
        aub=asset['ub']

        # define maximization of Sharpe Ratio using principle of duality
        def  f(x, weights, CovarReturns1, CovarReturns3, l, m):
            funcDenomr1 = np.sqrt(np.matmul(np.matmul(x, CovarReturns1), x.T) )
            funcDenomr3 = np.sqrt(np.matmul(np.matmul(x, CovarReturns3), x.T) )
            funcNumer1 = np.matmul(np.array(MeanReturns1),x.T)
            funcNumer3 = np.matmul(np.array(MeanReturns2),x.T)
            obj = -(m*funcNumer1/funcDenomr1+ (1-m)*funcNumer3/funcDenomr3) - l*((x-weights)@(x-weights))**0.5 # m: 장단기 비중 0 <= m <= 1 , l:기준포트(중재자)와의 차이 항에 파라미터를 곱한다. 

            return obj
        
        #define equality constraint representing fully invested portfolio
        def constraintEq(x):
            A=np.ones(x.shape)
            b=1
            constraintVal = np.matmul(A,x.T)-b 
            return constraintVal

        #define bounds and other parameters
        xinit=np.repeat(1/PortfolioSize, PortfolioSize)
        cons = ({'type': 'eq', 'fun':constraintEq},{'type': 'ineq', 'fun':lambda x: x@RISK_SCORE - rlb},{'type': 'ineq', 'fun':lambda x: rub - x@RISK_SCORE},{'type': 'ineq', 'fun':lambda x: (x-weights) / weights - alb},{'type': 'ineq', 'fun':lambda x: aub - (x-weights) / weights})
        lb = 0.02
        ub = 0.6
        bnds = tuple([(lb,ub) for x in xinit])
        
        #invoke minimize solver
        opt = optimize.minimize (f, x0 = xinit, args = (\
                                weights,  cov1, cov2, l, m), method = 'trust-constr',  \
                                bounds = bnds, constraints = cons, tol = 10**-3)
        #print(asset)
        
        return opt
    
    
def run_dws_allocation(market, level, weights, universe, prices) -> None:
    """
    run dws allocation at the month start trading date
    i.e. first trading day each month.
    With MLP portfolio weights, DWS strategy consider investor style. 
    Acorrding to the investor style, run_dws_allocation function tilt the original MLP portfolio weights.

    if args.database == "true", than the fuction update TB_PORT_ALLOC table

    Args:
        market (str): (US, KR) market
        level (int): risk level
        weights (DataFrame): MLP portfolio weights
        universe (DataFrame): universe of the MLP strategy
        prices (DataFrame): prices of the universe
    """
    extra = dict(user=args.user, activity="dws_allocation", category="script")

    if TODAY.date() != db.get_start_trading_date(market="KR", asofdate=TODAY):
        logger.info(msg=f"[SKIP] DWS allocation. {TODAY:%Y-%m-%d}", extra=extra)
        return

    logger.info(msg=f"[PASS] Start DWS allocation. {TODAY:%Y-%m-%d}", extra=extra)

    strategy = DwsStrategy()
        
    #계산을 위한 input 데이터 가공
    def annual_return_cov(period:int, prices, weights):
        startdate =  YESTERDAY - timedelta(days=period)
        daily_return = prices[weights.index.tolist()].loc[startdate:YESTERDAY].pct_change().dropna().copy()
        total_return = (1 + daily_return).cumprod(axis=0).iloc[-1]
        ar = (total_return ** (float(365.25/period))) - 1
        cov = daily_return.cov()*252.0
        return ar, cov
    
    short_period = 183 #day, 6 month
    long_period = 1096 #day, 3 years
    ar1,cov1 = annual_return_cov(short_period, prices, weights)
    ar2,cov2 = annual_return_cov(long_period, prices, weights)

    port_alloc = weights.to_frame().reset_index()
    port_alloc.columns = ['ticker', 'weights']
    port_alloc = pd.merge(left = port_alloc , right = universe, how = 'inner', on = 'ticker')
    ar1.name='anual_return1'
    ar2.name='anual_return2'
    port_alloc = pd.merge(left = port_alloc , right = ar1, how = 'inner', on = 'ticker')
    port_alloc = pd.merge(left = port_alloc , right = ar2, how = 'inner', on = 'ticker')

    #비중 표준화
    port_alloc['weights'] = strategy.clean_weights(port_alloc['weights'], 4, 1.0) # 중재자 비중
        
    #투자자 style 에 대한 for문
    for investor_style in [1,2,3,4,5]:
        portfolio = 'DWS_'+str(market)+'_'+str(investor_style)+'_'+str(level)
        portfolio_id = db.get_portfolio_id(portfolio=portfolio)

        allo = port_alloc.loc[:, ['stk_id', 'ticker']]
        
        if investor_style == 1:
            allo['weights'] = port_alloc["weights"]
        else:
            if investor_style==2 or investor_style==5:
                m=1
            elif investor_style==4 or investor_style==3:
                m=0
            allo['weights'] = DwsStrategy.calculate_weights(risk_level=level, investor_style=investor_style, l=1, m=m, asset= port_alloc, cov1=cov1, cov2=cov2).x                
            
        allo.index = allo.ticker
        weights_as_cl = pd.concat([allo['weights'], universe["strg_asset_class"]], axis=1, join="inner", keys=['weights', 'strg_asset_class'])
        
        weights = pd.Series()
        for asset_class in weights_as_cl.strg_asset_class.unique():
            split_weights = weights_as_cl[weights_as_cl.strg_asset_class == asset_class]["weights"]
            adjusted_weights = strategy.calc_adjusted_portfolio_weight(weights=split_weights)
            weights = weights.append(adjusted_weights)
        
        allo['weights'] = strategy.clean_weights(weights=weights, decimals=4, tot_weight=1.0)
        allo['rebal_dt'] = TODAY
        allo['port_id'] = portfolio_id
        
        risk_score = 0.0
        for idx, row in allo.iterrows():
            risk_score += row['weights'] * universe.loc[str(row['ticker'])].risk_score
        
        style = db.get_port_style(port_id=portfolio_id)
        msg = f"\n[PASS] DWS MP"
        msg += f"\n{TODAY.date()} | {market} level {level} - {style}"
        msg += f"\nrisk score {risk_score:.4f}\n"
        logger.info(msg, extra=extra)
        print(allo.loc[:, ['weights']].to_markdown())

        if args.database == "true":
            try:
                db.TbPortAlloc.insert(allo)
            except:
                try:
                    db.TbPortAlloc.update(allo)
                except:
                    db_alloc = db.get_alloc_weight_for_shares(strategy="DWS", market=market, level=f"{investor_style}_{level}")
                    db_alloc = db_alloc[db_alloc.rebal_dt == TODAY]

                    merge_df = allo.merge(db_alloc, on=["rebal_dt", "port_id", "stk_id"], how="outer")
                    delete_asset = merge_df[merge_df.weights_x.isnull()].stk_id.tolist()
                    update_asset = merge_df.dropna()
                    update_asset['weights'] = update_asset['weights_x']
                    insert_asset = merge_df[merge_df.weights_y.isnull()]
                    insert_asset['weights'] = insert_asset['weights_x']

                    db.delete_asset_port_alloc(rebal_dt=TODAY, port_id=portfolio_id, stk_id=delete_asset)
                    db.TbPortAlloc.update(update_asset)
                    db.TbPortAlloc.insert(insert_asset)
    
    logger.info(msg=f"[PASS] End DWS allocation. {TODAY:%Y-%m-%d}", extra=extra)