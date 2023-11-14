import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from hive import db
from config import get_args
from core.strategy.ablstrategy import AblStrategy
from core.analytics.pa import metrics

logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)
YESTERDAY = TODAY - timedelta(days=1)
REGIME = args.regime

logger.info(f"running factor application script {TODAY:%Y-%m-%d}")

def calc_daily_exp_rtn() -> None:
    """calc_exp_rtn function
    
    Calculate factors' post expected return
    
    """
    extra = dict(user=args.user, activity="daily_factor_expected_return_calculation", category="script")
    extra2 = dict(user=args.user, activity="daily_factor_expected_return_calculation", category="monitoring")

    date_check = db.check_weekend_holiday_date(market = "KR", asofdate = TODAY)
    
    if date_check == True:
        logger.info(msg=f"[SKIP] factor expected return calculation. {TODAY:%Y-%m-%d}", extra=extra)
        return
        
    logger.info(msg=f"[PASS] Start factor expected return calculation. {TODAY:%Y-%m-%d}", extra=extra)
    
    universe = db.load_universe("abl_us")
    price_asset = db.get_price(tickers=", ".join(list(universe.index))).loc[:YESTERDAY]
    price_factor = db.get_lens(TODAY)
   
    strategy = AblStrategy.load(
            universe=universe,
            price_asset=price_asset,
            price_factor=price_factor,
            regime=REGIME,
            asofdate=TODAY,
        )
    
    state = strategy.regime.get_state(strategy.date.strftime("%Y-%m-%d"))
    exp_ret_states = strategy.regime.expected_returns_by_states(
        price_df=price_factor.iloc[-strategy.regime_window:]
    )
   
    views = strategy.make_views(exp_ret_states.loc[state])
    prior_mu = metrics.expected_returns(
        price_factor, method="empirical"
    )
    prior_cov = metrics.covariance_matrix(
        price_factor, method="exponential", halflife=strategy.halflife,
    )
    post_mu, _ = metrics.blacklitterman(prior_mu, prior_cov, views)
    post_mu = pd.DataFrame(post_mu)
    post_mu.columns = ['exp_rtn']
    post_mu = post_mu.reset_index()
    post_mu['trd_dt'] = len(post_mu)*[price_factor.index[-1]]
    post_mu = post_mu[['trd_dt','factor','exp_rtn']]
    
    if args.database == "true":
        try:    
                
            db.TbLens.insert(post_mu)               

        except:

            try:
                db.TbLens.update(post_mu)
            except:
                price_factor = db.get_lens(TODAY)
                latest_price_factor = price_factor[price_factor.index.isin([price_factor.index[-1]])].dropna(axis='columns')
                
                latest_price_factor_list= latest_price_factor.columns.tolist()
                post_mu = post_mu[post_mu['factor'].isin(latest_price_factor_list)]
            
                db.TbLens.update(post_mu)
                
        logger.warning(msg=f"[PASS] factor expected return update completed. {TODAY:%Y-%m-%d}", extra=extra2)        
        logger.info(msg=f"[PASS] End factor expected return calculation. {TODAY:%Y-%m-%d}", extra=extra)



def calc_exp_rtn(date: datetime) -> None:
    """calc_exp_rtn function
    
    Calculate factors' post expected return
    
    """
    TODAY = date
    YESTERDAY = TODAY - timedelta(days=1)

    extra2 = dict(user=args.user, activity="historical_factor_expected_return_calculation", category="monitoring")

    date_check = db.check_weekend_holiday_date(market = "KR", asofdate = TODAY)
    
    if date_check == True:
        return
            
    universe = db.load_universe("abl_us")
    price_asset = db.get_price(tickers=", ".join(list(universe.index))).loc[:YESTERDAY]
    price_factor = db.get_lens(TODAY)
   
    strategy = AblStrategy.load(
            universe=universe,
            price_asset=price_asset,
            price_factor=price_factor,
            regime=REGIME,
            asofdate=TODAY,
        )
    
    state = strategy.regime.get_state(strategy.date.strftime("%Y-%m-%d"))
    exp_ret_states = strategy.regime.expected_returns_by_states(
        price_df=price_factor.iloc[-strategy.regime_window:]
    )
   
    views = strategy.make_views(exp_ret_states.loc[state])
    prior_mu = metrics.expected_returns(
        price_factor, method="empirical"
    )
    prior_cov = metrics.covariance_matrix(
        price_factor, method="exponential", halflife=strategy.halflife,
    )
    post_mu, _ = metrics.blacklitterman(prior_mu, prior_cov, views)
    post_mu = pd.DataFrame(post_mu)
    post_mu.columns = ['exp_rtn']
    post_mu = post_mu.reset_index()
    post_mu['trd_dt'] = len(post_mu)*[price_factor.index[-1]]
    post_mu = post_mu[['trd_dt','factor','exp_rtn']]
    
    if args.database == "true":
        try:    
                
            db.TbLens.insert(post_mu)               

        except:

            try:
                db.TbLens.update(post_mu)
            except:
                
                price_factor = db.get_lens(TODAY)
                latest_price_factor = price_factor[price_factor.index.isin([price_factor.index[-1]])].dropna(axis='columns')
                
                latest_price_factor_list= latest_price_factor.columns.tolist()
                post_mu = post_mu[post_mu['factor'].isin(latest_price_factor_list)]
                
                db.TbLens.update(post_mu)
                
        logger.warning(msg=f"[PASS] factor expected return update completed. {TODAY:%Y-%m-%d}", extra=extra2)
    

if __name__ == "__main__":
   
    extra = dict(user=args.user, activity="historical_factor_expected_return_calculation", category="script")
    logger.info(msg=f"[PASS] Start factor expected return calculation. {TODAY:%Y-%m-%d}", extra=extra)

    date_lst = pd.date_range(start='2014-01-02', end=args.date) 
    
    for i in date_lst:
        
        df_lens_ = calc_exp_rtn(date = i)
      
    logger.info(msg=f"[PASS] End factor expected return calculation. {TODAY:%Y-%m-%d}", extra=extra)
   
          