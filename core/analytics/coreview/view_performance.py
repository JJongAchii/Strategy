import os
import sys
import logging
import numpy as np
import pandas as pd
import sqlalchemy as sa

from datetime import date, timedelta
from dateutil import parser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../..")))
from hive import db
from config import get_args
from core.strategy.base import BaseStrategy
from core.analytics.utils import portfolio_update
from core.analytics.backtest import strategy_analytics

logger = logging.getLogger("sqlite")
args = get_args()
TODAY = parser.parse(args.date)

####################################################################################################
def run_view_performance_update():
    """
    Recall the views and their default weights to construct their model portfolio.
    Insert the book, adjusted weights, and NAV of the portfolios into DB.
    """
    extra = dict(user=args.user, activity="coreview_allocation", category="script")
    
    if TODAY.date() != db.get_start_trading_date(market="KR", asofdate=TODAY):
        logger.info(msg=f"[SKIP] VIEW performance comparison {TODAY:%Y-%m-%d}", extra=extra)
        return
    else:
        logger.info(msg=f"[PASS] start VIEW performance comparison. {TODAY:%Y-%m-%d}", extra=extra)
        
    # Data Preparation From DB
    view_table = db.TbViewInfo.query_df().sort_values(by="rebal_dt")

    core_df_total = pd.DataFrame()
    mlp_df_total = pd.DataFrame()
    alpha_df_total = pd.DataFrame()
    abl_df_total = pd.DataFrame()
    default_weights_df_total = pd.DataFrame()
    for year in range(2018,2024):
        for month in range(1,13):
            first_day_of_month = date(year, month, 1)
            
            if (first_day_of_month <= date(2018,2,1)):
                continue
            else:
                today = db.get_start_trading_date(market="KR",asofdate=first_day_of_month)
                df_view_table = view_table[view_table.rebal_dt == today]
                
                rebal_dt_core_view_table = pd.DataFrame()
                for target in df_view_table.target.tolist():
                    target_df_core_view_table = df_view_table[df_view_table.target == target][['rebal_dt','target','core_view']].set_index('rebal_dt')
                    target_df_core_view_table = target_df_core_view_table.pivot(columns='target', values="core_view")
                    rebal_dt_core_view_table = pd.concat([rebal_dt_core_view_table, target_df_core_view_table], axis=1)
                                
                core_df_total = pd.concat([core_df_total, rebal_dt_core_view_table])
                
                rebal_dt_ai_mlp_view_table = pd.DataFrame()
                for target in df_view_table.target.tolist():
                    target_df_ai_mlp_view_table = df_view_table[df_view_table.target == target][['rebal_dt','target','ai_mlp_view']].set_index('rebal_dt')
                    target_df_ai_mlp_view_table = target_df_ai_mlp_view_table.pivot(columns='target', values="ai_mlp_view")
                    rebal_dt_ai_mlp_view_table = pd.concat([rebal_dt_ai_mlp_view_table, target_df_ai_mlp_view_table], axis=1)
                                
                mlp_df_total = pd.concat([mlp_df_total, rebal_dt_ai_mlp_view_table])
                
                rebal_dt_ai_alpha_view_table = pd.DataFrame()
                for target in df_view_table.target.tolist():
                    target_df_ai_alpha_view_table = df_view_table[df_view_table.target == target][['rebal_dt','target','ai_alpha_view']].set_index('rebal_dt')
                    target_df_ai_alpha_view_table = target_df_ai_alpha_view_table.pivot(columns='target', values="ai_alpha_view")
                    rebal_dt_ai_alpha_view_table = pd.concat([rebal_dt_ai_alpha_view_table, target_df_ai_alpha_view_table], axis=1)
                                
                alpha_df_total = pd.concat([alpha_df_total, rebal_dt_ai_alpha_view_table])

                rebal_dt_ai_factor_view_table = pd.DataFrame() 
                for target in df_view_table.target.tolist():
                    target_df_ai_factor_view_table = df_view_table[df_view_table.target == target][['rebal_dt','target','ai_factor_view']].set_index('rebal_dt')
                    target_df_ai_factor_view_table = target_df_ai_factor_view_table.pivot(columns='target', values="ai_factor_view")
                    rebal_dt_ai_factor_view_table = pd.concat([rebal_dt_ai_factor_view_table, target_df_ai_factor_view_table], axis=1)
                                
                abl_df_total = pd.concat([abl_df_total, rebal_dt_ai_factor_view_table])
                
                rebal_dt_default_weight_table = pd.DataFrame()
                for target in df_view_table.target.tolist():
                    target_df_default_weight_table = df_view_table[df_view_table.target == target][['rebal_dt','target','default_weight']].set_index('rebal_dt')
                    target_df_default_weight_table = target_df_default_weight_table.pivot(columns='target', values="default_weight")
                    rebal_dt_default_weight_table = pd.concat([rebal_dt_default_weight_table, target_df_default_weight_table], axis=1)
                                
                default_weights_df_total = pd.concat([default_weights_df_total, rebal_dt_default_weight_table])
                
                
    # db_core_view_table.index = view_table.rebal_dt.unique()
    core_view = core_df_total[['미국','유럽','일본','중국(본토)','중국(홍콩)','신흥국','인디아','베트남','브라질','한국','러시아','선진국채','신흥국채','미국(IG)','미국(HY)','원자재','원유','금','부동산/리츠']]
    ai_mlp_view = mlp_df_total[['미국','유럽','일본','중국(본토)','중국(홍콩)','신흥국','인디아','베트남','브라질','한국','러시아','선진국채','신흥국채','미국(IG)','미국(HY)','원자재','원유','금','부동산/리츠']]
    ai_alpha_view = alpha_df_total[['미국','유럽','일본','중국(본토)','중국(홍콩)','신흥국','인디아','베트남','브라질','한국','러시아','선진국채','신흥국채','미국(IG)','미국(HY)','원자재','원유','금','부동산/리츠']]
    ai_factor_view = abl_df_total[['미국','유럽','일본','중국(본토)','중국(홍콩)','신흥국','인디아','베트남','브라질','한국','러시아','선진국채','신흥국채','미국(IG)','미국(HY)','원자재','원유','금','부동산/리츠']]
    default_weights_df_total = default_weights_df_total[['미국','유럽','일본','중국(본토)','중국(홍콩)','신흥국','인디아','베트남','브라질','한국','러시아','선진국채','신흥국채','미국(IG)','미국(HY)','원자재','원유','금','부동산/리츠']]
    default_weights_df_total.columns = ['SPY','VGK','EWJ','MCHI','EWH','VWO','INDA','VNM','EWZ','EWY','ERUS','GOVT','EBND','LQD','HYG','GSG','DBO','GLD','VNQ']
    
    tickers = [
        'ACWI','AGG',  # BM components
        'SPY','VGK','EWJ','MCHI','EWH','VWO','INDA','VNM','EWZ','EWY','ERUS',
        'GOVT','EBND', # '148070', we decided to except KR assets from performance comparison part
        'LQD','HYG',   # '239660', we decided to except KR assets from performance comparison part
        'GSG','DBO','GLD','VNQ'
    ]
    ticker_mapper = db.get_meta_mapper()

    prices = db.get_price(tickers=", ".join(tickers))
    prices = prices[(prices.index < TODAY.strftime("%Y-%m-%d")) & (prices.index >= '2018-01-01')][tickers]

    ####################################################################################################
    # Core View Preprocessing 
    tilting_core_view_7steps = np.add(np.multiply(np.subtract(core_view.iloc[:23,:], 4), 0.1), 1)
    tilting_core_view_5steps = np.add(np.multiply(np.subtract(core_view.iloc[23:,:], 3), 0.1), 1)
    tilting_core_view = pd.concat([tilting_core_view_7steps,tilting_core_view_5steps])
    tilted_core_weights = np.multiply(default_weights_df_total, tilting_core_view)

    core_equity = tilted_core_weights.iloc[:,:11].fillna(0)
    for i in range(len(core_equity)):
        weight_sum_asset_class = core_equity.iloc[i,:].sum()
        core_equity.iloc[i,:] = np.multiply(core_equity.iloc[i,:], 0.6/weight_sum_asset_class)
        core_equity.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=core_equity.iloc[i,:])).T

    core_treasury = tilted_core_weights.iloc[:,11:13].fillna(0)
    for i in range(len(core_treasury)):
        weight_sum_asset_class = core_treasury.iloc[i,:].sum()
        core_treasury.iloc[i,:] = np.multiply(core_treasury.iloc[i,:], 0.2/weight_sum_asset_class)
        core_treasury.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=core_treasury.iloc[i,:])).T

    core_credit = tilted_core_weights.iloc[:,13:15].fillna(0)
    for i in range(len(core_credit)):
        weight_sum_asset_class = core_credit.iloc[i,:].sum()
        core_credit.iloc[i,:] = np.multiply(core_credit.iloc[i,:], 0.1/weight_sum_asset_class)
        core_credit.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=core_credit.iloc[i,:])).T

    core_alternative = tilted_core_weights.iloc[:,15:19].fillna(0)
    for i in range(len(core_alternative)):
        weight_sum_asset_class = core_alternative.iloc[i,:].sum()
        core_alternative.iloc[i,:] = np.multiply(core_alternative.iloc[i,:], 0.1/weight_sum_asset_class)
        core_alternative.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=core_alternative.iloc[i,:])).T

    core_view_comprehensive_weights = pd.concat([core_equity,core_treasury,core_credit,core_alternative], axis=1)
    core_view_comprehensive_weights.index = pd.to_datetime(core_view_comprehensive_weights.index)
    core_allo = core_view_comprehensive_weights.stack().reset_index()
    core_allo.columns = ["rebal_dt", "ticker", "weights"]
    
    port_id = db.get_port_id(strategy="CV", market="US", level="CORE")
    core_allo["port_id"] = port_id
    core_allo["stk_id"] = core_allo.ticker.map(ticker_mapper)

    strategy_name = 'core_view'
    book, nav = strategy_analytics.calculate_nav(weight=core_view_comprehensive_weights, price=prices, strategy_name=strategy_name)
    core_book = book[strategy_name].copy()
    core_book["port_id"] = port_id
    core_nav = nav[strategy_name].copy()
    core_nav["port_id"] = port_id

    ####################################################################################################
    # Default Portfolio Preprocessing 
    default_port_comprehensive_weights = default_weights_df_total.copy()
    default_port_comprehensive_weights.index = pd.to_datetime(default_port_comprehensive_weights.index)
    default_port_allo = default_port_comprehensive_weights.stack().reset_index()
    default_port_allo.columns = ["rebal_dt", "ticker", "weights"]
    
    port_id = db.get_port_id(strategy="CV", market="US", level="DEFAULT")
    default_port_allo["port_id"] = port_id
    default_port_allo["stk_id"] = default_port_allo.ticker.map(ticker_mapper)

    strategy_name = 'default_port_view'
    book, nav = strategy_analytics.calculate_nav(weight=default_port_comprehensive_weights, price=prices, strategy_name=strategy_name)
    default_port_book = book[strategy_name].copy()
    default_port_book["port_id"] = port_id
    default_port_nav = nav[strategy_name].copy()
    default_port_nav["port_id"] = port_id

    ####################################################################################################
    # BM portfolio(ACWI 60% / AGG 40%)
    BM_comprehensive_weights = pd.DataFrame([[0.6,0.4]  for _ in range(len(core_view.index))],index=core_view.index, columns=["ACWI","AGG"])
    BM_comprehensive_weights.index = pd.to_datetime(BM_comprehensive_weights.index)
    port_id = db.get_port_id(strategy="BM", market="US", level=3)
    
    strategy_name = 'BM'
    book, nav = strategy_analytics.calculate_nav(weight=BM_comprehensive_weights, price=prices, strategy_name=strategy_name)
    BM_book = book[strategy_name].copy()
    BM_book["port_id"] = port_id
    BM_nav = nav[strategy_name].copy()
    BM_nav["port_id"] = port_id
    
    ####################################################################################################
    # AI MLP View Preprocessing 
    tilting_ai_mlp_view = np.add(np.multiply(np.subtract(ai_mlp_view, 3), 0.1), 1)
    tilted_ai_weights = np.multiply(default_weights_df_total, tilting_ai_mlp_view)

    ai_equity = tilted_ai_weights.iloc[:,:11].fillna(0)
    for i in range(len(ai_equity)):
        weight_sum_asset_class = ai_equity.iloc[i,:].sum()
        ai_equity.iloc[i,:] = np.multiply(ai_equity.iloc[i,:], 0.6/weight_sum_asset_class)
        ai_equity.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=ai_equity.iloc[i,:])).T

    ai_treasury = tilted_ai_weights.iloc[:,11:13].fillna(0)
    for i in range(len(ai_treasury)):
        weight_sum_asset_class = ai_treasury.iloc[i,:].sum()
        ai_treasury.iloc[i,:] = np.multiply(ai_treasury.iloc[i,:], 0.2/weight_sum_asset_class)
        ai_treasury.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=ai_treasury.iloc[i,:])).T

    ai_credit = tilted_ai_weights.iloc[:,13:15].fillna(0)
    for i in range(len(ai_credit)):
        weight_sum_asset_class = ai_credit.iloc[i,:].sum()
        ai_credit.iloc[i,:] = np.multiply(ai_credit.iloc[i,:], 0.1/weight_sum_asset_class)
        ai_credit.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=ai_credit.iloc[i,:])).T

    ai_alternative = tilted_ai_weights.iloc[:,15:19].fillna(0)
    for i in range(len(ai_alternative)):
        weight_sum_asset_class = ai_alternative.iloc[i,:].sum()
        ai_alternative.iloc[i,:] = np.multiply(ai_alternative.iloc[i,:], 0.1/weight_sum_asset_class)
        ai_alternative.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=ai_alternative.iloc[i,:])).T

    ai_mlp_view_comprehensive_weights = pd.concat([ai_equity,ai_treasury,ai_credit,ai_alternative], axis=1)
    ai_mlp_view_comprehensive_weights.index = pd.to_datetime(ai_mlp_view_comprehensive_weights.index)
    ai_allo_mlp = ai_mlp_view_comprehensive_weights.stack().reset_index()
    ai_allo_mlp.columns = ["rebal_dt", "ticker", "weights"]
    port_id = db.get_port_id(strategy="AI", market="US", level="MLP")
    
    ai_allo_mlp["port_id"] = port_id
    ai_allo_mlp["stk_id"] = ai_allo_mlp.ticker.map(ticker_mapper)

    strategy_name = 'ai_mlp_view'
    book, nav = strategy_analytics.calculate_nav(weight=ai_mlp_view_comprehensive_weights, price=prices, strategy_name=strategy_name)
    ai_book_mlp = book[strategy_name].copy()
    ai_book_mlp["port_id"] = port_id
    ai_nav_mlp = nav[strategy_name].copy()
    ai_nav_mlp["port_id"] = port_id
    

    # AI ALPHA View Preprocessing 
    tilting_ai_alpha_view = np.add(np.multiply(np.subtract(ai_alpha_view, 3), 0.1), 1)
    tilted_ai_weights = np.multiply(default_weights_df_total, tilting_ai_alpha_view)

    ai_equity = tilted_ai_weights.iloc[:,:11].fillna(0)
    for i in range(len(ai_equity)):
        weight_sum_asset_class = ai_equity.iloc[i,:].sum()
        ai_equity.iloc[i,:] = np.multiply(ai_equity.iloc[i,:], 0.6/weight_sum_asset_class)
        ai_equity.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=ai_equity.iloc[i,:])).T

    ai_treasury = tilted_ai_weights.iloc[:,11:13].fillna(0)
    for i in range(len(ai_treasury)):
        weight_sum_asset_class = ai_treasury.iloc[i,:].sum()
        ai_treasury.iloc[i,:] = np.multiply(ai_treasury.iloc[i,:], 0.2/weight_sum_asset_class)
        ai_treasury.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=ai_treasury.iloc[i,:])).T

    ai_credit = tilted_ai_weights.iloc[:,13:15].fillna(0)
    for i in range(len(ai_credit)):
        weight_sum_asset_class = ai_credit.iloc[i,:].sum()
        ai_credit.iloc[i,:] = np.multiply(ai_credit.iloc[i,:], 0.1/weight_sum_asset_class)
        ai_credit.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=ai_credit.iloc[i,:])).T

    ai_alternative = tilted_ai_weights.iloc[:,15:19].fillna(0)
    for i in range(len(ai_alternative)):
        weight_sum_asset_class = ai_alternative.iloc[i,:].sum()
        ai_alternative.iloc[i,:] = np.multiply(ai_alternative.iloc[i,:], 0.1/weight_sum_asset_class)
        ai_alternative.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=ai_alternative.iloc[i,:])).T

    ai_alpha_view_comprehensive_weights = pd.concat([ai_equity,ai_treasury,ai_credit,ai_alternative], axis=1)
    ai_alpha_view_comprehensive_weights.index = pd.to_datetime(ai_alpha_view_comprehensive_weights.index)
    ai_allo_alpha = ai_alpha_view_comprehensive_weights.stack().reset_index()
    ai_allo_alpha.columns = ["rebal_dt", "ticker", "weights"]
    port_id = db.get_port_id(strategy="AI", market="US", level="ALPHA")
    
    ai_allo_alpha["port_id"] = port_id
    ai_allo_alpha["stk_id"] = ai_allo_alpha.ticker.map(ticker_mapper)

    strategy_name = 'ai_alpha_view'
    book, nav = strategy_analytics.calculate_nav(weight=ai_alpha_view_comprehensive_weights, price=prices, strategy_name=strategy_name)
    ai_book_alpha = book[strategy_name].copy()
    ai_book_alpha["port_id"] = port_id
    ai_nav_alpha = nav[strategy_name].copy()
    ai_nav_alpha["port_id"] = port_id


    # AI FACTOR View Preprocessing 
    tilting_ai_factor_view = np.add(np.multiply(np.subtract(ai_factor_view, 3), 0.1), 1)
    tilted_ai_weights = np.multiply(default_weights_df_total, tilting_ai_factor_view)

    ai_equity = tilted_ai_weights.iloc[:,:11].fillna(0)
    for i in range(len(ai_equity)):
        weight_sum_asset_class = ai_equity.iloc[i,:].sum()
        ai_equity.iloc[i,:] = np.multiply(ai_equity.iloc[i,:], 0.6/weight_sum_asset_class)
        ai_equity.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=ai_equity.iloc[i,:])).T

    ai_treasury = tilted_ai_weights.iloc[:,11:13].fillna(0)
    for i in range(len(ai_treasury)):
        weight_sum_asset_class = ai_treasury.iloc[i,:].sum()
        ai_treasury.iloc[i,:] = np.multiply(ai_treasury.iloc[i,:], 0.2/weight_sum_asset_class)
        ai_treasury.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=ai_treasury.iloc[i,:])).T

    ai_credit = tilted_ai_weights.iloc[:,13:15].fillna(0)
    for i in range(len(ai_credit)):
        weight_sum_asset_class = ai_credit.iloc[i,:].sum()
        ai_credit.iloc[i,:] = np.multiply(ai_credit.iloc[i,:], 0.1/weight_sum_asset_class)
        ai_credit.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=ai_credit.iloc[i,:])).T

    ai_alternative = tilted_ai_weights.iloc[:,15:19].fillna(0)
    for i in range(len(ai_alternative)):
        weight_sum_asset_class = ai_alternative.iloc[i,:].sum()
        ai_alternative.iloc[i,:] = np.multiply(ai_alternative.iloc[i,:], 0.1/weight_sum_asset_class)
        ai_alternative.iloc[i,:] = pd.DataFrame(BaseStrategy.clean_weights(weights=ai_alternative.iloc[i,:])).T

    ai_factor_view_comprehensive_weights = pd.concat([ai_equity,ai_treasury,ai_credit,ai_alternative], axis=1)
    ai_factor_view_comprehensive_weights.index = pd.to_datetime(ai_factor_view_comprehensive_weights.index)
    ai_allo_factor = ai_factor_view_comprehensive_weights.stack().reset_index()
    ai_allo_factor.columns = ["rebal_dt", "ticker", "weights"]
    port_id = db.get_port_id(strategy="AI", market="US", level="ABL")
    
    ai_allo_factor["port_id"] = port_id
    ai_allo_factor["stk_id"] = ai_allo_factor.ticker.map(ticker_mapper)

    strategy_name = 'ai_factor_view'
    book, nav = strategy_analytics.calculate_nav(weight=ai_factor_view_comprehensive_weights, price=prices, strategy_name=strategy_name)
    ai_book_factor = book[strategy_name].copy()
    ai_book_factor["port_id"] = port_id
    ai_nav_factor = nav[strategy_name].copy()
    ai_nav_factor["port_id"] = port_id
    
    ########################################################################################################################
    if args.database == 'true':
        
        last_trading_date = db.get_last_trading_date("KR", TODAY)
        
        with db.session_local() as session:
            for port in db.TbPort.query().all():
                port_id = port.port_id
                portfolio = port.portfolio
                
                if portfolio.split("_")[0] not in ["BM", "CV", "AI"]:
                    continue
                                
                # TB_PORT_ALLOC
                recent_rebal_dt = (
                    session.query(sa.func.max(db.TbPortAlloc.rebal_dt))
                    .filter(
                        db.TbPortAlloc.weights.isnot(None),
                        db.TbPortAlloc.port_id == port_id,
                    )
                    .scalar()
                )
                for each_strategy_allocation in [default_port_allo, core_allo, ai_allo_mlp, ai_allo_alpha, ai_allo_factor]:
                    
                    if not recent_rebal_dt is None:
                        if recent_rebal_dt >= last_trading_date-timedelta(days=1):
                            continue
                    
                    if port_id in each_strategy_allocation.port_id.tolist():
                        each_strategy_allocation = each_strategy_allocation.set_index("rebal_dt").loc[recent_rebal_dt+timedelta(days=1):].reset_index()
                        try:
                            db.TbPortAlloc.insert(each_strategy_allocation)
                        except:
                            db.TbPortAlloc.update(each_strategy_allocation)

                # TB_PORT_BOOK
                recent_trd_dt_book = (
                    session.query(sa.func.max(db.TbPortBook.trd_dt))
                    .filter(
                        db.TbPortBook.weights.isnot(None),
                        db.TbPortBook.port_id == port_id,
                    )
                    .scalar()
                )
                for each_strategy_book in [core_book, default_port_book, BM_book, ai_book_mlp, ai_book_alpha, ai_book_factor]:
                    
                    if not recent_trd_dt_book is None:
                        if recent_trd_dt_book >= last_trading_date-timedelta(days=1):
                            continue
                    
                    if port_id in each_strategy_book.port_id.tolist():
                        each_strategy_book = each_strategy_book.loc[recent_trd_dt_book+timedelta(days=1):]
                        each_strategy_book = each_strategy_book.reset_index().rename(columns={'Date': 'trd_dt'})
                        each_strategy_book["stk_id"] = each_strategy_book.ticker.map(db.get_meta_mapper())
                        try:
                            db.TbPortBook.insert(each_strategy_book)
                        except:
                            db.TbPortBook.update(each_strategy_book)

                # TB_PORT_VALUE
                recent_trd_dt_value = (
                    session.query(sa.func.max(db.TbPortValue.trd_dt))
                    .filter(
                        db.TbPortValue.value.isnot(None),
                        db.TbPortValue.port_id == port_id,
                    )
                    .scalar()
                )
                for each_strategy_nav in [core_nav, default_port_nav, BM_nav, ai_nav_mlp, ai_nav_alpha, ai_nav_factor]:
                    
                    if not recent_trd_dt_value is None:
                        if recent_trd_dt_value >= last_trading_date-timedelta(days=1):
                            continue                    
                    
                    if port_id in each_strategy_nav.port_id.tolist():
                        each_strategy_nav = each_strategy_nav.loc[recent_trd_dt_value+timedelta(days=1):]
                        each_strategy_nav = each_strategy_nav.reset_index().rename(columns={'Date': 'trd_dt'})
                        each_strategy_nav["mdd_1y"] = each_strategy_nav.value.rolling(252).apply(portfolio_update.mdd)
                        each_strategy_nav["sharp_1y"] = each_strategy_nav.value.rolling(252).apply(portfolio_update.sharpe)
                        each_strategy_nav["mdd"] = each_strategy_nav.value.expanding(252).apply(portfolio_update.mdd)
                        each_strategy_nav["sharp"] = each_strategy_nav.value.expanding(252).apply(portfolio_update.sharpe)
                        try:
                            db.TbPortValue.insert(each_strategy_nav)
                        except:
                            db.TbPortValue.update(each_strategy_nav)

        logger.info(msg=f"[PASS] Insert ALLOC, BOOK, NAV. {TODAY:%Y-%m-%d}", extra=extra)
        
    else:
        total_nav = pd.concat(nav.values(), axis=1)
        total_nav.columns = nav.keys()
        total_nav = total_nav.fillna(method='ffill')
        strategy_analytics.result_metrics(total_nav)
    
    logger.info(msg=f"[PASS] End VIEW performance comparison. {TODAY:%Y-%m-%d}", extra=extra)
    