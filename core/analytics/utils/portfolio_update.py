import os
import sys
import logging
from dateutil import parser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../..")))
from config import get_args
from core.analytics.va import VirtualAccount
from core.analytics import backtest
from hive import db

logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)

logger.info(f"running portfolio update script {TODAY:%Y-%m-%d}")


STRATEGY_PARAMS = {
    "MLP": {
        "markets": ["US", "KR"],
        "levels": [1, 2, 3, 4, 5, "2_IRP", "3_IRP", "4_IRP"],
        "currency": None,
        "port_value": None
    },
    "ABL": {
        "markets": ["US", "KR"],
        "levels": [3, 4, 5, "5_IML", "5_GMM"],
        "currency": None,
        "port_value": None
    },
    "DWS": {
        "markets": ["US", "KR"],
        "levels": [f"{i}_{j}" for i in range(1, 6) for j in range(1, 6)],
        "currency": None,
        "port_value": None
    },
    "KB": {
        "markets": ["DAM"],
        "levels": [5],
        "currency": "KRW",
        "port_value": 3_000_000
    }
}


def update_shares():
    
    extra = dict(user=args.user, activity="update portfolio shares", category="script")
    logger.info(msg=f"[PASS] start update portfolio shares. {TODAY:%Y-%m-%d}", extra=extra)

    for strategy, params in STRATEGY_PARAMS.items():
        markets = params["markets"]
        levels = params["levels"]
        currency = params["currency"]
        port_value = params["port_value"]
            
        for market in markets:
            for level in levels:
                account = VirtualAccount(strategy=strategy, market=market, level=level, portfolio_value=port_value)
                if account.weights.empty:
                    continue
                book, nav = account.calculate_virtual_account_nav(weight=account.weights, currency=currency)
                account.update_shares_db()
                
                db.delete_ap_portfolio_info(strategy=strategy, market=market, level=level)
                port_id = db.get_port_id(strategy=strategy, market=market, level=level)
                
                nav["port_id"] = port_id
                nav = nav.reset_index().rename(columns={'Date': 'trd_dt'})
                
                book["port_id"] = port_id
                book = book.reset_index().rename(columns={'Date': 'trd_dt'})
                book["stk_id"] = book.ticker.map(db.get_meta_mapper())
                
                db.TbPortApValue.insert(nav)
                db.TbPortApBook.insert(book)

                        
    logger.info("[PASS] portfolio shares update complete.")


def mdd(x):
    return (x / x.expanding().max() - 1).min()


def sharpe(x):
    re = x.pct_change().dropna().fillna(0)

    return re.mean() / re.std() * (252 ** 0.5)


def run_update_db_portfolio() -> None:
    
    extra = dict(user=args.user, activity="update portfolio nav", category="script")
    logger.info(msg=f"[PASS] start update portfolio nav. {TODAY:%Y-%m-%d}", extra=extra)
    
    for strategy, params in STRATEGY_PARAMS.items():
        markets = params["markets"]
        levels = params["levels"]
        currency = params["currency"]
            
        for market in markets:
            for level in levels:
                weights = db.get_alloc_weight_for_shares(strategy=strategy, market=market, level=level)
                if weights.empty:
                    continue
                weights = weights.pivot(index="rebal_dt", columns="ticker", values="weights")
                weights.index.name = "Date"
                
                strategy_name = f"{strategy}_{market}_{level}"
                book, nav = backtest.calculate_nav(weight=weights, currency=currency, strategy_name=strategy_name)
                book = book[strategy_name]
                nav = nav[strategy_name]                
                
                db.delete_portfolio_info(strategy=strategy, market=market, level=level)
                port_id = db.get_port_id(strategy=strategy, market=market, level=level)
                
                nav["port_id"] = port_id
                nav = nav.reset_index().rename(columns={'Date': 'trd_dt'})
                
                book["port_id"] = port_id
                book = book.reset_index().rename(columns={'Date': 'trd_dt'})
                book["stk_id"] = book.ticker.map(db.get_meta_mapper())
                
                nav["mdd_1y"] = nav.value.rolling(252).apply(mdd)
                nav["sharp_1y"] = nav.value.rolling(252).apply(sharpe)
                nav["mdd"] = nav.value.expanding(252).apply(mdd)
                nav["sharp"] = nav.value.expanding(252).apply(sharpe)
                
                db.TbPortValue.insert(nav)
                db.TbPortBook.insert(book)


    logger.info("[PASS] portfolio nav update complete.")