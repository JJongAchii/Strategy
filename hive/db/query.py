from functools import lru_cache
from typing import Optional, Union, List
from dateutil import parser
from dateutil.relativedelta import relativedelta
from datetime import datetime, date, timedelta
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Query
from .client import engine, session_local
from .models import (
    TbRiskScore,
    TbStrategy,
    TbMeta,
    TbMetaClass,
    TbUniverse,
    TbTicker,
    TbDailyBar,
    TbHoliday,
    TbPort,
    TbPortAlloc,
    TbPortBook,
    TbPortValue,
    TbPortApBook,
    TbPortApValue,
    TbLens,
    TbMacro,
    TbMacroData,
    TbFX,
    TbProbIncrease,
    TbMetaUpdat,
    TbInvstStyRtn,
    TbViewInfo,
)


def read_sql_query(query: Query, **kwargs) -> pd.DataFrame:
    """Read sql query

    Args:
        query (Query): sqlalchemy.Query

    Returns:
        pd.DataFrame: read the query into dataframe.
    """
    return pd.read_sql_query(
        sql=query.statement,
        con=query.session.bind,
        index_col=kwargs.get("index_col", None),
        parse_dates=kwargs.get("parse_dates", None),
    )


def get_start_trading_date(market: str = "KR", asofdate: Optional[date] = None) -> date:
    with session_local() as session:
        query = session.query(TbHoliday.hol_dt).filter(TbHoliday.market == market)
        hol_df = read_sql_query(query)

        first_day_of_month = date(asofdate.year, asofdate.month, 1)

        if first_day_of_month.weekday() >= 5 or first_day_of_month in hol_df.values:
            current_day = first_day_of_month + timedelta(days=1)
            while current_day.weekday() >= 5 or current_day in hol_df.values:
                current_day += timedelta(days=1)
            start_trading_date = current_day
        else:
            start_trading_date = first_day_of_month
    return start_trading_date


def get_last_trading_date(market: str = "US", asofdate: Optional[date] = None) -> date:
    with session_local() as session:
        query = session.query(TbHoliday.hol_dt).filter(TbHoliday.market == market)
        hol_df = read_sql_query(query)
        current_day = (asofdate - timedelta(days=1)).date()
        if current_day.weekday() >= 5 or current_day in hol_df.values:
            while current_day.weekday() >= 5 or current_day in hol_df.values:
                current_day -= timedelta(days=1)

    return current_day


@lru_cache(maxsize=2)
def load_universe(
    strategy: Optional[str] = None, asofdate: Optional[str] = None
) -> pd.DataFrame:
    with session_local() as session:
        subq = session.query(
            TbRiskScore.stk_id, sa.func.max(TbRiskScore.trd_dt).label("max_date")
        ).group_by(TbRiskScore.stk_id)

        if asofdate:
            subq = subq.filter(TbRiskScore.trd_dt <= parser.parse(asofdate))

        subq = subq.subquery()

        query = (
            session.query(
                TbStrategy.strategy,
                TbMeta.ticker,
                TbMeta.stk_id,
                TbTicker.ticker_bloomberg,
                TbMeta.isin,
                TbMeta.name,
                TbMeta.iso_code,
                TbUniverse.wrap_asset_class_code,
                TbUniverse.strg_asset_class,
                TbMetaClass.asset_class_nm,
                TbMetaClass.region_nm,
                TbRiskScore.risk_score,
            )
            .select_from(TbUniverse)
            .join(TbStrategy, TbStrategy.strategy_id == TbUniverse.strategy_id)
            .join(TbRiskScore, TbRiskScore.stk_id == TbUniverse.stk_id)
            .join(TbMeta, TbMeta.stk_id == TbUniverse.stk_id)
            .join(TbMetaClass, TbMetaClass.stk_id == TbUniverse.stk_id)
            .join(TbTicker, TbTicker.stk_id == TbUniverse.stk_id)
            .join(
                subq,
                sa.and_(
                    subq.c.stk_id == TbRiskScore.stk_id,
                    subq.c.max_date == TbRiskScore.trd_dt,
                ),
            )
            .filter(TbUniverse.active == 1)
        )

        if strategy:
            query = query.filter(TbStrategy.strategy == strategy.upper())

        universe = read_sql_query(query, index_col="ticker")
        universe['asset_class'] = universe['region_nm'] + universe['asset_class_nm'] + universe['risk_score'].map(
            {1: '형(초저위험)', 2: '형(저위험)', 3: '형(중위험)', 4: '형(고위험)', 5: '형(초고위험)'})

        return universe

@lru_cache(maxsize=2)
def load_universe_for_predict(
    strategy: Optional[str] = None
) -> pd.DataFrame:
    with session_local() as session:
        
        query = (
            session.query(
                TbMeta.ticker,
                TbMeta.stk_id,
                TbUniverse.strg_asset_class
            )
            .select_from(TbUniverse)
            .join(TbStrategy, TbStrategy.strategy_id == TbUniverse.strategy_id)
            .join(TbMeta, TbMeta.stk_id == TbUniverse.stk_id)
            .distinct()
        )

        if strategy:
            query = query.filter(TbStrategy.strategy.ilike(f"%{strategy.upper()}%"))

        universe = read_sql_query(query, index_col="ticker")
        
        return universe
    
def get_universe_theme(universe_stk_id: list):
    
    with session_local() as session:
        query = (
            session.query(
                TbUniverse
            )
            .select_from(TbUniverse)
            .filter(TbUniverse.stk_id.in_(universe_stk_id),
                    )
        )    
        data = read_sql_query(query=query)
        
        return data

def get_tb_macro_data():
        
    with session_local() as session:
        query = (
            session.query(
                TbMacro.ticker,
                TbMacro.future_ticker,
                TbMacroData
            )
            .select_from(TbMacroData)
            .join(TbMacro, TbMacro.macro_id == TbMacroData.macro_id)
        )
        data = read_sql_query(query=query)
        
        return data

def get_universe(
    strategy: Optional[str] = None, asofdate: Optional[str] = None
) -> pd.DataFrame:
    with session_local() as session:
        subq = session.query(
            TbRiskScore.stk_id, sa.func.max(TbRiskScore.trd_dt).label("max_date")
        ).group_by(TbRiskScore.stk_id)

        if asofdate:
            subq = subq.filter(TbRiskScore.trd_dt <= parser.parse(asofdate))

        subq = subq.subquery()

        query = (
            session.query(
                TbStrategy.strategy,
                TbMeta.ticker,
                TbMeta.stk_id,
                TbTicker.ticker_bloomberg,
                TbMeta.isin,
                TbMeta.name,
                TbMeta.iso_code,
                TbUniverse.wrap_asset_class_code,
                TbUniverse.strg_asset_class,
                TbMetaClass.asset_class_nm,
                TbMetaClass.region_nm,
                TbRiskScore.risk_score,
            )
            .select_from(TbUniverse)
            .filter(TbUniverse.active == 1)
        )

        if strategy:
            query = query.filter(TbStrategy.strategy == strategy.upper())

        universe = read_sql_query(query, index_col="ticker")
       
        return universe
    
@lru_cache(maxsize=2)
def get_price(tickers: Union[str, List] = "SPY, AGG") -> pd.DataFrame:
    tickers = (
        tickers
        if isinstance(tickers, (list, set, tuple))
        else tickers.replace(",", " ").upper().split()
    )

    with session_local() as session:
        query = (
            session.query(TbDailyBar.trd_dt, TbDailyBar.adj_value, TbMeta.ticker)
            .select_from(TbDailyBar)
            .filter(TbMeta.ticker.in_(tickers))
            .join(TbMeta, TbMeta.stk_id == TbDailyBar.stk_id)
        )

        return read_sql_query(query, parse_dates="trd_dt").pivot(
            index="trd_dt", columns="ticker", values="adj_value"
        )


def get_index_value(tickers: str = "SPX Index") -> pd.DataFrame:
    tickers = (
        tickers
        if isinstance(tickers, (list, set, tuple))
        else tickers.replace(",", " ")
    )

    with session_local() as session:
        query = (
            session.query(TbMacroData.trd_dt, TbMacroData.value, TbMacro.ticker)
            .select_from(TbMacroData)
            .filter(TbMacro.ticker == tickers)
            .join(TbMacro, TbMacro.macro_id == TbMacroData.macro_id)
        )

        return read_sql_query(query, parse_dates="trd_dt").pivot(
            index="trd_dt", columns="ticker", values="value"
        )


@lru_cache(maxsize=2)
def get_volume(tickers: Union[str, List] = "SPY, AGG") -> pd.DataFrame:
    tickers = (
        tickers
        if isinstance(tickers, (list, set, tuple))
        else tickers.replace(",", " ").upper().split()
    )

    with session_local() as session:
        query = (
            session.query(TbMetaUpdat.trd_dt, TbMetaUpdat.trd_volume, TbMeta.ticker)
            .select_from(TbMetaUpdat)
            .filter(TbMeta.ticker.in_(tickers))
            .join(TbMeta, TbMeta.stk_id == TbMetaUpdat.stk_id)
        )

        return read_sql_query(query, parse_dates="trd_dt").pivot(
            index="trd_dt", columns="ticker", values="trd_volume"
        )


@lru_cache(maxsize=2)
def get_aum(tickers: Union[str, List] = "SPY, AGG") -> pd.DataFrame:
    tickers = (
        tickers
        if isinstance(tickers, (list, set, tuple))
        else tickers.replace(",", " ").upper().split()
    )

    with session_local() as session:
        query = (
            session.query(TbMetaUpdat.trd_dt, TbMetaUpdat.aum, TbMeta.ticker)
            .select_from(TbMetaUpdat)
            .filter(TbMeta.ticker.in_(tickers))
            .join(TbMeta, TbMeta.stk_id == TbMetaUpdat.stk_id)
        )

        return read_sql_query(query, parse_dates="trd_dt").pivot(
            index="trd_dt", columns="ticker", values="aum"
        )


@lru_cache(maxsize=2)
def get_close_price(tickers: Union[str, List] = "SPY, AGG") -> pd.DataFrame:
    tickers = (
        tickers
        if isinstance(tickers, (list, set, tuple))
        else tickers.replace(",", " ").upper().split()
    )

    with session_local() as session:
        query = (
            session.query(TbDailyBar.trd_dt, TbDailyBar.close_prc, TbMeta.ticker)
            .select_from(TbDailyBar)
            .filter(TbMeta.ticker.in_(tickers))
            .join(TbMeta, TbMeta.stk_id == TbDailyBar.stk_id)
        )

        return read_sql_query(query, parse_dates="trd_dt").pivot(
            index="trd_dt", columns="ticker", values="close_prc"
        )


def get_meta_mapper() -> dict:
    mapper = {}

    with session_local() as session:
        query = session.query(TbMeta.ticker, TbMeta.stk_id)

        for rec in query.all():
            mapper[rec.ticker] = rec.stk_id

    return mapper


def get_portfolio_id(portfolio: str) -> int:
    with session_local() as session:
        return (
            session.query(TbPort.port_id)
            .filter(TbPort.portfolio == portfolio.upper())
            .scalar()
        )



def delete_portfolio_from_date(portfolio: int, asofdate: date):
    with session_local() as session:
        session.query(TbPortBook).filter(sa.and_(TbPortBook.port_id == portfolio, TbPortBook.trd_dt > asofdate)).delete()
        session.query(TbPortValue).filter(sa.and_(TbPortValue.port_id == portfolio, TbPortValue.trd_dt > asofdate)).delete()
        session.commit()


def get_portfolio_allocation(
    portfolio: str = "MLP_US_3", asofdate: Optional[date] = None
):
    with session_local() as session:
        query = (
            session.query(TbMeta.ticker, TbPortAlloc.weights)
            .select_from(TbPortAlloc)
            .join(TbMeta)
            .join(TbPort)
        )

        query = query.filter(TbPortAlloc.rebal_dt == asofdate)
        query = query.filter(TbPort.portfolio == portfolio)

        return read_sql_query(query=query, index_col="ticker").squeeze()


def get_portfolio_list():
    with session_local() as session:
        return TbPort.query_df()


def get_portfolio_min_date(portfolio: str):
    with session_local() as session:
        return (
            session.query(
                sa.func.min(TbPortValue.trd_dt).label("min_date")
            )
            .join(TbPort,
                sa.and_(TbPort.port_id == TbPortValue.port_id,
                        TbPort.portfolio == portfolio.upper())
                )
        ).scalar()


def get_portfolio_first_date(portfolio: str):
    with session_local() as session:
        subquery = (
            session.query(
                sa.func.min(TbPortAlloc.rebal_dt).label("min_date"),
                TbPortAlloc.port_id
            )
            .join(TbPort,
                sa.and_(TbPort.port_id == TbPortAlloc.port_id,
                        TbPort.portfolio == portfolio.upper())
                )
            .group_by(TbPortAlloc.port_id)
        ).subquery()

        query = (
            session.query(
                subquery.c.min_date,
                subquery.c.port_id,
                TbPortAlloc.stk_id,
                TbPortAlloc.weights,
            )
            .join(subquery,
                sa.and_(subquery.c.port_id == TbPortAlloc.port_id,
                        subquery.c.min_date == TbPortAlloc.rebal_dt)
                )
        )
        return read_sql_query(query)


def get_portfolio_max_date(portfolio: str):
    """get the existing gross return data that has not been updated in the portfolio book"""
    with session_local() as session:
        return (
            session.query(
                sa.func.max(TbPortValue.trd_dt).label("max_date")
            )
            .join(TbPort,
                sa.and_(TbPort.port_id == TbPortValue.port_id,
                        TbPort.portfolio == portfolio.upper())
                )
        ).scalar()


def trading_date_until_max_date(max_date):
    with session_local() as session:
        query = (
            session.query(TbDailyBar.trd_dt)
            .filter(TbDailyBar.trd_dt > max_date)
            .distinct()
        )
        return read_sql_query(query).sort_values("trd_dt")


def get_portfolio_allocation_list(portfolio):
    with session_local() as session:
        query = (
            session.query(TbPortAlloc.rebal_dt.label("trd_dt"), TbPortAlloc.port_id, TbPortAlloc.stk_id, TbPortAlloc.weights)
            .join(TbPort,
                sa.and_(TbPort.port_id == TbPortAlloc.port_id,
                        TbPort.portfolio == portfolio.upper())
                )
            .distinct()
        )
        return read_sql_query(query).sort_values("trd_dt")


def get_portfolio_book_at_max_date(portfolio: str, max_date: date):
    """
    Use the subquery in the main query to get the rows
    corresponding to the maximum date and the specified portfolio_id
    """
    with session_local() as session:
        query = (
            session.query(
                TbPortBook.port_id,
                TbPortBook.stk_id,
                TbPortBook.weights
            )
            .join(
                TbPort,
                sa.and_(
                    TbPort.port_id == TbPortBook.port_id,
                    TbPort.portfolio == portfolio.upper()
                )
            )
            .filter(TbPortBook.trd_dt == max_date)
        )
        return read_sql_query(query)


def get_portfolio_value_at_max_date(portfolio: str, max_date: date):
    with session_local() as session:
        query = (
            session.query(
                TbPortValue.port_id,
                TbPortValue.value
            )
            .join(
                TbPort,
                sa.and_(
                    TbPort.port_id == TbPortValue.port_id,
                    TbPort.portfolio == portfolio.upper()
                )
            )
            .filter(TbPortValue.trd_dt == max_date)
        )
        return read_sql_query(query)


def get_gross_return_at_trading_date(trade_date: date, stk_list: pd.Series):
    with session_local() as session:
        query = (
            session.query(
                TbDailyBar.trd_dt,
                TbDailyBar.stk_id,
                TbDailyBar.gross_rtn
            )
            .filter(
                TbDailyBar.trd_dt == trade_date,
                TbDailyBar.stk_id.in_(stk_list)
            )
        )
        return read_sql_query(query)


@lru_cache(maxsize=1)
def get_lens(today):
    with session_local() as session:
        query = (session.query(TbLens)
        .filter(TbLens.trd_dt < today)
        .order_by(TbLens.trd_dt))        
        return read_sql_query(query, parse_dates="trd_dt").pivot(
            index="trd_dt", columns="factor", values="value"
        )


def get_macro_id():
    with session_local() as session:
        query = (session.query(TbMacro)       
        )
        
        return read_sql_query(query)


def get_macro_data(asofdate: Optional[date] = datetime.strptime('20100101','%Y%m%d')):
    with session_local() as session:
        query = (session.query(TbMacroData)
        .filter(TbMacroData.trd_dt >= asofdate)         
        )
        
        return read_sql_query(query, parse_dates="trd_dt")
        

def get_macro_data_by_created_date(asofdate: Optional[date] = datetime.strptime('20100101','%Y%m%d')):
    with session_local() as session:
        query = (session.query(TbMacroData)
        .filter(TbMacroData.created_date >= asofdate)         
        )
        
        return read_sql_query(query, parse_dates="created_date")


def get_lei(asofdate: date):
    with session_local() as session:
        query = (
            session.query(TbMacroData.trd_dt, TbMacroData.adj_value, TbMacro.name)
            .join(TbMacro)
            .filter(TbMacro.ticker == "OEUSKLAC Index")
            .filter(TbMacroData.trd_dt <= asofdate)
        )
        return (
            read_sql_query(query, parse_dates="trd_dt").pivot(
                index="trd_dt", columns="name", values="adj_value"
            )
            - 100
        )
    

def get_last_trading_date_price(date:datetime, market:str):
    """get one last adj_value from DB"""
    with session_local() as session:
        query = session.query(TbHoliday.hol_dt).filter(TbHoliday.market == market)
        hol_df = read_sql_query(query)
        current_day = date.date() - timedelta(days=1)
        if current_day.weekday() >= 5 or current_day in hol_df.values:
            while current_day.weekday() >= 5 or current_day in hol_df.values:
                current_day -= timedelta(days=1)

    adj_value_df_last = TbDailyBar.query_df(trd_dt=current_day)[
        ["trd_dt", "stk_id", "close_prc", "gross_rtn", "adj_value"]] 

    return adj_value_df_last


def get_last_trading_date_index(date:datetime):
    """get last Bloomberg index data from DB"""
    with session_local() as session:
        query = session.query(TbHoliday.hol_dt).filter(TbHoliday.market == 'US')
        hol_df = read_sql_query(query)
        current_day = date.date() - timedelta(days=1)
        if current_day.weekday() >= 5 or current_day in hol_df.values:
            while current_day.weekday() >= 5 or current_day in hol_df.values:
                current_day -= timedelta(days=1)

    index_df_last = get_macro_data(current_day)

    return index_df_last


def get_last_trading_date_index_by_created_date(date:datetime):
    """get last Bloomberg index data from DB"""
    with session_local() as session:
        query = session.query(TbHoliday.hol_dt).filter(TbHoliday.market == 'US')
        hol_df = read_sql_query(query)
        current_day = date.date() - timedelta(days=1)
        if current_day.weekday() >= 5 or current_day in hol_df.values:
            while current_day.weekday() >= 5 or current_day in hol_df.values:
                current_day -= timedelta(days=1)

    index_df_last_created_date = get_macro_data_by_created_date(current_day)
    index_df_last_created_date.created_date = index_df_last_created_date.created_date.apply(lambda x: x.date())

    return index_df_last_created_date


def get_last_two_trading_dates_price(date:datetime, market:str):
    """get two last adj_value from DB"""
    with session_local() as session:
        query = session.query(TbHoliday.hol_dt).filter(TbHoliday.market == market)
        hol_df = read_sql_query(query)
        last_date_list = []
        current_day = (date - timedelta(days=1)).date()
        if current_day.weekday() >= 5 or current_day in hol_df.values:
            while current_day.weekday() >= 5 or current_day in hol_df.values:
                current_day -= timedelta(days=1)
                last_trading_date = current_day
            last_date_list.append(last_trading_date)
            second_last_trading_date = last_trading_date - timedelta(days=1)
            if second_last_trading_date.weekday() >= 5 or second_last_trading_date in hol_df.values:
                current_day = second_last_trading_date - timedelta(days=1)
                while current_day.weekday() >= 5 or current_day in hol_df.values:
                    current_day -= timedelta(days=1)
                second_last_trading_date = current_day
                last_date_list.append(second_last_trading_date)
            else:
                second_last_trading_date = second_last_trading_date
                last_date_list.append(second_last_trading_date)
        else:
            last_trading_date = current_day
            last_date_list.append(last_trading_date)
            second_last_trading_date = last_trading_date - timedelta(days=1)
            if second_last_trading_date.weekday() >= 5 or second_last_trading_date in hol_df.values:
                current_day = second_last_trading_date - timedelta(days=1)
                while current_day.weekday() >= 5 or current_day in hol_df.values:
                    current_day -= timedelta(days=1)
                second_last_trading_date = current_day
                last_date_list.append(second_last_trading_date)
            else:
                second_last_trading_date = second_last_trading_date
                last_date_list.append(second_last_trading_date)

    adj_value_df_last = TbDailyBar.query_df(trd_dt=last_date_list[0])[
        ["trd_dt", "stk_id", "close_prc", "gross_rtn", "adj_value"]] 
    adj_value_df_2nd_last = TbDailyBar.query_df(trd_dt=last_date_list[1])[
        ["trd_dt", "stk_id", "close_prc", "gross_rtn", "adj_value"]] 

    return adj_value_df_last, adj_value_df_2nd_last


def get_market(tickers: Union[str, List] = "SPY, AGG") -> pd.DataFrame:
    tickers = (
        tickers
        if isinstance(tickers, (list, set, tuple))
        else tickers.replace(",", " ").upper().split()
    )

    with session_local() as session:
        query = (
            session.query(
                TbMeta.ticker, TbMeta.iso_code
            )
            .filter(TbMeta.ticker.in_(tickers))
            )

        return read_sql_query(query).set_index("ticker")
    

def get_fx(currency: Union[str, List] = "KRW, USD") -> pd.DataFrame:
    currency = (
        currency
        if isinstance(currency, (list, set, tuple))
        else currency.replace(",", " ").upper().split()
    )
    
    with session_local() as session:
        query = (
            session.query(
                TbFX
            )
            
            .filter(TbFX.currency.in_(currency))
        )
        
        return read_sql_query(query).set_index("trd_dt")
    

def get_alloc_weight_for_shares(
    strategy: str = "MLP", market: str = "US", level: Union[int, str] = 5
) -> pd.DataFrame:
    
    with session_local() as session:
            query = (
                    session.query(
                        TbPortAlloc,
                        TbMeta.ticker
                    )   
                    .select_from(TbPortAlloc)
                    .join(
                        TbMeta,
                        TbMeta.stk_id == TbPortAlloc.stk_id
                    )
                    .join(
                        TbDailyBar,
                        sa.and_(
                            TbDailyBar.trd_dt == TbPortAlloc.rebal_dt,
                            TbDailyBar.stk_id == TbPortAlloc.stk_id,
                        ),
                    )
                    .join(TbPort, sa.and_(
                        TbPort.portfolio == (f"{strategy}_{market}_{level}"),
                        TbPort.port_id == TbPortAlloc.port_id
                    ))
                )
            return read_sql_query(query=query, parse_dates="rebal_dt")
        
def get_alloc_weight(strategy: str) -> pd.DataFrame:
    
    with session_local() as session:
        query = (
                session.query(
                    TbPortAlloc,
                    TbMeta.ticker
                )   
                .select_from(TbPortAlloc)
                .join(
                    TbMeta,
                    TbMeta.stk_id == TbPortAlloc.stk_id
                )
                .join(TbPort, sa.and_(
                    TbPort.portfolio == (f"{strategy}"),
                    TbPort.port_id == TbPortAlloc.port_id
                ))
            )
        return read_sql_query(query=query, parse_dates="rebal_dt")
        
          
def delete_portfolio_info(strategy: str = None, market: str = None, level: int = None):
    with session_local() as session:
        session.query(TbPortBook).filter(
                TbPortBook.port_id.in_(
                    session.query(TbPortBook.port_id)
                    .join(TbPort, sa.and_(
                        TbPort.portfolio == (f"{strategy}_{market}_{level}"),
                        TbPort.port_id == TbPortBook.port_id
                    ))
                    .subquery()
                )
            ).delete(synchronize_session=False)
        session.query(TbPortValue).filter(
                TbPortValue.port_id.in_(
                    session.query(TbPortValue.port_id)
                    .join(TbPort, sa.and_(
                        TbPort.portfolio == (f"{strategy}_{market}_{level}"),
                        TbPort.port_id == TbPortValue.port_id
                    ))
                    .subquery()
                )
            ).delete(synchronize_session=False)
        session.commit()
        

def delete_ap_portfolio_info(strategy: str = None, market: str = None, level: int = None):
    with session_local() as session:
        session.query(TbPortApBook).filter(
                TbPortApBook.port_id.in_(
                    session.query(TbPortApBook.port_id)
                    .join(TbPort, sa.and_(
                        TbPort.portfolio == (f"{strategy}_{market}_{level}"),
                        TbPort.port_id == TbPortApBook.port_id
                    ))
                    .subquery()
                )
            ).delete(synchronize_session=False)
        session.query(TbPortApValue).filter(
                TbPortApValue.port_id.in_(
                    session.query(TbPortApValue.port_id)
                    .join(TbPort, sa.and_(
                        TbPort.portfolio == (f"{strategy}_{market}_{level}"),
                        TbPort.port_id == TbPortApValue.port_id
                    ))
                    .subquery()
                )
            ).delete(synchronize_session=False)
        session.commit()
        

def get_port_id(strategy: str = None, market: str = None, level: int = None) -> int:
    with session_local() as session:
        return (
            session.query(TbPort.port_id)
            .filter(TbPort.portfolio == (f"{strategy}_{market}_{level}"))
            .scalar()
        )

def get_mlp_port_id():
    with session_local() as session:

        query = (
            session.query(
                TbPort.port_id,
                TbPort.portfolio
            )
            .filter(
                TbPort.portfolio.like('MLP_%')
            )
            .order_by(TbPort.portfolio)
        )
        data = read_sql_query(query)
        data['risk_level']=data['portfolio'].str[-1]
        return data


def delete_asset_port_alloc(rebal_dt: date, port_id: int, stk_id: List):

    with session_local() as session:
        session.query(TbPortAlloc).filter(
            sa.and_(
                TbPortAlloc.rebal_dt == rebal_dt, 
                TbPortAlloc.port_id == port_id,
                TbPortAlloc.stk_id.in_(stk_id)
            )
        ).delete()
        session.commit()
        print(
            f"delete in tb_port_alloc: {len(stk_id)} records complete."
        )

         
def delete_isin_prob_increase(rebal_dt: date, isin: List):

    with session_local() as session:
        session.query(TbProbIncrease).filter(
            sa.and_(
                TbProbIncrease.rebal_dt == rebal_dt, 
                TbProbIncrease.isin.in_(isin)
            )
        ).delete()
        session.commit()
        print(
            f"delete in tb_prob_increase: {len(isin)} records complete."
        )
        

def get_probability_increase(market: Optional[str] = None) -> pd.DataFrame:
    
    with session_local() as session:
        query = (
            session.query(
                TbProbIncrease
            )
        )
        if market:
            query = query.filter(TbProbIncrease.iso_code == market.upper())

                
        return read_sql_query(query=query, parse_dates="rebal_dt")
        
        
@lru_cache(maxsize=2)        
def get_price_from_universe():
    
    with session_local() as session:
        query = (
            session.query(
                TbDailyBar.trd_dt, 
                TbDailyBar.adj_value, 
                TbMeta.ticker
            )
            .select_from(TbUniverse)
            .join(TbMeta, TbMeta.stk_id == TbUniverse.stk_id)
            .join(TbDailyBar, TbDailyBar.stk_id == TbUniverse.stk_id)
            .distinct()
        )
        
        return read_sql_query(query, parse_dates="trd_dt").pivot(
            index="trd_dt", columns="ticker", values="adj_value"
        )
    

def get_port_value(port_id: List):
    
    with session_local() as session:
        query = (
            session.query(
                TbPortValue.trd_dt, 
                TbPortValue.port_id, 
                TbPortValue.value
            )
            .select_from(TbPortValue)
            .filter(TbPortValue.port_id.in_(port_id))
        )
        
        return read_sql_query(query, parse_dates="trd_dt").pivot(
            index="trd_dt", columns="port_id", values="value"
        )

def get_macro_data_from_ticker(tickers: Union[str, List] = "SPX Index, KOSPI Index", asofdate: Optional[date] = datetime.strptime('20100101','%Y%m%d'))  -> pd.DataFrame:
    tickers = (
        tickers
        if isinstance(tickers, (list, set, tuple))
        else tickers.replace(",", " ").upper().split()
    )

    with session_local() as session:
        query = (
            session.query(TbMacroData.trd_dt, TbMacroData.adj_value, TbMacro.ticker)
            .select_from(TbMacroData)
            .filter(
            sa.and_(
            TbMacro.ticker.in_(tickers),
            TbMacroData.trd_dt >= asofdate
            )
            )
            .join(TbMacro, TbMacro.macro_id == TbMacroData.macro_id)         
        )
        
        return read_sql_query(query, parse_dates="trd_dt").pivot(
            index="trd_dt", columns="ticker", values="adj_value"
        )
        
        
def get_port_style(port_id: int):
    with session_local() as session:
        return (
            session.query(TbPort.gbi)
            .filter(TbPort.port_id == port_id)
            .scalar()
        )


def get_IML_data(
        start: Optional[date] = datetime.strptime('20090601','%Y%m%d'), 
        end: Optional[date] = datetime.strptime('20230630','%Y%m%d')
) -> pd.DataFrame:
    with session_local() as session:
        query = (
            session.query(
                TbMacro.ticker,
                TbMacroData.trd_dt,
                TbMacroData.adj_value,
            )
            .join(TbMacro, TbMacro.macro_id == TbMacroData.macro_id)
            .filter(TbMacro.memo == 'IML')
            .filter(sa.and_(TbMacroData.trd_dt >= start, TbMacroData.trd_dt < end))
            .order_by(TbMacroData.macro_id)
            .order_by(TbMacroData.trd_dt)
        )
        data = read_sql_query(query).pivot(index="trd_dt", columns="ticker", values="adj_value")
        data.columns = ["term","inflation","treasury"]
        data["short_yield"] = data["treasury"] - data["term"] - data["inflation"]
        data = data[["short_yield","inflation"]]

    return data


def get_GMM_data(
        start: Optional[date] = datetime.strptime('20090601','%Y%m%d'), 
        end: Optional[date] = datetime.strptime('20230630','%Y%m%d'),
        rolling: int = 5
) -> pd.DataFrame:
    with session_local() as session:
        query = (
            session.query(
                TbDailyBar.stk_id,
                TbDailyBar.trd_dt,
                TbDailyBar.gross_rtn
            )
            .join(TbMeta, sa.and_(
                TbMeta.stk_id == TbDailyBar.stk_id,
                TbMeta.source != 'bloomberg'
                )
                )
            .filter(sa.and_(TbDailyBar.trd_dt >= start, TbDailyBar.trd_dt < end))
            .order_by(TbDailyBar.stk_id)
            .order_by(TbDailyBar.trd_dt)
        )
        data = read_sql_query(query) 

    onedata = pd.DataFrame()
    onedata["trd_dt"] = pd.DataFrame(sorted(data.trd_dt.unique(),reverse=True))
    for stk_id in data.stk_id.unique().tolist():
        id_data = data[data["stk_id"]==int(f'{stk_id}')][["trd_dt","gross_rtn"]]
        if len(id_data) < rolling:
            pass
        else:
            id_data.columns = ["trd_dt",f"{stk_id}"]
            onedata = pd.merge(left=onedata, right=id_data, how="left", left_on="trd_dt", right_on="trd_dt")

    return onedata


def check_weekend_holiday_date(market: str = "US", asofdate: Optional[date] = None) -> date:
    with session_local() as session:
        query = session.query(TbHoliday.hol_dt).filter(TbHoliday.market == market)
        hol_df = read_sql_query(query)
        unless_trading_date = False
        current_day = asofdate.date()
        
        if market == "US":
            if current_day.weekday() == 6 or current_day.weekday() == 0 or current_day in hol_df.values:
                unless_trading_date = True

        if market == "KR":
            if current_day.weekday() >= 5 or current_day in hol_df.values:
                unless_trading_date = True

    return unless_trading_date
