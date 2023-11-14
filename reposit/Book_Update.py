import pandas as pd

from hive_old import db
import sqlalchemy as sa


def get_portfolio_max_date():
    """ get the existing gross return data that has not been updated in the portfolio book"""
    with db.session_local() as session:
        return (
            session.query(
                sa.func.max(db.TbPortfolioBook.date).label('max_date'),
                db.TbPortfolioBook.portfolio_id
            )
            .group_by(db.TbPortfolioBook.portfolio_id)
        ).all()


def gross_return_until_max_date():
    with db.session_local() as session:
        query = (
            session.query(
                db.TbTimeSeries.date,
                db.TbTimeSeries.meta_id,
                db.TbTimeSeries.value
            )
            .filter(
                db.TbTimeSeries.date > port_num.max_date,
                db.TbTimeSeries.field == 'GROSS_RETURN'
            )
            .distinct()
        )

        return db.read_sql_query(query).sort_values('date')


def get_portfolio_allocation_list():
    with db.session_local() as session:
        query = (
            session.query(
                db.TbPortfolioAllocation.date,
                db.TbPortfolioAllocation.meta_id,
                db.TbPortfolioAllocation.weights
            )
            .filter(db.TbPortfolioAllocation.portfolio_id == port_num.portfolio_id)
            .distinct()
        )

        return db.read_sql_query(query).sort_values('date')


def updated_portfolio_max_date():
    with db.session_local() as session:
        return (
            session.query(
                sa.func.max(db.TbPortfolioBook.date).label("max_date"),
                db.TbPortfolioBook.portfolio_id
            )
            .filter(db.TbPortfolioBook.portfolio_id == port_num.portfolio_id)
            .group_by(db.TbPortfolioBook.portfolio_id)
            .subquery()
        )


def get_portfolio_book_at_max_date(subquery):
    """
    Use the subquery in the main query to get the rows
    corresponding to the maximum date and the specified portfolio_id
    """
    with db.session_local() as session:
        query = (
            session.query(
                db.TbPortfolioBook.portfolio_id,
                db.TbPortfolioBook.meta_id,
                db.TbPortfolioBook.weights
            )
            .join(
                subquery,
                (db.TbPortfolioBook.portfolio_id == subquery.c.portfolio_id) &
                (db.TbPortfolioBook.date == subquery.c.max_date)
            )
        )

        return db.read_sql_query(query)


def get_portfolio_value_at_max_date(subquery):
    with db.session_local() as session:
        query = (
            session.query(
                db.TbPortfolioValue.portfolio_id,
                db.TbPortfolioValue.value
            )
            .join(
                subquery,
                (db.TbPortfolioValue.portfolio_id == subquery.c.portfolio_id) &
                (db.TbPortfolioValue.date == subquery.c.max_date)
            )
        )

        return db.read_sql_query(query)


port_id = get_portfolio_max_date()

for port_num in port_id:

    return_df = gross_return_until_max_date()
    reb_date_df = get_portfolio_allocation_list()

    for trade_date in return_df['date'].unique():

        subquery = updated_portfolio_max_date()
        port_book_df = get_portfolio_book_at_max_date(subquery)
        port_val_df = get_portfolio_value_at_max_date(subquery)

        update_df = port_book_df.merge(return_df[return_df['date'] == trade_date], on='meta_id')
        weight_sum = sum(update_df['weights'])
        update_df['weights'] = update_df['weights'] * (1 + update_df['value'])
        if weight_sum != 0:
            weight_sum = sum(update_df['weights']) / weight_sum
        else:
            continue

        port_val_df['value'] = port_val_df['value'] * weight_sum
        port_val_df['date'] = trade_date

        update_df['weights'] = update_df['weights'] / sum(update_df['weights'])
        update_df = update_df.drop('value', axis=1)

        if trade_date in reb_date_df['date'].values:
            update_df = reb_date_df[reb_date_df['date'] == trade_date]
            update_df['portfolio_id'] = port_num.portfolio_id

        db.TbPortfolioBook.insert(update_df)
        db.TbPortfolioValue.insert(port_val_df)


