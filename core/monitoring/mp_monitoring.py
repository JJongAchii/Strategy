import os
import sys
import logging
from dateutil import parser
from datetime import timedelta
import pandas as pd
import sqlalchemy as sa


sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import get_args
from hive import db

logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)
YESTERDAY = TODAY - timedelta(days=1)
TENWEEKSAGO = TODAY - timedelta(weeks=10)

logger.info(f"running mp monitoring script {TODAY:%Y-%m-%d}")


def run_num_allocation():
    """check number of allocation holdings"""
    extra = dict(user=args.user, activity="num_allocation_check", category="monitoring")
    with db.session_local() as session:
        query = (
            session.query(
                db.TbPort.portfolio,
                db.TbPortAlloc.rebal_dt,
                sa.func.count(db.TbPortAlloc.stk_id).label("num_asset"),
            )
            .select_from(db.TbPortAlloc)
            .join(db.TbPort)
            .filter(db.TbPortAlloc.rebal_dt == TODAY)
            .group_by(db.TbPortAlloc.rebal_dt, db.TbPort.portfolio)
            .order_by(db.TbPortAlloc.rebal_dt)
        )

        data = db.read_sql_query(query)
        
        if data.empty:
            logger.info(msg=f"[SKIP] portfolio num check.", extra=extra)
            return
        
        checks = data[data.num_asset != 10]

        if checks.empty:
            logger.info("[PASS] Num Asset == 10.", extra=extra)
            return

        for _, check in checks.iterrows():
            port = check.get("portfolio")
            asofdate = check.get("rebal_dt")
            num_asset = check.get("num_asset")
            logger.warning(msg=f"[FAIL] {port} | Date: {asofdate} | Num Asset: {num_asset}", extra=extra)


def run_sum_allocation():
    """check number of allocation holdings"""
    extra = dict(user=args.user, activity="sum_allocation_check", category="monitoring")
    with db.session_local() as session:
        query = (
            session.query(
                db.TbPort.portfolio,
                db.TbPortAlloc.rebal_dt,
                sa.func.sum(db.TbPortAlloc.weights).label("sum_weight"),
            )
            .select_from(db.TbPortAlloc)
            .join(db.TbPort)
            .filter(db.TbPortAlloc.rebal_dt == TODAY)
            .group_by(db.TbPortAlloc.rebal_dt, db.TbPort.portfolio)
            .order_by(db.TbPortAlloc.rebal_dt)
        )

        data = db.read_sql_query(query)
        
        if data.empty:
            logger.info(msg=f"[SKIP] portfolio sum check.", extra=extra)
            return

        checks = data[data.sum_weight != 1.0]

        if checks.empty:
            logger.info("[PASS] Sum Weight == 100%", extra=extra)
            return

        for _, check in checks.iterrows():
            port = check.get("portfolio")
            asofdate = check.get("rebal_dt")
            sum_weight = check.get("sum_weight")
            logger.warning(msg=f"[FAIL] {port} | Date: {asofdate} | Sum Weight: {sum_weight*100}%", extra=extra)
            
            
def run_risk_score():
    """check risk score of portfolio"""
    extra = dict(user=args.user, activity="risk_score_check", category="monitoring")
    with db.session_local() as session:
        query = (
            session.query(
                db.VwPortRiskScore,
                db.TbPort.portfolio,
                db.TbPort.gbi2_cd
            )
            .filter(db.VwPortRiskScore.trd_dt == TODAY)
            .join(db.TbPort, db.TbPort.port_id == db.VwPortRiskScore.port_id)
        )
        
        data = db.read_sql_query(query)
        
        if data.empty:
            logger.info(msg=f"[SKIP] portfolio risk score check.", extra=extra)
            return
        
        checks = data[(data.risk_score > 2.2) & (data.gbi2_cd == 2) |
                    (data.risk_score > 3.2) & (data.gbi2_cd == 3) |
                    (data.risk_score > 4.2) & (data.gbi2_cd == 4)]

        if checks.empty:
            logger.info("[PASS] No riskscores exceed the threshold", extra=extra)
            return

        for _, check in checks.iterrows():
            port = check.get("portfolio")
            asofdate = check.get("trd_dt")
            risk_score = check.get("risk_score")
            logger.warning(msg=f"[FAIL] {port} | Date: {asofdate} | Risk Score: {risk_score}", extra=extra)
            
            
def run_turnover_check():
    """check assets turnover in portfolio"""
    extra = dict(user=args.user, activity="turn_over_check", category="monitoring")
    last_month = TODAY - timedelta(days=TODAY.day)
    last_month_m = last_month.month
    last_month_y = last_month.year
    
    with db.session_local() as session:
        query = (
            session.query(
                db.TbPortAlloc.rebal_dt,
                db.TbPort.portfolio,
                db.TbMeta.ticker,
                db.TbPortAlloc.weights
            )
            .filter(
                (db.TbPortAlloc.rebal_dt == TODAY) |
                ((sa.extract('month', db.TbPortAlloc.rebal_dt) == last_month_m) &
                (sa.extract('year', db.TbPortAlloc.rebal_dt) == last_month_y))
            )
            .join(db.TbPort, db.TbPort.port_id == db.TbPortAlloc.port_id)
            .join(db.TbMeta, db.TbMeta.stk_id == db.TbPortAlloc.stk_id)
        )
        data = db.read_sql_query(query=query, parse_dates="rebal_dt")
    
    curr_port = data[data.rebal_dt == TODAY]
    prev_port = data[data.rebal_dt != TODAY]
    
    merg_port = curr_port.merge(prev_port, on=["portfolio", "ticker"], how="left").fillna(0)
    merg_port["turnover"] = (merg_port.weights_x - merg_port.weights_y).abs()
    
    if curr_port.empty:
        logger.info(msg=f"[SKIP] turn over check.", extra=extra)
        return

    checks = merg_port[merg_port.turnover >= 0.5]
    
    if checks.empty:
        logger.info("[PASS] No turnovers exceed the threshold", extra=extra)
        return

    for _, check in checks.iterrows():
        port = check.get("portfolio")
        asofdate = check.get("rebal_dt_x")
        ticker = check.get("ticker")
        turnover = check.get("turnover")
        logger.warning(msg=f"[FAIL] {port} | Date: {asofdate:%Y-%m-%d} | Ticker: {ticker} | Turn Over: {turnover}", extra=extra)
