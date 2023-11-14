import os
import sys
import logging
import pandas as pd
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import get_args
from hive import db

logger = logging.getLogger("sqlite")


if __name__ == "__main__":
    
    args = get_args()
    
    try:
        if args.script:
            if args.script == "dbshares":

                from core.analytics.utils import portfolio_update

                portfolio_update.update_shares()
                
            elif args.script == "dbnav":

                from core.analytics.utils import portfolio_update

                portfolio_update.run_update_db_portfolio()

            elif args.script == "view":
                
                from core.analytics.coreview import view_maker, view_performance

                view_maker.run_coreview_process(today=pd.to_datetime(args.date).date())
                view_performance.run_view_performance_update()

            elif args.script == "portexprtn":

                from core.analytics.simulation.port_exp_rtn import PortExpectedReturn

                port_expected_return=PortExpectedReturn()
                port_expected_return.run_port_exp_rtn()

    except Exception as error:
        extra = dict(user=args.user, activity=args.script, category="monitoring")
        
        logger.error(msg=f'[ERROR] {args.script}\n{error}', extra=extra)