import os
import sys
import logging
from dateutil import parser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../..")))
from config import get_args

logger = logging.getLogger("sqlite")


if __name__ == "__main__":

    args = get_args()
    args.script = "equities"
    try:
        if args.script:
            if args.script == "equities":
                
                from core.analytics.cma import equities
                equities.run_equities_market_assumption()
                
            elif args.script == "fixedincome":
                
                from core.analytics.cma import fixedincome
                fixedincome.run_fixedincome_market_assumption()
                
            elif args.script == "liquidities":
                
                from core.analytics.cma import liquidities
                liquidities.run_liquidities_market_assumption()
                
            elif args.script == "commodities":
                
                from core.analytics.cma import commodities
                commodities.run_commodities_market_assumption()
                
            elif args.script == "all":
                
                from core.analytics.cma import equities
                equities.run_equities_market_assumption()
                
                from core.analytics.cma import fixedincome
                fixedincome.run_fixedincome_market_assumption()
                
                from core.analytics.cma import liquidities
                liquidities.run_liquidities_market_assumption()
                
                from core.analytics.cma import commodities
                commodities.run_commodities_market_assumption()
                
                         
    except Exception as error:
        extra = dict(user=args.user, activity=args.script, category="monitoring")
        
        logger.error(msg=f'[ERROR] {args.script}\n{error}', extra=extra)