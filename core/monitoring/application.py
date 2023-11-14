import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import get_args


logger = logging.getLogger("sqlite")

if __name__ == "__main__":

    args = get_args()

    try:
        if args.script == "mp":
            
            from core.monitoring import mp_monitoring
            
            mp_monitoring.run_num_allocation()
            mp_monitoring.run_sum_allocation()
            mp_monitoring.run_risk_score()
            mp_monitoring.run_turnover_check()
            
        elif args.script == "pricekr":
            
            from core.monitoring import price_monitoring
            
            price_monitoring.kr_price_monitoring()

        elif args.script == "priceus":
            
            from core.monitoring import price_monitoring
            
            price_monitoring.us_price_monitoring()

        elif args.script == "index":
            
            from core.monitoring import index_monitoring
            
            index_monitoring.index_monitoring()

        elif args.script == "fx":
            
            from core.monitoring import fx_monitoring
            
            fx_monitoring.fx_monitoring()
            
    except Exception as error:
        extra_ = dict(user=args.user, activity=args.script, category="monitoring")
        logger.error(msg=f'[ERROR] {args.script}\n{error}', extra=extra_)