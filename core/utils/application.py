import os
import sys
import logging
from dateutil import parser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import get_args
from core.utils import price_update

logger = logging.getLogger("sqlite")


if __name__ == "__main__":

    args = get_args()

    try:
        if args.script == "pricehistory":
            TODAY = parser.parse(args.date)
            price_update.run_upload_historical_data(end=TODAY)

        elif args.script == "pricekr":
            price_update.run_update_daily_KR()
            
        elif args.script == "priceus":
            price_update.run_update_daily_US()

        elif args.script == "index":
            from core.utils import index_update
            index_update.run_update_index()
            
        elif args.script == "fx":
            from core.utils import fx_update
            fx_update.run_update_fx()
    
    except Exception as error:
        extra_ = dict(user=args.user, activity=args.script, category="monitoring")
        
        logger.error(msg=f'[ERROR] {args.script}\n{error}', extra=extra_)