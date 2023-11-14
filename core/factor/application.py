import os
import sys
import logging
from dateutil import parser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import get_args

logger = logging.getLogger("sqlite")


if __name__ == "__main__":

    args = get_args()                                               
    try:
        if args.script:
            if args.script == "factorlens":
                
                from core.factor.factor_lens import FactorLens
                FactorLens()
                
            elif args.script == "factor_expectation":
                
                from core.factor.factor_expected_return import *
                calc_daily_exp_rtn()
            
                
    except Exception as error:
        extra = dict(user=args.user, activity=args.script, category="monitoring")
        
        logger.error(msg=f'[ERROR] {args.script}\n{error}', extra=extra)

        
