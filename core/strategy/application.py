import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import get_args

logger = logging.getLogger("sqlite")


if __name__ == "__main__":
    
    args = get_args()
    
    try:
        if args.script:
            if args.script == "predict":
                from core.model.ML import mlp_prediction
                mlp_prediction.run_mlp_prediction()
                
            elif args.script == "mlp" or args.script == "dws":
                from core.strategy import mlpstrategy
                mlpstrategy.run_mlp_allocation()
                
            elif args.script == "abl":
                from core.strategy import ablstrategy
                ablstrategy.run_abl_allocation()

            elif args.script == "mlpirp":
                from core.strategy import mlpirpstrategy
                mlpirpstrategy.run_mlp_irp_allocation()

    except Exception as error:
        extra_ = dict(user=args.user, activity=args.script, category="monitoring")
        
        logger.error(msg=f'[ERROR] {args.script}\n{error}', extra=extra_)
