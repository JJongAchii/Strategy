import os
import sys
import logging
import pandas as pd
from dateutil import parser
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../..")))
from hive import db
from config import get_args

logger = logging.getLogger("sqlite")


if __name__ == "__main__":

    args = get_args()
    TODAY = parser.parse(args.date)

    # try:
    if args.script:
        if args.script == "iml":
            
            from core.model.regime.iml import run_regime_iml
            state_iml = run_regime_iml()
            
        if args.script == "gmm":
            
            from core.model.regime.gmm import run_regime_gmm
            state_gmm = run_regime_gmm()
            
        if args.script == "lei":
            
            from core.model.regime.lei import run_regime_lei
            state_lei = run_regime_lei()
            
        if args.script == 'all':
            
            regime = 'IML'
            from core.model.regime.iml import run_regime_iml
            state_iml = run_regime_iml(regime, TODAY.date())
            
            regime = 'GMM'
            from core.model.regime.gmm import run_regime_gmm
            state_gmm = run_regime_gmm(regime, TODAY.date())
            
            regime = 'lei'
            from core.model.regime.lei import run_regime_lei
            state_lei = run_regime_lei(regime, TODAY.date())
            
            if state_iml.empty and state_gmm.empty and state_lei.empty:
                print("SKIP")
            else:
                print("yes")
                regime_state = pd.concat([state_iml,state_lei,state_gmm])
                print(regime_state)
                if args.database == 'true':
                    
                    try:
                        db.TbRegime.insert(regime_state)
                    except:
                        db.TbRegime.update(regime_state)

        if args.script == 'historical':
            
            for year in range(2021,2024):
                for month in range(1,13):
                    first_day_of_month = date(year, month, 1)
                    start_trading_date=db.get_start_trading_date(market="KR", asofdate=first_day_of_month)
                    if start_trading_date >= date(2021,1,1) and start_trading_date < date.today():
                        TODAY = start_trading_date

                        regime = 'IML'
                        from core.model.regime.iml import run_regime_iml
                        state_iml = run_regime_iml(regime, TODAY)
                        
                        regime = 'GMM'
                        from core.model.regime.gmm import run_regime_gmm
                        state_gmm = run_regime_gmm(regime, TODAY)
                        
                        regime = 'lei'
                        from core.model.regime.lei import run_regime_lei
                        state_lei = run_regime_lei(regime, TODAY)
                        
                        regime_state = pd.concat([state_iml,state_lei,state_gmm])
                        print(regime_state)
                    
                        if args.database == 'true':
                            
                            try:
                                db.TbRegime.insert(regime_state)
                            except:
                                db.TbRegime.update(regime_state)
                        
                        
    # except Exception as error:
    #     extra = dict(user=args.user, activity=args.script, category="monitoring")
    #     logger.error(msg=f'[ERROR] {args.script}\n{error}', extra=extra)