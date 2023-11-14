import os
import sys
import logging
import pandas as pd
import numpy as np
from dateutil import parser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))

from config import get_args
from hive import db

logger = logging.getLogger("sqlite")

args = get_args()

TODAY = parser.parse(args.date)
start_date = TODAY - pd.DateOffset(years=3)

logger.info(f"running calculating risk score script {TODAY:%Y-%m-%d}")


price = db.get_price_from_universe()
# price.index = pd.to_datetime(price.index)
price = price.pct_change()
price_3Y = price.loc[start_date:TODAY]

def cal(p: pd.Series) -> float:
    
    loss_rate_2p5 = np.percentile(p.dropna(), 2.5)
    absolute_loss_rate = abs(loss_rate_2p5)
    annualization_adjustment = np.sqrt(250)
    result = absolute_loss_rate * annualization_adjustment
    
    return result


result = price_3Y.aggregate(cal, axis=0).to_frame().reset_index()
result.columns = ["ticker", "vol"]
result["stk_id"] = result.ticker.map(db.get_meta_mapper())

conditions = [
    result['vol'] > 0.4,
    result['vol'] > 0.2,
    result['vol'] > 0.1,
    result['vol'] > 0.01,
]
values = [5, 4, 3, 2]

# Default value if no condition is met
default_value = 1

# Use numpy.select to set the 'riskscore' column based on the conditions
result['risk_score'] = np.select(conditions, values, default_value)

db.TbRiskScore.insert(result)


