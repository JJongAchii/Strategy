import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import args
from utils import price_update


if args.script == "historicaldata":
    price_update.run_upload_historical_data()

elif args.script == "krdailydata":
    price_update.run_update_daily_KR()
    
elif args.script == "usdailydata":
    price_update.run_update_daily_US()

