import os
import argparse
from datetime import date


PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(PROJECT_FOLDER, "output")
LOGDB_FOLDER = os.path.join(PROJECT_FOLDER, "log")
ML_MODELS_FOLDER = os.path.join(OUTPUT_FOLDER, "ml_models")
COREVIEW_ML_MODELS_FOLDER = os.path.join(OUTPUT_FOLDER, "ml_models/coreview")
ALLOC_FOLDER = os.path.join(OUTPUT_FOLDER, "alloc")



####################################################################################################
def get_args():
    # parse arguments
    parse = argparse.ArgumentParser(description="Running Script.")
    parse.add_argument("-s", "--script")
    parse.add_argument("-d", "--date", default=date.today().strftime("%Y-%m-%d"))
    parse.add_argument("-u", "--user")
    parse.add_argument("-r", "--regime", default = 'lei')
    parse.add_argument("-db", "--database")
    args = parse.parse_args()
    
    return args