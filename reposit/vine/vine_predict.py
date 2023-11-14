from datetime import datetime, date, timedelta
import pandas as pd


TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)

def run_mlp_prediction() -> None:

    from core.strategy.mlpstrategy import PredSettings
    from core.analytics.prediction import BaggedMlpClassifier

    price_df = pd.read_excel("vine_universe.xlsx", sheet_name="Prices", index_col="Date", parse_dates=True).astype(float)

    pred_settings = PredSettings(
        train=True, save=False
    ).dict()
    prediction_model = BaggedMlpClassifier(**pred_settings)
    data = prediction_model.predict(price_df=price_df)

    data.to_clipboard()

run_mlp_prediction()