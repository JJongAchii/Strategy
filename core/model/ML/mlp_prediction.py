import os
import sys
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Union
from dateutil import parser
from datetime import timedelta
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier


sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../..")))
from config import get_args, ML_MODELS_FOLDER
from core.strategy import utils
from hive import db

logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)
YESTERDAY = TODAY - timedelta(days=1)


class BaggedMlpClassifier:
    """asset return prediction class binary"""

    def __init__(
        self,
        train: bool = False,
        save: bool = False,
        model_path: str = "/",
        lags: int = 21,
        lookback_window: int = 50,
        hidden_layer_sizes: tuple = (256,),
        random_state: int = 100,
        max_iter: int = 1000,
        early_stopping: bool = True,
        validation_fraction: float = 0.15,
        shuffle: bool = False,
        n_estimators: int = 100,
        max_features: float = 0.50,
        max_samples: float = 0.50,
        bootstrap: bool = False,
        bootstrap_features: bool = False,
        n_jobs: int = -1,
        min_periods: int = 252,
        forward_window: int = 21,
    ) -> None:
        """initialize"""
        self.train = train
        self.save = save
        self.model_path = model_path
        self.lags = lags
        self.lookback_window = lookback_window

        # Base estimator - Multi-Layer Perceptron.
        base_estimator = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            shuffle=shuffle,
        )

        # Bagging Classifier with the base estimator.
        self.model = BaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        self.min_periods = min_periods
        self.forward_window = forward_window

    def make_features(self, price_series: pd.Series):
        """create features based on the price series data"""

        data = pd.DataFrame(price_series.dropna())
        data.columns = ["p"]
        # Initialize store of features.
        pre_features = ["r", "min", "max", "mom", "vol"]
        # Make first layer of features.
        data["r"] = np.log1p(data["p"].pct_change().fillna(0))
        data["min"] = (
            data["p"].rolling(self.lookback_window).min().divide(data["p"]) - 1
        )
        data["max"] = (
            data["p"].rolling(self.lookback_window).max().divide(data["p"]) - 1
        )
        data["mom"] = data["r"].rolling(self.lookback_window).mean()
        data["vol"] = data["r"].rolling(self.lookback_window).std()
        # Make moving average features.
        for win in [5, 20, 50, 200]:
            if win <= self.lookback_window:
                data[f"sma{win}"] = (
                    data["p"].rolling(win).mean().divide(data["p"]).subtract(1)
                )
                pre_features.append(f"sma{win}")

        # Make target
        data["target"] = (
            data["p"].pct_change(self.forward_window).shift(-self.forward_window)
        )

        features = []
        for pre_feature in pre_features:
            for lag in range(0, self.lags + 1):
                name = f"{pre_feature}_lag_{lag}"
                data[name] = data[pre_feature].shift(lag)
                features.append(name)
        target_x = data[features].iloc[-1:]
        data.dropna(inplace=True)
        train_x, train_y = data[features], data["target"]

        return train_x, train_y, target_x

    def load_model(self, name: str) -> Union[BaggingClassifier, None]:
        """load model based on the name"""

        fullpath = os.path.join(self.model_path, f"{name}_model.sav")
        print(fullpath)
        try:
            model = pickle.load(open(fullpath, "rb"))
            return model
        except Exception as exception:
            warnings.warn(f"problem with load model at {fullpath}")
            return None

    def save_model(self, name: str) -> bool:
        """save model based on the name"""
        fullpath = os.path.join(self.model_path, f"{name}_model.sav")
        if not os.path.exists(self.model_path):
            warnings.warn(f"model path does not exist, creating path {fullpath}")
            os.makedirs(self.model_path)
        try:
            pickle.dump(self.model, open(fullpath, "wb"))
            return True
        except Exception as exception:
            warnings.warn(f"problem with saving model at {fullpath}")
            return False

    def predict(self, price_df: pd.DataFrame) -> pd.Series:
        """_summary_

        Args:
            price_df (pd.DataFrame): _description_

        Returns:
            pd.Series: _description_
        """
        probability = {}

        for ticker in price_df:
            try:
                prob = self.predict_one(price_df[ticker])
                print(f"{ticker}: {prob:.4f}")
                probability[ticker] = prob
            except Exception as exception:
                print(exception)
                warnings.warn("error predicting model.")

        return pd.Series(probability)

    def predict_one(self, price_series: pd.Series) -> float:
        """_summary_

        Args:
            price_series (pd.Series): _description_

        Returns:
            float: _description_
        """
        name = str(price_series.name)
        train_x, train_y, target_x = self.make_features(price_series=price_series)
        if self.train:
            self.model.fit(train_x, np.where(train_y > 0, 1, 0))
            if self.save:
                self.save_model(name=name)
            return self.model.predict_proba(target_x).flatten()[1]

        fullpath = os.path.join(self.model_path, f"{name}_model.sav")
        model = pickle.load(open(fullpath, "rb"))
        if model is None:
            return 0.0
        prob = model.predict_proba(target_x).flatten()[1]
        del model
        return prob


def run_mlp_prediction() -> None:
    """
    run mlp prediction at the half year start
    i.e. Jan 1st & Jul 1st of each year.
    """
    extra = dict(user=args.user, activity="mlp_prediction", category="script")
    if not utils.is_start_of_half_year(TODAY=TODAY):
        logger.info(msg=f"[SKIP] mlp prediction. {TODAY:%Y-%m-%d}", extra=extra)
        return

    from core.strategy.mlpstrategy import PredSettings

    logger.info(msg=f"[PASS] start mlp prediction. {TODAY:%Y-%m-%d}", extra=extra)
    model_path = os.path.join(
        ML_MODELS_FOLDER, utils.get_ml_model_training_date(asofdate=TODAY).strftime("%Y-%m-%d")
    )
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    universe = db.load_universe_for_predict(strategy=f"mlp")
    price_df = db.get_price(tickers=", ".join(universe.index.tolist())).loc[:YESTERDAY]
    
    for asset_class in universe.strg_asset_class.unique():
        ac_tickers = universe[
            universe.strg_asset_class == asset_class
        ].index
        if asset_class in ["fixedincome", "liquidity"]:
            pred_settings = PredSettings(
                train=True, save=True, model_path=model_path, max_samples= 1.0
            ).dict()
        elif asset_class in ["equity", "alternative"]:
            pred_settings = PredSettings(
                train=True, save=True, model_path=model_path, max_samples= 0.5
            ).dict()
            
        prediction_model = BaggedMlpClassifier(**pred_settings)
        prediction_model.predict(price_df=price_df[ac_tickers])
    
    logger.info(
        msg=f"[PASS] mlp prediction. {TODAY:%Y-%m-%d}", extra=extra
    )