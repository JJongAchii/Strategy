from typing import Union, Dict
import pandas as pd
from hive_old import db


class Index:

    ticker: Union[str, Dict]

    def get_data(self) -> pd.DataFrame:

        if not hasattr(self, "ticker"):
            raise ValueError(f"{self} has no defined ticker.")

        if isinstance(self.ticker, str):
            return db.TbTimeSeries.price(tickers=self.ticker)

        if isinstance(self.ticker, dict):
            data = db.TbTimeSeries.price(tickers=list(self.ticker.keys()))
            data = data.rename(columns=self.ticker)
            return data

        raise ValueError(f"ticker must be str or dict, but {self.ticker} was given")


class Rate(Index):
    ticker: str = "LGY7TRUH"


class Equity(Index):
    ticker: str = "MXCXDMHR"


class Credit:
    ticker: Dict = dict(
        uscredit="LUACTRUU",
        eucredit="LP05TRUH",
        usjunk="LF98TRUU",
        eujunk="LP01TRUH",
    )
