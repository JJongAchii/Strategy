from pydantic.main import BaseModel
import numpy as np
import pandas as pd
import sqlalchemy as sa
from hive import db


def clean_weights(weights: pd.Series, decimals: int = 4, tot_weight=None) -> pd.Series:
    """Clean weights based on the number decimals and maintain the total of weights.

    Args:
        weights (pd.Series): asset weights.
        decimals (int, optional): number of decimals to be rounded for
            weight. Defaults to 4.

    Returns:
        pd.Series: clean asset weights.
    """
    # clip weight values by minimum and maximum.
    if not tot_weight:
        tot_weight = weights.sum().round(4)
    weights = weights.round(decimals=decimals)
    # repeat round and weight calculation.
    for _ in range(10):
        weights = weights / weights.sum() * tot_weight
        weights = weights.round(decimals=decimals)
        if weights.sum() == tot_weight:
            return weights
    # if residual remains after repeated rounding.
    # allocate the the residual weight on the max weight.
    residual = tot_weight - weights.sum()
    # !!! Error may occur when there are two max weights???
    weights.iloc[np.argmax(weights)] += np.round(residual, decimals=decimals)
    return weights


class AssetClassSumWeight(BaseModel):
    """total of asset weight for each asset class"""

    equity: float
    fixedincome: float
    alternative: float
    liquidity: float

    @classmethod
    def from_level(cls, level: int = 5) -> "AssetClassSumWeight":
        """get asset class weight based on the risk level"""
        if level == 1:
            return cls(
                equity=0.05,
                fixedincome=0.10,
                alternative=0.05,
                liquidity=0.80,
            )
        if level == 2:
            return cls(
                equity=0.10,
                fixedincome=0.10,
                alternative=0.10,
                liquidity=0.70,
            )
        if level == 3:
            return cls(
                equity=0.30,
                fixedincome=0.15,
                alternative=0.05,
                liquidity=0.50,
            )
        if level == 4:
            return cls(
                equity=0.50,
                fixedincome=0.20,
                alternative=0.05,
                liquidity=0.25,
            )
        if level == 5:
            return cls(
                equity=0.80,
                fixedincome=0.10,
                alternative=0.05,
                liquidity=0.05,
            )
        raise NotImplementedError(
            "level only takes integers from 1 to 5. " + f"but {level} was given."
        )


with db.session_local() as session:
    for market in ["us", "kr"]:
        for level in [3, 4, 5]:
            # result = []

            data = db.read_sql_query(
                session.query(db.TbPortAlloc, db.TbUniverse.strg_asset_class)
                .join(db.TbPort)
                .join(
                    db.TbUniverse,
                    sa.and_(
                        db.TbUniverse.stk_id == db.TbPortAlloc.stk_id,
                        db.TbUniverse.strategy_id == db.TbPort.strategy_id,
                    ),
                )
                .filter(db.TbPort.portfolio == f"abl_{market}_{level}".upper())
            )
            acsw = AssetClassSumWeight.from_level(level=level).dict()

            print(data.groupby("rebal_dt").weights.sum())
            for rebal_dt in data.rebal_dt.unique():
                reb_data = data[data.rebal_dt == rebal_dt]

                check_weight = 0.0
                check_number = 0
                for strg_asset_class in reb_data.strg_asset_class.unique():
                    ac_data = reb_data[reb_data.strg_asset_class == strg_asset_class]
                    tot_weight = acsw[strg_asset_class]
                    print(ac_data.weights.sum(), tot_weight)
                    check_weight += tot_weight
                    check_number += len(ac_data)
                    # print(ac_data.weights.sum(), tot_weight)
                    assert ac_data.weights.sum().round(4) == tot_weight
                    # weights = clean_weights(ac_data.weights, 4, tot_weight=tot_weight)
                    # if ac_data.weights.equals(weights):
                    #     print("pass")
                    #     continue
                    # ac_data["weights"] = weights
                    # db.TbPortAlloc.update(ac_data)
                assert round(check_weight, 4) == 1.00, f"{check_weight}, \n {reb_data}"
                assert round(check_number, 4) == 10, f"{check_number}, \n {reb_data}"