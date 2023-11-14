"""
Weighting Functions
"""
import numpy as np
import pandas as pd


def exponential(
    item: pd.DataFrame,
    alpha: float = 0.05,
    com: float = None,
    span: float = None,
    halflife: float = None,
) -> pd.DataFrame:
    """
    Transform data into exponentially weighted.

    Args:
        item (pd.DataFrame): data to be transformed.
        alpha (float, optional): exponential alpha. Defaults to 0.05.
        com (float, optional): center of mass. Defaults to None.
        span (float, optional): decay span. Defaults to None.
        halflife (float, optional): decay halflife. Defaults to None.

    Returns:
        pd.DataFrame: exponentially weighted data
    """
    # Calculte exponential alpha if not provided.
    if com is not None:
        alpha = 1 / (1 + com)
    if span is not None:
        alpha = 2 / (span + 1)
    if halflife is not None:
        alpha = 1 - np.exp(-np.log(2) / halflife)

    window = len(item)

    weight = pd.Series(
        data = np.flip(list((1 - alpha) ** i for i in range(window))),
        index = item.index,
    )

    if isinstance(item, pd.DataFrame):
        return item.multiply(weight, axis=0)
    result = item.multiply(weight)
    result.name = item.name
    return result