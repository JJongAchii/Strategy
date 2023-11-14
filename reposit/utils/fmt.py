"""
Format functions
"""
import pandas as pd


def as_format(
    item: ...,
    format_str: str = ".2f"
) -> ...:
    """
    Map a format string over a object.

    Args:
        item (pd.Series, pd.DataFrame, other): item.
        decimals (int, optional): number of decimals. Defaults to 2.

    Returns:
        ...: formatted item
    """
    if isinstance(item, pd.Series):
        return item.dropna().map(lambda x: format(x, format_str))
    if isinstance(item, pd.DataFrame):
        return item.apply(as_format, format_str=format_str)
    return format(item, format_str)


def as_percent(item, decimals=2):
    """
    Map a format string over a object.

    Args:
        item (pd.Series, pd.DataFrame, other): item.
        decimals (int, optional): number of decimals. Defaults to 2.

    Returns:
        ...: formatted item
    """
    return as_format(item, f'.{decimals}%')


def as_float(item, decimals=2):
    """
    Map a format string over a object.

    Args:
        item (pd.Series, pd.DataFrame, other): item.
        decimals (int, optional): number of decimals. Defaults to 2.

    Returns:
        ...: formatted item
    """
    return as_format(item, f'.{decimals}f')


def as_date(date: ..., fmt: str = '%Y-%m-%d') -> str:
    """
    Format date string

    Args:
        date (any): any date
        fmt (str, optional): output date format.
        Defaults to '%Y-%m-%d'.

    Returns:
        str: date format
    """
    return pd.Timestamp(date).strftime(fmt)