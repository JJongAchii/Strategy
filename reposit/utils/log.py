"""
Logging relatied utility functions.
"""
import sys
import time
import logging
import functools
import pandas as pd

def make_signature(*args, **kwargs):
    args_repr = []
    for a in args:
        args_repr.append(repr(a))
    kwargs_repr = []
    for key, value in kwargs.items():
        if isinstance(value, pd.DataFrame):
            kwargs_repr.append(f'{key} = {type(value)}({value.shape})')
        else:
            kwargs_repr.append(f"{key}={value!r}")
    signature = '\n' + ', \n'.join(args_repr + kwargs_repr)
    return signature

def debug(func:callable=None, message:str=None) -> None:
    """
    Debug decorator helps to track parameters and execution time.

    Args:
        func (callable, optional): function.
        message (str, optional): customized display message. Defaults to None.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, message=message, **kwargs):
            logger = get_logger(arg=func.__name__, level='DEBUG')
            if message is not None:
                logger.debug(message)
            signature = make_signature(*args, **kwargs)
            logger.debug(f"arguments: {signature}")
            try:
                start = pd.to_datetime(time.ctime())
                result = func(*args, **kwargs)
                end = pd.to_datetime(time.ctime())
                message = f"excecution time : {end - start}; memory: {sys.getsizeof(func)} bytes"
                logger.debug(message)
                return result
            except Exception as exception:
                logger.exception(f"Exception raised in {func.__name__}. exception: {str(exception)}")
                raise exception
        return wrapper
    if func is None:
        return decorator
    return decorator(func)

def func_scope(func) -> str:
    """
    Function scope name

    Args:
        func: python function

    Returns:
        str: module_name.func_name
    """
    cur_mod = sys.modules[func.__module__]
    return f'{cur_mod.__name__}.{func.__name__}'

def get_logger(
    arg: str or callable,
    level=logging.DEBUG,
    fmt: str = '%(asctime)s:%(name)s:%(levelname)s:%(message)s',
    stream: bool = True,
    filename: str = None,
) -> logging.Logger:
    """ Generate logger

    Args:
        arg (str, callable): logger name or running function.
        level (str, logging.LEVEL): specific log level.
        fmt (str): logger format.
            Default to `%(asctime)s:%(name)s:%(levelname)s:%(message)s`
        stream (bool): if add stream handler. Default to True.
        filename (str): if add file handler. Default to None.

    Returns:
        logging: logger
    """
    log_name = func_scope(arg) if callable(arg) else arg
    logger = logging.getLogger(log_name)
    if isinstance(level, str): level = getattr(logging, level.upper())
    logger.setLevel(level=level)
    if len(logger.handlers) == 0:
        formatter = logging.Formatter(fmt=fmt)
        if stream:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(fmt=formatter)
            logger.addHandler(stream_handler)
        if filename is not None:
            if not filename.endswith('.log'): filename += '.log'
            file_handler = logging.FileHandler(filename=filename)
            file_handler.setFormatter(fmt=formatter)
            logger.addHandler(file_handler)
    return logger

def terminal_progress(
    current_bar:int,
    total_bar:int,
    prefix:str='',
    suffix:str='',
    bar_length:int=50
) -> None:
    # pylint: disable=expression-not-assigned
    """
    Calls in a loop to create a terminal progress bar.

    Args:
        current_bar (int): Current iteration.
        total_bar (int): Total iteration.
        prefix (str, optional): Prefix string. Defaults to ''.
        suffix (str, optional): Suffix string. Defaults to ''.
        bar_length (int, optional): Character length of the bar.
            Defaults to 50.

    References:
        https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
    """
    # Calculate the percent completed.
    percents = current_bar / float(total_bar)
    # Calculate the length of bar.
    filled_length = int(round(bar_length * current_bar / float(total_bar)))
    # Fill the bar.
    block = '█' * filled_length + '-' * (bar_length - filled_length)
    # Print new line.
    sys.stdout.write(f"\r{prefix} |{block}| {percents:.2%} {suffix}")

    if current_bar == total_bar:
        sys.stdout.write('\n')
    sys.stdout.flush()
