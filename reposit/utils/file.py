"""
File relatied utility functions.
"""
import os
import time
import pandas as pd
import json
from collections import OrderedDict

project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def abspath(path: str) -> str:
    """
    Absolute path

    Args:
        path (str): path string of file or directory.

    Returns:
        str: absolute path string.
    """
    return os.path.abspath(path)


def all_files(path: str, keyword: str = '', ext: str = '') -> list:
    """
    Search all files with criteria

    Args:
        path (str): path string.
        keyword (str, optional): keyword to search. Defaults to ''.
        ext (str, optional): file extensions. Defaults to ''.

    Returns:
        list: all file names with criteria fulfilled
    """

    return [
        abspath(os.path.join(path, p)) for p in os.listdir(path)
        if not p.startswith('~') and p.endswith(ext) and keyword in p
    ]


def create_folder(path: str) -> None:
    """
    Make folder as well as all parent folders if not exists

    Args:
        path: full path name
        is_file: whether input is name of file
    """
    os.makedirs(path)


def modified_time(path: str) -> pd.Timestamp:
    """
    get modified time of file.

    Args:
        path (str): file path

    Returns:
        pd.Timestamp: modified time of file.
    """
    return pd.to_datetime(time.ctime(os.path.getmtime(path)))


def sort_by_modified_time(path: str) -> list:
    """
    Sort files by modified time.

    Args:
        path (str): path string.

    Returns:
        list: list of files sorted by modified time
    """
    return sorted(all_files(path), key=modified_time, reverse=True)


def latest_file(path: str) -> str:
    """
    Latest modified file in folder

    Args:
        path (str): path string.

    Returns:
        str: latest file path.
    """
    return sort_by_modified_time(path=path)[0]


def exists(path: str) -> bool:
    """
    Check if path exits.

    Args:
        path (str): path string.

    Returns:
        bool: file exits
    """
    return os.path.exists(path)


def dataframe_to_json(df, json_name, idx):

    data_list = []
    for n, row in df.iterrows():
        data = OrderedDict()
        for i in idx:
            data[i] = row[i]
        data_list.append(data)

    with open(json_name, 'w', encoding='utf-8') as writeJsonfile:
        json.dump(data_list, writeJsonfile, indent=4, default=str, ensure_ascii=False)


def excel_to_json(excel_name, json_name, idx):

    df = pd.read_excel(excel_name, engine='openpyxl')

    data_list = []
    for n, row in df.iterrows():
        data = OrderedDict()
        for i in idx:
            data[i] = row[i]
        data_list.append(data)

    with open(json_name, 'w', encoding='utf-8') as writeJsonfile:
        json.dump(data_list, writeJsonfile, indent=4, default=str, ensure_ascii=False)


def create_allocation_file(mp, filename):

    json_name = os.path.join(project_folder, 'config', 'universe.json')
    mp.name += ' weight'

    data = pd.read_json(json_name)
    data.set_index('ticker', inplace=True)
    data = data.loc[mp.index]
    data = pd.concat([data, mp], axis=1)
    data.to_csv(filename, encoding='utf-8-sig')

    s_risk_score = data['risk_score'] * mp

    return s_risk_score.sum()


