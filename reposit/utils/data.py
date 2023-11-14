"""
Query data function
"""
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
import database as db
from utils.file import project_folder



def metadata_fund(index_col='ticker', **kwargs):
    return pd.read_csv(
        f'{project_folder}/database/funds.csv',
        index_col=index_col,
        **kwargs
    )


def price_fund():
    return pd.read_csv(
        f'{project_folder}/database/price_funds.csv',
        index_col=['date'],
        parse_dates=['date'],
    )

def price_factor():

    return pd.read_csv(
        f'{project_folder}/database/price_factor.csv',
        index_col=['date'],
        parse_dates=['date'],
    )

def download(*args, **kwargs):
    return yf.download(*args, **kwargs)['Adj Close']


def oecd_lei_us():
    ticker = 'USALOLITONOSTSAM'
    data = pdr.DataReader(ticker, "fred", start="1900-01-01")[ticker]
    data.name = 'OECD_LEI_US'
    return data


def conference_lei_us():
    ticker = 'LEI_TOTL'
    data = db.price(ticker)[ticker]
    data.name = 'Conf_LEI_US'
    return data


def factor_index():

    tickers = {
        'LGY7TRUH' : 'rate',
        'MXCXDMHR' : 'equity',
        'LUACTRUU' : 'uscredit',
        'LP05TRUH' : 'eucredit',
        'LF98TRUU' : 'usjunk',
        'LP01TRUH' : 'eujunk',
        'BCOMTR' : 'commodity',
        'SPXT' : 'localequity',
        'PUT' : 'shortvol',
        'BCIT5T' : 'localinflation',
        'DXY' : 'currency',
        'M1EF' : 'emergingequity',
        'EMUSTRUU' : 'emergingbond',
        'M1WD' : 'developedequity',
        'LEGATRUU' : 'developedbond',
        'M1WD000$' : 'momentum',
        'M1WD000V' : 'value',
        'M1WD000G' : 'growth',
        'M1WDMVOL' : 'lowvol',
        'M1WDSC' : 'smallcap',
    }

    data = db.sql.time_series(list(tickers.keys()))
    data = data.rename(columns=tickers)
    return data.loc['2000-12-31':]



