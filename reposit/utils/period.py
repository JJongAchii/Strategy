"""
Interesting Periods.
"""

from collections import OrderedDict
import pandas as pd

interesting_periods = OrderedDict()

# Dotcom bubble
interesting_periods['Dotcom'] = (
    pd.Timestamp('20000310'),
    pd.Timestamp('20000910')
)

# Lehmann Brothers
interesting_periods['Lehman'] = (
    pd.Timestamp('20080801'),
    pd.Timestamp('20081001')
)

# 9/11
interesting_periods['9/11'] = (
    pd.Timestamp('20010911'),
    pd.Timestamp('20011011')
)

# 05/08/11  US down grade and European Debt Crisis 2011
interesting_periods['US downgrade/European Debt Crisis'] = (
    pd.Timestamp('20110805'),
    pd.Timestamp('20110905')
)

# 16/03/11  Fukushima melt down 2011
interesting_periods['Fukushima'] = (
    pd.Timestamp('20110316'),
    pd.Timestamp('20110416')
)

# 01/08/03  US Housing Bubble 2003
interesting_periods['US Housing'] = (
    pd.Timestamp('20030108'),
    pd.Timestamp('20030208')
)

# 06/09/12  EZB IR Event 2012
interesting_periods['EZB IR Event'] = (
    pd.Timestamp('20120910'),
    pd.Timestamp('20121010')
)

# August 2007, March and September of 2008, Q1 & Q2 2009,
interesting_periods['Aug07'] = (
    pd.Timestamp('20070801'),
    pd.Timestamp('20070901')
)
interesting_periods['Mar08'] = (
    pd.Timestamp('20080301'),
    pd.Timestamp('20080401')
)
interesting_periods['Sept08'] = (
    pd.Timestamp('20080901'),
    pd.Timestamp('20081001')
)
interesting_periods['2009Q1'] = (
    pd.Timestamp('20090101'),
    pd.Timestamp('20090301')
)
interesting_periods['2009Q2'] = (
    pd.Timestamp('20090301'),
    pd.Timestamp('20090601')
)

# Flash Crash (May 6, 2010 + 1 week post),
interesting_periods['Flash Crash'] = (
    pd.Timestamp('20100505'),
    pd.Timestamp('20100510')
)

# April and October 2014).
interesting_periods['Apr14'] = (
    pd.Timestamp('20140401'),
    pd.Timestamp('20140501')
)
interesting_periods['Oct14'] = (
    pd.Timestamp('20141001'),
    pd.Timestamp('20141101')
)

# Market down-turn in August/Sept 2015
interesting_periods['Fall2015'] = (
    pd.Timestamp('20150815'),
    pd.Timestamp('20150930')
)

# Market regimes
interesting_periods['Low Volatility Bull Market'] = (
    pd.Timestamp('20050101'),
    pd.Timestamp('20070801')
)

interesting_periods['GFC Crash'] = (
    pd.Timestamp('20070801'),
    pd.Timestamp('20090401')
)

interesting_periods['Recovery'] = (
    pd.Timestamp('20090401'),
    pd.Timestamp('20130101')
)

# interesting_periods['New Normal'] = (
#     pd.Timestamp('20130101'),
#     pd.Timestamp('today')
# )

interesting_periods['Covid-19'] = (
    pd.Timestamp('20200306'),
    pd.Timestamp('20200409')
)