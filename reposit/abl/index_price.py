tickers = ["LGY7TRUH Index", "MXCXDMHR Index", "LUACTRUU Index", "LP05TRUH Index",
           "LF98TRUU Index", "LP01TRUH Index", "BCOMTR Index", "SPXT Index", "PUT Index",
           "BCIT5T Index", "DXY Curncy", "M1EF Index", "EMUSTRUU Index", "M1WD Index",
           "LEGATRUU Index", "M1WD000$ Index", "M1WD000V Index", "M1WD000G Index",
           "M1WDMVOL Index", "M1WDSC Index", "M1WDQU Index"]

from xbbg import blp

data = blp.bdh(tickers, "PX_LAST", "1997-1-1")

name = {
    "LGY7TRUH Index": "rate",
    "MXCXDMHR Index": "equity",
    "LUACTRUU Index": "uscredit",
    "LP05TRUH Index": "eucredit",
    "LF98TRUU Index": "usjunk",
    "LP01TRUH Index": "eujunk",
    "BCOMTR Index": "commodity",
    "SPXT Index": "localequity",
    "PUT Index": "shortvol",
    "BCIT5T Index": "localinflation",
    "DXY Curncy": "currency",
    "M1EF Index": "emergingequity",
    "EMUSTRUU Index": "emergingbond",
    "M1WD Index": "developedequity",
    "LEGATRUU Index": "developedbond",
    "M1WD000$ Index": "momentum",
    "M1WD000V Index": "value",
    "M1WD000G Index": "growth",
    "M1WDMVOL Index": "lowvol",
    "M1WDSC Index": "smallcap",
    "M1WDQU Index": "quality",
}

data = data.rename(columns=name)

data.to_csv("index_price.csv")