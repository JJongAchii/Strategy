import pandas as pd
import pandas_datareader as pdr


def calculate_sample_benchmarks():
    RiskAllocation = {
        "5": {"069500": 0.70, "114260": 0.25, "153130": 0.05},
        "4": {"069500": 0.60, "114260": 0.35, "153130": 0.05},
        "3": {"069500": 0.50, "114260": 0.45, "153130": 0.05},
        "2": {"069500": 0.40, "114260": 0.50, "153130": 0.10},
        "1": {"069500": 0.30, "114260": 0.60, "153130": 0.10},
    }

    result = []

    for key, value in RiskAllocation.items():

        ds = []
        for k, v in value.items():
            d = pdr.DataReader(k, data_source="naver", start='1990-1-1')['Close'].astype(float)
            d.name = k
            ds.append(d)

        pri_return = pd.concat(ds, axis=1).dropna().pct_change().fillna(0)

        bm = pri_return.multiply(pd.Series(value)).sum(axis=1).add(1).cumprod()

        bm.name = f"PGA-BM-{key}"

        result.append(bm)

    result = pd.concat(result, axis=1)

    result = result.resample('D').last().ffill().stack().reset_index()

    result.columns = ['Date', 'Benchmark', 'Value']
    result["Market"] = "KR"
    return result


bm = calculate_sample_benchmarks()
bm.to_clipboard()