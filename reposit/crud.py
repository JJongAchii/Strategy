from hive_old import db
import pandas as pd


df = pd.DataFrame(columns=['date', 'portfolio_id', 'value'])
df = df.append([['2023-04-04', 3, 13.13]])
print(df)
