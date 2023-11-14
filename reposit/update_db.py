from hive_old import db
import pandas as pd

value1_df = pd.read_excel("mlp_data_port/KR_3.xlsx", sheet_name="value")
value2_df = pd.read_excel("mlp_data_port/KR_4.xlsx", sheet_name="value")
value3_df = pd.read_excel("mlp_data_port/KR_5.xlsx", sheet_name="value")
value4_df = pd.read_excel("mlp_data_port/US_3.xlsx", sheet_name="value")
value5_df = pd.read_excel("mlp_data_port/US_4.xlsx", sheet_name="value")
value6_df = pd.read_excel("mlp_data_port/US_5.xlsx", sheet_name="value")

value1_df['portfolio_id'] = 1
value2_df['portfolio_id'] = 2
value3_df['portfolio_id'] = 3
value4_df['portfolio_id'] = 4
value5_df['portfolio_id'] = 5
value6_df['portfolio_id'] = 6

data = pd.concat([value1_df, value2_df, value3_df, value4_df, value5_df, value6_df])
# data = data.set_index(['date', 'portfolio_id']).stack().reset_index()
# data = data[data != 0].dropna()
# data.columns = ['date', 'portfolio_id', 'meta_id', 'weights']

with db.session_local() as session:

    db.TbPortfolioValue.delete()
    db.TbPortfolioValue.insert(data)

    # query = (
    #     session.query(
    #         db.TbMeta.id,
    #         db.TbMeta.ticker
    #     )
    # )
    # meta = db.read_sql_query(query)
    # meta_id = meta.set_index('ticker')['id'].to_dict()
    # data['meta_id'] = data['meta_id'].map(meta_id)
    # print(data)
    #
    # db.TbPortfolioBook.delete()
    # db.TbPortfolioBook.insert(data)
