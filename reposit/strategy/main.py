import os
import sys
import datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px


sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from reposit.strategy.strategy_jjong import *
from jjongdb import db


def get_universe():
    with db.session_local() as session:
        query = (
            session.query(
                db.TbMeta.ticker,
                db.TbMeta.isin,
                db.TbMeta.name,
                db.TbMeta.asset_class,
                db.TbMeta.sector,
                db.TbMeta.iso_code,
                db.TbMeta.marketcap,
                db.TbMeta.fee
            )
            .order_by(db.TbMeta.meta_id)
        )
        return db.read_sql_query(query=query)


universe_list = get_universe()
algo_list = {
    "equal weights": "eq",
    "customize weights": "custom",
    "target volatility": "target_vol",
    "absolute momentum": "abs_mmt",
    "dual momentum": "dual_mmt",
    "dual momentum2": "dual_mmt2",
    "weighted momentum": "weighted_mmt",
    "Meb Faber Momentum": "meb_mmt"
}
freq = {
    "monthly" : "M",
    "yearly" : "Y"
}
custom_weights = None
if "nav" not in st.session_state:
    st.session_state.nav = pd.DataFrame()
if "weights_dict" not in st.session_state:
    st.session_state.weights_dict = {}
if "bt_instance" not in st.session_state:
    st.session_state.bt_instance = ""



st.title("Backtesting Tool")

with st.sidebar:

    st.title("Make your Strategy")
    
    with st.form("strategy settings"):
        
        strategy_name = st.text_input("strategy name")
        
        tickers = st.multiselect("Set universe", universe_list.ticker)
        rebal_freq = st.selectbox("Set rebalance frequency", freq.keys())
        placeholder_for_method = st.empty()
        placeholder_for_optional_by_method = st.empty()
        
        start_date = st.date_input("start date", datetime.date(2000, 1, 1))
        end_date = st.date_input("end date")
        
        submit = st.form_submit_button("Run Backtest")
            

with placeholder_for_method:
    selected_method = st.selectbox("Choose Algorithm", options=algo_list.keys())
    method = algo_list[selected_method]
    
with placeholder_for_optional_by_method:
    if method == "custom":
        editor_dict = {}
        for ticker in tickers:
            editor_dict[ticker] = 1 / len(tickers)

        custom_weights = st.data_editor(editor_dict)
        

with st.expander("Show all ETF list"):
    st.dataframe(universe_list, hide_index=True)
        
    
tab1, tab2, tab3 = st.tabs(["Backtest Result", "Asset Class Allocation", "Asset Correlation"])


if submit:
    st.session_state.bt_instance = Backtest(strategy_name=strategy_name)
    universe = st.session_state.bt_instance.universe(tickers=tickers)
    price = st.session_state.bt_instance.data(tickers=universe)
    
    weights = st.session_state.bt_instance.rebalance(
        price=price, 
        method=method,
        freq=freq[rebal_freq],
        custom_weight=custom_weights, 
        start=start_date,
        end=end_date
    )
    
    st.session_state.weights_dict[strategy_name] = weights
    
    st.session_state.nav = st.session_state.bt_instance.result(price=price, weight=weights)
    

with tab1:
    if not st.session_state.nav.empty:
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("start date", st.session_state.nav.index[0])
        with col2:
            end_date = st.date_input("end date")
        
        result = st.session_state.bt_instance.report(nav=st.session_state.nav, start=start_date, end=end_date)
        nav_slice = st.session_state.nav.loc[start_date:end_date]
        nav_slice = nav_slice.pct_change().add(1).cumprod() * 1000
        
        ### Performance of backtesting ###
        st.header("Performance of backtesting")
        fig = px.line(nav_slice, x=nav_slice.index, y=nav_slice.columns)
        st.plotly_chart(fig, theme=None)

        ### Metrics ###
        st.header("Metrics")
        st.write(result.T)
        
        ### Monthly returns ###
        st.header("Monthly returns")
        monthly_ret = resample_data(price=nav_slice, freq="M", type="tail").pct_change()
        fig = px.bar(monthly_ret, barmode="group")
        st.plotly_chart(fig, theme=None)
        
        ### Yearly returns ###
        st.header("Yearly returns")
        yearly_ret = resample_data(price=nav_slice, freq="Y", type="tail").pct_change()
        
        fig = px.bar(yearly_ret, barmode="group")
        st.plotly_chart(fig, theme=None)
        
        ### 1 year rolling charts ###
        st.header("1 year rolling charts")
        rolling_1_year_returns = nav_slice.pct_change(periods=252)
        fig = px.line(rolling_1_year_returns, x=rolling_1_year_returns.index, y=rolling_1_year_returns.columns)
        st.plotly_chart(fig, theme=None)
        
        ### 3 year rolling charts ###
        st.header("3 year rolling charts")
        rolling_3_year_returns = nav_slice.pct_change(periods=252*3)
        fig = px.line(rolling_3_year_returns, x=rolling_3_year_returns.index, y=rolling_3_year_returns.columns)
        st.plotly_chart(fig, theme=None)
        
        

with tab2:
    
    if st.session_state.weights_dict:
        strategy = st.selectbox("Choose Strategy", st.session_state.weights_dict.keys())
        weights = st.session_state.weights_dict[strategy]
        
        seleted_date = st.selectbox("Select Date", weights.index.strftime('%Y-%m-%d'))
        seleted_weights = weights[weights.index == seleted_date]
        pivot = seleted_weights.stack().reset_index()
        pivot.columns = ["Date", "ticker", "weights"]
        
        fig = px.pie(pivot, values=pivot.weights, names=pivot.ticker)
        st.plotly_chart(fig)
        
        st.write(weights)
        
        
with tab3:
    tickers = st.multiselect("Compare assets correlation", universe_list.ticker)
    y_ticker = st.multiselect("Select Dependent Variable", universe_list.ticker)
    
    if len(tickers) > 1 and len(y_ticker) > 1:
        all_tickers = tickers + y_ticker
        
        bt = Backtest()
        price = bt.data(tickers=all_tickers)
        corr = price.corr()
        
        fig = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig)
        
        betas = regression(independent_x=price[y_ticker], dependent_y=price[tickers])
        st.write(betas)