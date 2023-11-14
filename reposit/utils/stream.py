"""
Streamlit Function
"""


import streamlit as st
from utils import data, fin, plt
from regime import USLEIHP


@st.cache()
def get_fund_price():
    price = data.price_fund().loc['2009':]
    price = price.dropna(thresh=252*5, axis=1)
    return price

@st.cache()
def get_fund_data():
    return data.metadata_fund()

@st.cache()
def get_factor_price():
    price = data.price_factor().loc['2009':]
    return price


@st.cache
def manager_behavior(
    price, price_factor, static_window, active_window, model
):
    return fin.manager_behavior(
        price=price,
        price_factor=price_factor,
        static_window=static_window,
        active_window=active_window,
        model=model,
    )


def main():
    price_fund = get_fund_price()
    price_factor = get_factor_price()
    meta_fund = get_fund_data()

    def display_fund(ticker):
        fund = meta_fund.loc[ticker]
        return f"[{fund.equity_style_box}]{fund.loc['name']}"


    regime = USLEIHP().fit()

    with st.form("Fund Evaluation"):


        c1, c2, c3 = st.columns([2, 1, 1])

        funds = c1.multiselect(
            label='Select funds for evaluation',
            options=meta_fund.index,
            format_func=display_fund,
        )

        static_window = c2.number_input(
            label="Static Window",
            min_value=252,
            max_value=252*5,
            value=252,
            step=252,
            format=None,
            key=None,
            help=None,
        )
        active_window = c3.number_input(
            label="Dynamic Window",
            min_value=63,
            max_value=252*3,
            value=63,
            step=63,
            format=None,
            key=None,
            help=None,
        )

        submitted = st.form_submit_button("Submit")

        if submitted:

            cols = st.columns([1] * len(funds))

            for idx, fund in enumerate(funds,0):

                price = price_fund[fund].dropna()

                cols[idx].header(f"Manager Skill Contribution: {fund}")

                result = manager_behavior(
                    price=price, price_factor=price_factor,
                    static_window=static_window,
                    active_window=active_window,
                    model='linear',
                )

                fig = plt.line(
                    df=result,
                    yaxis_tickformat='.2f',
                )
                cols[idx].plotly_chart(
                    figure_or_data=fig,
                    use_container_width=True,
                )

                cols[idx].header("Dynamic Portfolio Factor Tilt.")

                static_beta = fin.factor_exposure(
                    price=price.iloc[-static_window:],
                    price_factor=price_factor,
                    model='lasso',
                )

                active_beta = fin.factor_exposure(
                    price=price.iloc[-active_window:],
                    price_factor=price_factor,
                    model='lasso',
                )


                beta_diff = active_beta - static_beta

                fig = plt.bar(beta_diff, yaxis_tickformat='.2%')
                cols[idx].plotly_chart(fig, use_container_width=True)

                exp = regime.fwd_monthly_pri_return(price)

                cols[idx].header("Regime Performance")

                fig = plt.bar(exp, yaxis_tickformat='.2%')
                cols[idx].plotly_chart(fig, use_container_width=True)

            # fig = regime.analyze(price_fund[funds])
            # st.plotly_chart(fig, use_container_width=True)


