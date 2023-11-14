""" base strategy class """
import os
import sys
import logging
import warnings
from dateutil import parser
from typing import Any, List, Optional, Dict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections.abc import Iterable
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")))
from config import get_args
from hive import db

logger = logging.getLogger("sqlite")

args = get_args()
TODAY = parser.parse(args.date)
YESTERDAY = TODAY - timedelta(days=1)


def fwd_monthly_pri_return(price: pd.DataFrame) -> pd.DataFrame:
    """ this is a pass through function"""
    return to_pri_return(price, forward=True, resample_by='M')

def to_pri_return(
    price: pd.DataFrame,
    periods: int or Iterable = 1,
    freq : pd.DateOffset or None = None,
    forward: bool = False,
    resample_by: str = None,
    binary: bool = False,
) -> pd.DataFrame:
    """
    Calculate price return data.

    Args:
        price (pd.DataFrame): price data.
        periods (int/Iterable): number(s) of periods.
        freq (pd.DateOffset): offset frequency.
        forward (bool, optional): if calculate forward. Defaults to False.
        resample_by (str, optional): resample period of data. Defaults to None.
        binary (bool, optional): if return only binary. Defaults to None.

    Returns:
        pd.DataFrame: price return data
    """
    if isinstance(periods, Iterable):

        result = list()

        for period in periods:

            pri_return = to_pri_return(
                price=price, periods=period, freq=freq,
                resample_by=resample_by, forward=forward, binary=binary,
            )

            if isinstance(pri_return, pd.Series):
                pri_return.name = period
                result.append(pri_return)

            elif isinstance(pri_return, pd.DataFrame):

                pri_return = pd.concat(
                    objs = [pri_return],
                    keys=[period], axis=1
                ).stack()

                pri_return.index.names = ['date', 'ticker']

                result.append(pri_return)

        return pd.concat(result, axis=1)


    if isinstance(price, (pd.Series, pd.DataFrame)):

        if resample_by is not None:
            price = price.resample(rule=resample_by).last()

        price_shift = price.shift(
            periods=periods, freq=freq
        ).dropna(how='all').resample('D').last().ffill().filter(
            items=price.index, axis=0
        )

        if forward:
            price_shift = price.shift(
                periods=-periods, freq=freq
            ).dropna(how='all').resample('D').last().ffill().filter(
                items=price.index, axis=0
            )

            result = price_shift / price - 1
        else:
            price_shift = price.shift(
                periods=periods, freq=freq
            ).dropna(how='all').resample('D').last().ffill().filter(
                items=price.index, axis=0
            )

            result = price / price_shift - 1

        if binary: return result.apply(np.sign)

        return result

    raise TypeError('not supported type.')


class BaseRegime:
    """
    Base class for market regime indicator.

    future:
        make a report page.
    """

    def __init__(self, data) -> None:

        self.data = data


    @property
    def label(self) -> pd.Series:
        """
        make daily regime label.

        Returns:
            pd.Series: daily regime label.
        """
        try:
            start = self._label.index[0]
            date_range = pd.date_range(start, end=datetime.date.today())
            return self._label.reindex(date_range).ffill()
        except AttributeError as exception:
            raise AttributeError("regime class has not fitted for label. please run `Regime.fit()` first.") from exception

    @label.setter
    def label(self, label:pd.Series):
        self._label = label
        self._label.name = 'label'

    @property
    def regime(self):
        """
        Store your regime defination here.
        as well as any other information attached to each regime state.

        Returns:
            dict: regime definations.
        """
        return {"regime_state" : "please add other information here.:"}

    def add_label(self, arg:..., ) -> ...:
        """
        helper function to add regime state label onto passed argument.

        Args:
            arg (Any): argument you want to label.

        Returns:
            ...: argument with regime label.
        """
        if isinstance(arg, pd.DataFrame):
            label = self.label
            start = label.index[0]
            arg = arg.loc[start:]
            return arg.assign(label=label)

        if isinstance(arg, pd.Series):
            if arg.name is None:
                arg.name = "original"
            return self.add_label(arg=arg.to_frame())

        raise TypeError("Regime.add_label method supports pd.Series, pd.DataFrame only.")

    def fwd_monthly_pri_return(self, price: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate forward 1M return in each regime.

        Args:
            price (pd.DataFrame): price data.

        Returns:
            pd.DataFrame: forward return for each regime.
        """
        return self.add_label(fwd_monthly_pri_return(price)).groupby(by=['label']).mean()

    def expected_returns(self, price:pd.DataFrame, resample_rule:...='M'):
        resampled_price = price.resample(rule=resample_rule).last()
        resampled_forward_return = resampled_price.pct_change(1).shift(-1)
        labeled_resampled_forward_return = self.add_label(resampled_forward_return).groupby(by=['label']).mean()
        return labeled_resampled_forward_return

    def excess_returns(self, price:pd.DataFrame, resample_rule:...='M'):
        # Prepare resampled data set.
        resampled_price = price.resample(rule=resample_rule).last().dropna()
        resampled_forward_return = resampled_price.pct_change(1).shift(-1)
        # Analyze regime performance
        labeled_resampled_forward_return = self.add_label(resampled_forward_return).groupby(by=['label'])
        regime_mean_forward_return = labeled_resampled_forward_return.mean()
        regime_relative_forward_return = regime_mean_forward_return.subtract(resampled_forward_return.mean())
        return regime_relative_forward_return

    def analyze(self, price:pd.DataFrame, resample_rule:...='M', weight_tilt:float=0.20) -> None:
        # Prepare resampled data set.
        resampled_price = price.resample(rule=resample_rule).last().dropna()
        resampled_drawdown = resampled_price.divide(resampled_price.expanding().max()).subtract(1)
        resampled_forward_return = resampled_price.pct_change(1).shift(-1)
        # Some parameters
        num_asset = len(resampled_price.columns)
        # Analyze regime performance
        labeled_resampled_forward_return = self.add_label(resampled_forward_return).groupby(by=['label'])
        regime_mean_forward_return = labeled_resampled_forward_return.mean()
        regime_relative_forward_return = regime_mean_forward_return.subtract(resampled_forward_return.mean())
        print(f'regime relative forward return:\n{regime_relative_forward_return.to_string()}')

        best_assets = regime_relative_forward_return.idxmax(axis=1)

        # Make Allocation
        neutral_allocation = pd.Series({asset : 1 / num_asset for asset in resampled_price.columns})

        regime_allocation = {}

        for state, ticker in best_assets.items():
            allo = neutral_allocation * (1 - weight_tilt)
            allo.loc[ticker] += weight_tilt
            regime_allocation[state] = allo

        end_month_weight = pd.DataFrame(self.label.map(regime_allocation).to_dict()).T.resample('M').last()
        price_allocation = end_month_weight.multiply(resampled_forward_return).dropna().sum(axis=1).add(1).cumprod()
        price_neutral = neutral_allocation.multiply(resampled_forward_return).sum(axis=1).add(1).cumprod()
        end_month_weight = end_month_weight.loc[price_allocation.index[0]:]

        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
        except ImportError as exception:
            raise ImportError(
                "please install plotly packages before running analyze."
                ) from exception

        fig = make_subplots(
            rows=4, cols=2,
            specs=[
                [{'secondary_y':True, 'colspan':2}, None],
                [{'colspan':2}, None],
                [{'secondary_y':True, 'colspan':2}, None],
                [{}, {}],
                ],
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            subplot_titles='x'*100,
            shared_xaxes=True,
        )
        annotations = fig['layout']['annotations']

        asset_color = {asset: px.colors.qualitative.Alphabet[idx] for idx, asset in enumerate(price)}
        state_color = {state: px.colors.qualitative.Bold[idx] for idx, state in enumerate(self.regime.keys())}

        annotations[0].text = 'Allocation Result'
        annotations[1].text = 'Allocation Weight'
        annotations[2].text = 'Regime Map'
        annotations[3].text = 'Forward Mean Return (%)'
        annotations[4].text = 'Forward Return Boxplot (%)'

        for state in self.regime.keys():
            if state == 'neutral': continue
            label=self.label.loc[resampled_price.index[0]:]
            trace = go.Scatter(
                x=label.index,
                y=(label==state).astype(int)*100,
                mode='none',
                name=state.upper(),
                stackgroup='one',
                line_color = state_color[state],
                opacity=0.,
                )
            fig.add_trace(trace, row=3, col=1, secondary_y=True)

        trace = go.Scatter(
            x=price_allocation.index,
            y=price_allocation.round(2).values,
            name = price_allocation.name,
            showlegend=False,
            marker_color='#AA0DEF',
        )
        fig.add_trace(trace, row=1, col=1)

        trace = go.Scatter(
            x=price_neutral.index,
            y=price_neutral.round(2).values,
            name = price_neutral.name,
            showlegend=False,
            marker_color='#E15F99',
        )
        fig.add_trace(trace, row=1, col=1)

        alpha = price_allocation.pct_change().subtract(price_neutral.pct_change()).add(1).cumprod().subtract(1)

        trace = go.Scatter(
            x=alpha.index,
            y=alpha.values,
            name = 'alpha',
            # mode = 'lines',
            fill = 'tozeroy',
            showlegend=False,
            marker_color='#E15F99',
        )
        fig.add_trace(trace, row=1, col=1, secondary_y=True)

        for asset in resampled_forward_return:
            trace = go.Scatter(
                x = end_month_weight.index,
                y = end_month_weight[asset].round(4) * 100,
                name=asset,
                mode='lines',
                legendgroup=asset,
                showlegend=False,
                marker_color=asset_color[asset],
                line_color = asset_color[asset],
                stackgroup='one'
            )
            fig.add_trace(trace, row=2, col=1)
            # price drawdown trace.
            trace = go.Scatter(
                x=resampled_drawdown.index,
                y=resampled_drawdown[asset].round(4) * 100,
                name=asset,
                legendgroup=asset,
                showlegend=True,
                marker_color=asset_color[asset],
            )
            fig.add_trace(trace, row=3, col=1)

            data = self.add_label(resampled_forward_return[asset])
            grouped_data = data.groupby(by=['label'])
            regime_mean_forward_return = grouped_data.mean()
            trace=go.Bar(
                x=regime_mean_forward_return.index,
                y=regime_mean_forward_return[asset].round(4) * 100,
                name=asset,
                legendgroup=asset,
                showlegend=False,
                marker_color=asset_color[asset],
            )
            fig.add_trace(trace, row=4, col=1)

            trace = go.Box(
                x=data.label,
                y=data[asset].round(4) * 100,
                name=asset,
                boxpoints='outliers',
                legendgroup=asset,
                showlegend=False,
                line_color=asset_color[asset],
            )
            fig.add_trace(trace, row=4, col=2)

        fig.update_layout(boxmode='group',
                        hovermode='x unified',
                        height=800,
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False),
                        )

        fig.update_xaxes(categoryorder='array',
                        categoryarray=list(self.regime.keys()))

        return fig
    

def run_abl_regime_allocation(regime: str, today: datetime.date = date.today()) -> None:
    """
    run mlp allocation at the month start trading date
    i.e. first trading day each month.
    
    Args:
        regime (str): the regime module name.
        today (datetime.date, optional): the exact date of runnging this function. Defaults to date.today().
    """    
    extra = dict(user=args.user, activity="abl_allocation", category="script")

    if today != db.get_start_trading_date(market="KR", asofdate=today):
        logger.info(msg=f"[SKIP] {regime.upper()} ABL allocation. {today:%Y-%m-%d}", extra=extra)
        return

    logger.info(msg=f"[PASS] Start {regime.upper()} ABL allocation. {today:%Y-%m-%d}", extra=extra)
    
    ticker_mapper = db.get_meta_mapper()
    OUTPUT_COLS = ["isin", "ticker_bloomberg", "asset_class", "risk_score", "name"]

    universe = db.load_universe("abl_us")
    price_asset = db.get_price(tickers=", ".join(list(universe.index))).loc[:YESTERDAY]
    price_factor = db.get_lens(today)

    from core.strategy.ablstrategy import AblStrategy
    
    strategy = AblStrategy.load(
        universe=universe,
        price_asset=price_asset,
        price_factor=price_factor,
        regime=regime,
        asofdate=today,
        level=5,
    )

    us_weights = strategy.allocate()

    us_risk_score = 0.0
    for asset, weight in us_weights.items():
        us_risk_score += weight * strategy.universe.loc[str(asset)].risk_score
    
    msg = f"\n[PASS] {regime.upper()} ABL MP"
    msg += f"\n{today} | US level 5 {regime.upper()}"
    msg += f"\nrisk score {us_risk_score:.4f}\n"
    msg += us_weights.to_markdown()
    logger.info(msg, extra=extra)

    if args.database == "true":
        
        us_uni = strategy.universe[OUTPUT_COLS]
        us_uni[f"{today:%Y-%m-%d} weight"] = us_uni.index.map(us_weights.to_dict())
        us_uni = us_uni.dropna()

        us_weights = us_weights.to_frame().reset_index()
        us_weights.columns = ["ticker", "weights"]
        us_weights["rebal_dt"] = today
        portfolio_id = db.get_portfolio_id(portfolio=f"abl_us_5_{regime}")
        us_weights["port_id"] = portfolio_id
        us_weights["stk_id"] = us_weights.ticker.map(ticker_mapper)
        
        try:
            db.TbPortAlloc.insert(us_weights)
        except:
            try:
                db.TbPortAlloc.update(us_weights)
            except:
                db_alloc = db.get_alloc_weight_for_shares(strategy="ABL", market="US", level=f"5_{regime.upper()}")
                db_alloc = db_alloc[db_alloc.rebal_dt == today]

                merge_df = us_weights.merge(db_alloc, on=["rebal_dt", "port_id", "stk_id"], how="outer")
                delete_asset = merge_df[merge_df.weights_x.isnull()].stk_id.tolist()
                update_asset = merge_df.dropna()
                update_asset['weights'] = update_asset['weights_x']
                insert_asset = merge_df[merge_df.weights_y.isnull()]
                insert_asset['weights'] = insert_asset['weights_x']

                db.delete_asset_port_alloc(rebal_dt=today, port_id=portfolio_id, stk_id=delete_asset)
                try:
                    db.TbPortAlloc.insert(insert_asset)
                except:
                    db.TbPortAlloc.update(update_asset)
                
                    
    logger.info(msg=f"[PASS] End {regime.upper()} ABL allocation. {today:%Y-%m-%d}", extra=extra)