

import pandas as pd
import plotly.express as px

layout = dict(
            title = None,
            xaxis_title = None,
            xaxis_tickformat = None,
            xaxis_showgrid = False,
            yaxis_title = None,
            yaxis_tickformat = None,
            yaxis_showgrid = False,
            showlegend=True,
            legend_title = None,
            legend_yanchor = None,
            legend_xanchor = None,
            legend_x = None,
            legend_y = None,
            legend_font_size=10,
            font_family = "Helvetica monospace",
            font_size = 12,
            margin={'l':10, 'r':10, 't':10, 'b':10},
            height=230,
            hovermode='x',
            )

def line(df, **kwargs):
    fig = px.line(df, color_discrete_sequence=px.colors.qualitative.Set1)
    lo = layout.copy()
    lo.update(kwargs)
    return fig.update_layout(lo)


def bar(df, **kwargs):
    fig = px.bar(df, color_discrete_sequence=px.colors.qualitative.Set1)
    lo = layout.copy()
    lo.update(kwargs)
    return fig.update_layout(lo)

def area(df, **kwargs):
    fig = px.area(df, color_discrete_sequence=px.colors.qualitative.Set1)
    lo = layout.copy()
    lo.update(kwargs)
    return fig.update_layout(lo)





