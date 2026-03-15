"""
visualizations.py
=================
Advanced stock-market style charts using Plotly.
Returns Plotly figure objects for use in the Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DARK_TEMPLATE = 'plotly_dark'


def price_trend_with_ma(dates, prices, title='Price Trend with Moving Averages'):
    """
    Line chart with 7-day and 30-day moving averages.
    Stock-market style with dark theme.
    """
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'price': prices})
    df = df.sort_values('date')
    df['ma_7'] = df['price'].rolling(window=7, min_periods=1).mean()
    df['ma_30'] = df['price'].rolling(window=30, min_periods=1).mean()

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['price'],
        mode='lines', name='Price',
        line=dict(color='#818cf8', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(129, 140, 248, 0.05)',
    ))

    # 7-day MA
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['ma_7'],
        mode='lines', name='7-Day MA',
        line=dict(color='#fbbf24', width=2, dash='dot'),
    ))

    # 30-day MA
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['ma_30'],
        mode='lines', name='30-Day MA',
        line=dict(color='#34d399', width=2),
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template=DARK_TEMPLATE,
        height=500,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def bollinger_bands_chart(dates, prices, window=20, num_std=2,
                          title='Bollinger Bands'):
    """
    Bollinger Bands chart: price with upper/lower bands based on
    rolling mean ± num_std * rolling std.
    """
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'price': prices})
    df = df.sort_values('date')
    df['sma'] = df['price'].rolling(window=window, min_periods=1).mean()
    df['std'] = df['price'].rolling(window=window, min_periods=1).std().fillna(0)
    df['upper'] = df['sma'] + num_std * df['std']
    df['lower'] = df['sma'] - num_std * df['std']

    fig = go.Figure()

    # Upper band
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['upper'],
        mode='lines', name='Upper Band',
        line=dict(color='rgba(248, 113, 113, 0.4)', width=1),
    ))

    # Lower band (fill to upper)
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['lower'],
        mode='lines', name='Lower Band',
        line=dict(color='rgba(56, 189, 248, 0.4)', width=1),
        fill='tonexty',
        fillcolor='rgba(129, 140, 248, 0.08)',
    ))

    # SMA
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['sma'],
        mode='lines', name=f'{window}-Day SMA',
        line=dict(color='#fbbf24', width=2),
    ))

    # Price
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['price'],
        mode='lines', name='Price',
        line=dict(color='#818cf8', width=1.5),
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template=DARK_TEMPLATE,
        height=500,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def price_volatility_chart(dates, prices, window=14,
                           title='Price Volatility (Rolling Std Dev)'):
    """
    Two-panel chart: price on top, rolling standard deviation on bottom.
    """
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'price': prices})
    df = df.sort_values('date')
    df['volatility'] = df['price'].rolling(window=window, min_periods=1).std().fillna(0)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=['Price', f'Volatility ({window}-Day Rolling Std)'],
    )

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['price'],
        mode='lines', name='Price',
        line=dict(color='#818cf8', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(129, 140, 248, 0.05)',
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df['date'], y=df['volatility'],
        name='Volatility',
        marker_color='rgba(248, 113, 113, 0.6)',
    ), row=2, col=1)

    fig.update_layout(
        title=title,
        template=DARK_TEMPLATE,
        height=600,
        hovermode='x unified',
        showlegend=False,
    )
    fig.update_yaxes(title_text='Price (₹)', row=1, col=1)
    fig.update_yaxes(title_text='Volatility', row=2, col=1)
    return fig


def seasonal_trends_chart(dates, prices, title='Seasonal Price Patterns'):
    """
    Box plot showing price distribution by month.
    """
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'price': prices})
    df['month'] = df['date'].dt.strftime('%b')
    df['month_num'] = df['date'].dt.month

    # Order by month
    df = df.sort_values('month_num')

    fig = px.box(
        df, x='month', y='price',
        title=title,
        labels={'month': 'Month', 'price': 'Price (₹)'},
        template=DARK_TEMPLATE,
        color='month',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=450, showlegend=False)
    return fig


def forecast_chart(historical_dates, historical_prices,
                   forecast_dates, forecast_prices,
                   lower_bound=None, upper_bound=None,
                   model_name='', title='Price Forecast'):
    """
    Combined historical + forecast chart with confidence bands.
    """
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=historical_dates, y=historical_prices,
        mode='lines', name='Historical',
        line=dict(color='#818cf8', width=2),
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_prices,
        mode='lines', name=f'Forecast ({model_name})',
        line=dict(color='#34d399', width=2, dash='dash'),
    ))

    # Confidence bands
    if upper_bound is not None:
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=upper_bound,
            mode='lines', name='Upper Bound',
            line=dict(color='rgba(251, 191, 36, 0.3)', width=1),
            showlegend=False,
        ))
    if lower_bound is not None:
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=lower_bound,
            mode='lines', name='Lower Bound',
            line=dict(color='rgba(56, 189, 248, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(129, 140, 248, 0.08)',
            showlegend=False,
        ))

    # Divider line at forecast start
    if len(forecast_dates) > 0:
        x_val = str(forecast_dates[0])[:10]
        fig.add_shape(
            type='line', x0=x_val, x1=x_val, y0=0, y1=1,
            yref='paper', line=dict(color='rgba(255,255,255,0.3)', dash='dot'),
        )
        fig.add_annotation(
            x=x_val, y=1.05, yref='paper',
            text='Forecast Start',
            showarrow=False,
            font=dict(color='rgba(255,255,255,0.5)', size=10),
        )

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template=DARK_TEMPLATE,
        height=500,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def festival_impact_chart(festivals_data, title='Festival Price Impact'):
    """
    Bar chart showing expected price during each festival.
    festivals_data: list of dicts from FestivalAnalyzer.estimate_festival_price()
    """
    if not festivals_data:
        fig = go.Figure()
        fig.add_annotation(text='No festival data available',
                           xref='paper', yref='paper', x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16, color='gray'))
        fig.update_layout(template=DARK_TEMPLATE, height=300)
        return fig

    names = [d['festival'] for d in festivals_data]
    current = [d['current_price'] for d in festivals_data]
    expected = [d['expected_price'] for d in festivals_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Current Price',
        x=names, y=current,
        marker_color='#818cf8',
    ))
    fig.add_trace(go.Bar(
        name='Expected Festival Price',
        x=names, y=expected,
        marker_color='#34d399',
    ))

    fig.update_layout(
        title=title,
        barmode='group',
        template=DARK_TEMPLATE,
        height=450,
        yaxis_title='Price (₹)',
    )
    return fig


# Import plotly.express for the seasonal chart
import plotly.express as px
