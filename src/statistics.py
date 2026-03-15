"""
statistics.py
=============
Dataset statistics and visualization generator.
Returns Plotly figures for use in the Streamlit dashboard.
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Dark theme for all charts
DARK_TEMPLATE = 'plotly_dark'
CHART_COLORS = px.colors.qualitative.Set2


class DatasetAnalyzer:
    """Generate statistical insights and visualizations from the dataset."""

    def __init__(self, models_dir=None):
        models_dir = models_dir or MODELS_DIR
        self.products = None
        self.category_stats = {}
        self.category_trends = {}

        products_path = os.path.join(models_dir, 'dashboard_products.pkl')
        if os.path.exists(products_path):
            try:
                self.products = pd.read_pickle(products_path)
            except Exception:
                self.products = None

        stats_path = os.path.join(models_dir, 'dashboard_category_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                self.category_stats = json.load(f)

        trends_path = os.path.join(models_dir, 'dashboard_category_trends.json')
        if os.path.exists(trends_path):
            with open(trends_path) as f:
                self.category_trends = json.load(f)

    def avg_price_by_category(self):
        """Horizontal bar chart: average discounted price by category."""
        if not self.category_stats:
            return _empty_fig('No category data available')

        cats = []
        prices = []
        for cat, stats in self.category_stats.items():
            cats.append(cat)
            prices.append(stats.get('avg_discounted', 0))

        df = pd.DataFrame({'Category': cats, 'Avg Price (₹)': prices})
        df = df.sort_values('Avg Price (₹)', ascending=True).tail(20)

        fig = px.bar(
            df, x='Avg Price (₹)', y='Category',
            orientation='h',
            title='Average Price by Category',
            template=DARK_TEMPLATE,
            color='Avg Price (₹)',
            color_continuous_scale='Viridis',
        )
        fig.update_layout(height=600, showlegend=False)
        return fig

    def median_price_chart(self):
        """Bar chart: median price by top categories."""
        if self.products is None:
            return _empty_fig('No product data available')

        median = self.products.groupby('main_category')['discounted_price_clean'].median()
        median = median.dropna().sort_values(ascending=False).head(15)

        fig = px.bar(
            x=median.index, y=median.values,
            title='Median Price by Category (Top 15)',
            labels={'x': 'Category', 'y': 'Median Price (₹)'},
            template=DARK_TEMPLATE,
            color=median.values,
            color_continuous_scale='Plasma',
        )
        fig.update_layout(height=500, showlegend=False)
        return fig

    def price_distribution(self):
        """Histogram: distribution of discounted prices."""
        if self.products is None:
            return _empty_fig('No product data available')

        prices = self.products['discounted_price_clean'].dropna()
        # Cap at 99th percentile for better visualization
        cap = prices.quantile(0.99)
        prices = prices[prices <= cap]

        fig = px.histogram(
            prices,
            nbins=80,
            title='Price Distribution (Discounted Prices)',
            labels={'value': 'Price (₹)', 'count': 'Number of Products'},
            template=DARK_TEMPLATE,
            color_discrete_sequence=['#818cf8'],
        )
        fig.update_layout(height=450, showlegend=False)
        return fig

    def top_expensive_categories(self):
        """Bar chart: top 15 most expensive categories."""
        if not self.category_stats:
            return _empty_fig('No category data available')

        data = []
        for cat, stats in self.category_stats.items():
            data.append({
                'Category': cat,
                'Avg Retail Price (₹)': stats.get('avg_retail', 0),
                'Avg Discounted Price (₹)': stats.get('avg_discounted', 0),
            })

        df = pd.DataFrame(data).sort_values('Avg Retail Price (₹)', ascending=False).head(15)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Retail Price', x=df['Category'], y=df['Avg Retail Price (₹)'],
            marker_color='#f87171',
        ))
        fig.add_trace(go.Bar(
            name='Discounted Price', x=df['Category'], y=df['Avg Discounted Price (₹)'],
            marker_color='#34d399',
        ))
        fig.update_layout(
            title='Top 15 Most Expensive Categories',
            barmode='group',
            template=DARK_TEMPLATE,
            height=500,
        )
        return fig

    def discount_distribution(self):
        """Histogram: distribution of discount percentages."""
        if not self.category_stats:
            return _empty_fig('No data available')

        data = []
        for cat, stats in self.category_stats.items():
            data.append({
                'Category': cat,
                'Avg Discount %': stats.get('avg_discount_pct', 0),
            })

        df = pd.DataFrame(data).sort_values('Avg Discount %', ascending=False)

        fig = px.bar(
            df, x='Category', y='Avg Discount %',
            title='Average Discount by Category',
            template=DARK_TEMPLATE,
            color='Avg Discount %',
            color_continuous_scale='RdYlGn',
        )
        fig.update_layout(height=500, showlegend=False)
        return fig

    def brand_vs_rating(self):
        """Scatter/bubble chart: brand price vs rating."""
        if self.products is None:
            return _empty_fig('No product data available')

        # Get top brands by product count
        top_brands = self.products['brand'].value_counts().head(20).index
        df = self.products[self.products['brand'].isin(top_brands)].copy()

        df['rating'] = pd.to_numeric(df['product_rating'], errors='coerce')
        df = df.dropna(subset=['rating', 'discounted_price_clean'])

        brand_stats = df.groupby('brand').agg(
            avg_price=('discounted_price_clean', 'mean'),
            avg_rating=('rating', 'mean'),
            count=('brand', 'size'),
        ).reset_index()

        fig = px.scatter(
            brand_stats,
            x='avg_rating', y='avg_price',
            size='count', text='brand',
            title='Brand: Price vs Rating',
            labels={'avg_rating': 'Average Rating', 'avg_price': 'Average Price (₹)', 'count': 'Products'},
            template=DARK_TEMPLATE,
            color='avg_price',
            color_continuous_scale='Viridis',
        )
        fig.update_traces(textposition='top center', textfont_size=9)
        fig.update_layout(height=550, showlegend=False)
        return fig

    def category_price_trends(self, categories=None, n_categories=5):
        """Line chart: price trends over time for top categories."""
        if not self.category_trends:
            return _empty_fig('No trend data available')

        if categories is None:
            categories = list(self.category_trends.keys())[:n_categories]

        fig = go.Figure()
        for cat in categories:
            trend = self.category_trends.get(cat)
            if not trend:
                continue
            dates = trend['dates']
            prices = trend['median_prices']
            # Filter out None values
            valid = [(d, p) for d, p in zip(dates, prices) if p is not None]
            if not valid:
                continue
            d, p = zip(*valid)
            fig.add_trace(go.Scatter(
                x=list(d), y=list(p),
                mode='lines', name=cat,
                line=dict(width=2),
            ))

        fig.update_layout(
            title='Category Price Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Median Price (₹)',
            template=DARK_TEMPLATE,
            height=500,
            hovermode='x unified',
        )
        return fig

    def summary_stats(self):
        """Return summary statistics as a dict."""
        if self.products is None:
            return {}

        dp = self.products['discounted_price_clean'].dropna()
        rp = self.products['retail_price_clean'].dropna()

        return {
            'total_products': len(self.products),
            'total_categories': self.products['main_category'].nunique(),
            'total_brands': self.products['brand'].nunique(),
            'avg_price': round(float(dp.mean()), 2),
            'median_price': round(float(dp.median()), 2),
            'min_price': round(float(dp.min()), 2),
            'max_price': round(float(dp.max()), 2),
            'avg_retail': round(float(rp.mean()), 2),
            'avg_discount_pct': round(
                float(((rp - dp) / rp * 100).mean()), 1
            ) if len(rp) > 0 else 0,
        }


def _empty_fig(msg):
    """Return an empty Plotly figure with a message."""
    fig = go.Figure()
    fig.add_annotation(text=msg, xref='paper', yref='paper', x=0.5, y=0.5,
                       showarrow=False, font=dict(size=16, color='gray'))
    fig.update_layout(template=DARK_TEMPLATE, height=300)
    return fig
