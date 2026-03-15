"""
timeseries.py
=============
Time-series price forecasting using ARIMA and Prophet.
Converts wide-format price data to long format and generates
forecasts for 7, 30, and 90 days.
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


class TimeSeriesForecaster:
    """Forecast product/category prices using ARIMA and Prophet."""

    def __init__(self, models_dir=None):
        models_dir = models_dir or MODELS_DIR
        self.category_trends = {}
        self.price_history = None

        # Load category trends
        trends_path = os.path.join(models_dir, 'dashboard_category_trends.json')
        if os.path.exists(trends_path):
            with open(trends_path) as f:
                self.category_trends = json.load(f)

        # Load price history matrix
        history_path = os.path.join(models_dir, 'dashboard_price_history.pkl')
        if os.path.exists(history_path):
            try:
                self.price_history = pd.read_pickle(history_path)
            except Exception:
                self.price_history = None

    def get_category_series(self, category):
        """Get time series for a category as (dates, prices) arrays."""
        trend = self.category_trends.get(category)
        if not trend:
            return None, None

        dates = pd.to_datetime(trend['dates'])
        prices = np.array(trend['median_prices'], dtype=float)

        # Remove NaN
        mask = ~np.isnan(prices)
        return dates[mask], prices[mask]

    def get_product_series(self, product_id):
        """Get time series for a specific product."""
        if self.price_history is None:
            return None, None

        row = self.price_history[self.price_history['uniq_id'] == product_id]
        if row.empty:
            return None, None

        price_cols = sorted([c for c in self.price_history.columns if c.startswith('price_20')])
        prices = row[price_cols].values.flatten().astype(float)
        dates = pd.to_datetime([c.replace('price_', '') for c in price_cols])

        mask = ~np.isnan(prices)
        return dates[mask], prices[mask]

    def forecast_arima(self, dates, prices, days_ahead=30):
        """
        Forecast using ARIMA model.
        Returns dict with forecast dates, values, and confidence intervals.
        """
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        warnings.filterwarnings('ignore')

        if len(prices) < 30:
            return None

        # Try different ARIMA orders, pick best AIC
        best_aic = np.inf
        best_order = (5, 1, 0)
        for p in [3, 5, 7]:
            for d in [0, 1]:
                for q in [0, 1, 2]:
                    try:
                        model = ARIMA(prices, order=(p, d, q))
                        fit = model.fit()
                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue

        # Fit best model
        model = ARIMA(prices, order=best_order)
        fit = model.fit()
        forecast = fit.get_forecast(steps=days_ahead)
        pred_mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)

        last_date = dates[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead,
            freq='D',
        )

        # conf_int may be ndarray or DataFrame depending on statsmodels version
        if hasattr(conf_int, 'iloc'):
            lower_vals = conf_int.iloc[:, 0].values
            upper_vals = conf_int.iloc[:, 1].values
        else:
            ci = np.array(conf_int)
            lower_vals = ci[:, 0]
            upper_vals = ci[:, 1]

        return {
            'model': 'ARIMA',
            'order': best_order,
            'dates': forecast_dates,
            'predicted': np.maximum(0, pred_mean),
            'lower': np.maximum(0, lower_vals),
            'upper': upper_vals,
        }

    def forecast_prophet(self, dates, prices, days_ahead=30):
        """
        Forecast using Facebook Prophet.
        Returns dict with forecast dates, values, and confidence intervals.
        """
        from prophet import Prophet
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

        if len(prices) < 30:
            return None

        df = pd.DataFrame({'ds': dates, 'y': prices})

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True if len(prices) > 180 else False,
            changepoint_prior_scale=0.05,
        )

        # Add Indian holidays
        indian_holidays = pd.DataFrame({
            'holiday': [
                'Republic Day', 'Holi', 'Independence Day', 'Dussehra',
                'Diwali', 'Big Billion Days', 'Amazon Great Indian Sale',
            ],
            'ds': pd.to_datetime([
                '2025-01-26', '2025-03-14', '2025-08-15', '2025-10-02',
                '2025-10-20', '2025-10-01', '2025-08-05',
            ]),
            'lower_window': [-2, -1, -3, -2, -5, -7, -5],
            'upper_window': [1, 1, 3, 2, 5, 3, 3],
        })
        # Add 2026 dates
        holidays_2026 = pd.DataFrame({
            'holiday': [
                'Republic Day', 'Holi', 'Independence Day',
            ],
            'ds': pd.to_datetime([
                '2026-01-26', '2026-03-04', '2026-08-15',
            ]),
            'lower_window': [-2, -1, -3],
            'upper_window': [1, 1, 3],
        })
        all_holidays = pd.concat([indian_holidays, holidays_2026], ignore_index=True)
        model.holidays = all_holidays

        model.fit(df)

        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)

        # Extract only future dates
        future_forecast = forecast.tail(days_ahead)

        return {
            'model': 'Prophet',
            'dates': future_forecast['ds'].values,
            'predicted': np.maximum(0, future_forecast['yhat'].values),
            'lower': np.maximum(0, future_forecast['yhat_lower'].values),
            'upper': future_forecast['yhat_upper'].values,
        }

    def forecast(self, category=None, product_id=None, days_ahead=30):
        """
        Generate forecasts using both ARIMA and Prophet.

        Args:
            category: Category name to forecast
            product_id: Specific product ID to forecast
            days_ahead: Number of days to forecast (7, 30, or 90)

        Returns dict with ARIMA and Prophet results plus historical data.
        """
        if product_id:
            dates, prices = self.get_product_series(product_id)
        elif category:
            dates, prices = self.get_category_series(category)
        else:
            return {'error': 'Provide either category or product_id'}

        if dates is None or len(dates) < 30:
            return {'error': 'Insufficient data for forecasting'}

        results = {
            'historical': {
                'dates': dates,
                'prices': prices,
            },
            'days_ahead': days_ahead,
        }

        # ARIMA forecast
        try:
            results['arima'] = self.forecast_arima(dates, prices, days_ahead)
        except Exception as e:
            results['arima'] = {'error': str(e)}

        # Prophet forecast
        try:
            results['prophet'] = self.forecast_prophet(dates, prices, days_ahead)
        except Exception as e:
            results['prophet'] = {'error': str(e)}

        return results

    def get_categories(self):
        """Return list of available categories."""
        return list(self.category_trends.keys())
