"""
festival_effect.py
==================
Analyze the impact of Indian festivals on product prices.
Uses historical price data to compute festival_impact_index
and estimate expected prices during festivals.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Indian festivals and sale events with approximate dates
# Data range: 2025-05-01 to 2026-01-31
FESTIVALS = {
    'Independence Day': {
        'dates': ['2025-08-15'],
        'window_before': 7,
        'window_after': 3,
        'description': 'Independence Day Sale',
    },
    'Dussehra': {
        'dates': ['2025-10-02'],
        'window_before': 5,
        'window_after': 2,
        'description': 'Dussehra/Navratri Offers',
    },
    'Diwali': {
        'dates': ['2025-10-20'],
        'window_before': 10,
        'window_after': 5,
        'description': 'Diwali Festive Sale',
    },
    'Holi': {
        'dates': ['2026-03-04'],
        'window_before': 5,
        'window_after': 2,
        'description': 'Holi Sale',
    },
    'Republic Day': {
        'dates': ['2026-01-26'],
        'window_before': 5,
        'window_after': 3,
        'description': 'Republic Day Sale',
    },
    'Big Billion Days': {
        'dates': ['2025-09-26'],
        'window_before': 0,
        'window_after': 7,
        'description': 'Flipkart Big Billion Days',
    },
    'Amazon Great Indian Sale': {
        'dates': ['2025-08-05', '2026-01-15'],
        'window_before': 0,
        'window_after': 5,
        'description': 'Amazon Great Indian Festival/Sale',
    },
}


class FestivalAnalyzer:
    """Analyze price changes during Indian festivals and sale events."""

    def __init__(self, models_dir=None):
        models_dir = models_dir or MODELS_DIR
        self.category_trends = {}
        self.price_history = None
        self.dates = []
        self.festival_impacts = {}

        # Load data
        trends_path = os.path.join(models_dir, 'dashboard_category_trends.json')
        if os.path.exists(trends_path):
            with open(trends_path) as f:
                self.category_trends = json.load(f)

        history_path = os.path.join(models_dir, 'dashboard_price_history.pkl')
        if os.path.exists(history_path):
            try:
                self.price_history = pd.read_pickle(history_path)
            except Exception:
                self.price_history = None

        if self.category_trends:
            sample = next(iter(self.category_trends.values()))
            self.dates = pd.to_datetime(sample.get('dates', []))

        self._compute_all_impacts()

    def _compute_all_impacts(self):
        """Pre-compute festival impact indices for all categories."""
        if not self.category_trends or len(self.dates) == 0:
            return

        for cat, trend in self.category_trends.items():
            prices = np.array(trend['median_prices'], dtype=float)
            cat_impacts = {}

            for fest_name, fest_info in FESTIVALS.items():
                impacts = []
                for date_str in fest_info['dates']:
                    fest_date = pd.Timestamp(date_str)

                    # Check if festival falls within our data range
                    if fest_date < self.dates[0] or fest_date > self.dates[-1]:
                        continue

                    # Find index of festival date
                    idx = self._find_date_index(fest_date)
                    if idx is None:
                        continue

                    w_before = fest_info['window_before']
                    w_after = fest_info['window_after']

                    # Festival window prices
                    fest_start = max(0, idx - w_before)
                    fest_end = min(len(prices), idx + w_after + 1)
                    fest_prices = prices[fest_start:fest_end]
                    fest_prices = fest_prices[~np.isnan(fest_prices)]

                    # Baseline: 30 days before the festival window
                    base_end = max(0, fest_start)
                    base_start = max(0, base_end - 30)
                    base_prices = prices[base_start:base_end]
                    base_prices = base_prices[~np.isnan(base_prices)]

                    if len(fest_prices) > 0 and len(base_prices) > 0:
                        fest_avg = float(np.mean(fest_prices))
                        base_avg = float(np.mean(base_prices))
                        if base_avg > 0:
                            impact = (fest_avg - base_avg) / base_avg * 100
                            impacts.append(impact)

                if impacts:
                    avg_impact = float(np.mean(impacts))
                    cat_impacts[fest_name] = {
                        'impact_pct': round(avg_impact, 2),
                        'description': fest_info['description'],
                        'direction': 'drop' if avg_impact < 0 else 'increase',
                    }

            if cat_impacts:
                self.festival_impacts[cat] = cat_impacts

    def _find_date_index(self, target_date):
        """Find the index of target_date in self.dates (nearest match)."""
        if len(self.dates) == 0:
            return None
        diffs = np.abs(self.dates - target_date)
        idx = int(diffs.argmin())
        if diffs[idx].days > 3:
            return None
        return idx

    def get_festival_impact(self, category):
        """
        Get festival impact data for a category.

        Returns dict: {festival_name: {impact_pct, description, direction}}
        """
        return self.festival_impacts.get(category, {})

    def estimate_festival_price(self, current_price, category, festival_name=None):
        """
        Estimate expected prices during festivals.

        Returns list of dicts:
        [{festival, expected_price, expected_discount_pct, confidence}]
        """
        cat_impacts = self.festival_impacts.get(category, {})
        results = []

        festivals_to_check = (
            {festival_name: cat_impacts[festival_name]}
            if festival_name and festival_name in cat_impacts
            else cat_impacts
        )

        if not festivals_to_check:
            # Use overall averages as fallback
            for fest_name, fest_info in FESTIVALS.items():
                results.append({
                    'festival': fest_name,
                    'description': fest_info['description'],
                    'current_price': round(current_price, 2),
                    'expected_price': round(current_price * 0.90, 2),
                    'expected_discount_pct': 10.0,
                    'confidence': 'low',
                    'data_available': False,
                })
            return results

        for fest_name, impact_data in festivals_to_check.items():
            impact_pct = impact_data['impact_pct']
            expected_price = current_price * (1 + impact_pct / 100)
            expected_price = max(0, expected_price)

            discount_from_current = 0.0
            if current_price > 0:
                discount_from_current = max(0, (1 - expected_price / current_price) * 100)

            # Confidence based on magnitude and data
            confidence = 'medium'
            if abs(impact_pct) > 15:
                confidence = 'high'
            elif abs(impact_pct) < 3:
                confidence = 'low'

            results.append({
                'festival': fest_name,
                'description': impact_data['description'],
                'current_price': round(current_price, 2),
                'expected_price': round(expected_price, 2),
                'expected_discount_pct': round(discount_from_current, 1),
                'impact_pct': impact_pct,
                'direction': impact_data['direction'],
                'confidence': confidence,
                'data_available': True,
            })

        # Sort by expected discount (biggest drops first)
        results.sort(key=lambda x: x.get('expected_discount_pct', 0), reverse=True)
        return results

    def get_all_festivals(self):
        """Return list of all tracked festivals."""
        return list(FESTIVALS.keys())

    def get_festival_dates(self):
        """Return festival dates info."""
        return {name: info['dates'] for name, info in FESTIVALS.items()}

    def get_categories(self):
        """Return categories with festival impact data."""
        return list(self.festival_impacts.keys())
