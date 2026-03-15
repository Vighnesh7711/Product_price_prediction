"""
prepare_dashboard.py
====================
Process the Flipkart dataset and create all artifacts needed by the dashboard.

Usage:
    python src/prepare_dashboard.py

Output files (saved to models/):
    - dashboard_products.pkl
    - dashboard_category_trends.json
    - dashboard_seasonal.json
    - dashboard_category_stats.json
    - dashboard_config.json
    - dashboard_price_history.pkl
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import re
import sys

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'flipkart_com-ecommerce_sample.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


def clean_price(val):
    if pd.isna(val):
        return np.nan
    s = re.sub(r'[^\d.]', '', str(val))
    try:
        return float(s)
    except ValueError:
        return np.nan


def extract_main_category(cat_tree):
    if pd.isna(cat_tree):
        return 'Unknown'
    parts = str(cat_tree).split('>>')
    main = parts[0].strip().replace('[', '').replace(']', '').replace('"', '')
    return main if main else 'Unknown'


def main():
    print("=" * 60)
    print("  PREPARING DASHBOARD DATA")
    print("=" * 60)

    # ── 1. Load dataset ───────────────────────────────────────────
    print("\n Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded: {len(df):,} products x {len(df.columns)} columns")

    # ── 2. Identify columns ──────────────────────────────────────
    price_date_cols = sorted([c for c in df.columns if c.startswith('price_20')])
    dates = [c.replace('price_', '') for c in price_date_cols]
    print(f"   Price date columns: {len(price_date_cols)}")
    print(f"   Date range: {dates[0]} to {dates[-1]}")

    # ── 3. Extract categories ────────────────────────────────────
    print("\n Extracting categories...")
    df['main_category'] = df['product_category_tree'].apply(extract_main_category)
    categories = df['main_category'].value_counts()
    print(f"   Found {len(categories)} categories")
    print(f"   Top 10: {categories.head(10).index.tolist()}")

    # ── 4. Clean prices ──────────────────────────────────────────
    print("\n Cleaning prices...")
    df['retail_price_clean'] = df['retail_price'].apply(clean_price)
    df['discounted_price_clean'] = df['discounted_price'].apply(clean_price)
    for col in price_date_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── 5. Compute category trends ───────────────────────────────
    print("\n Computing category trends...")
    category_trends = {}
    for cat in categories.head(25).index:
        cat_df = df[df['main_category'] == cat]
        if len(cat_df) < 3:
            continue

        daily_prices = []
        for col in price_date_cols:
            med = cat_df[col].median()
            daily_prices.append(round(float(med), 2) if not pd.isna(med) else None)

        first_valid = next((p for p in daily_prices if p and p > 0), None)
        if first_valid:
            price_index = [round(p / first_valid * 100, 2) if p else None for p in daily_prices]
        else:
            price_index = daily_prices

        valid = [p for p in daily_prices if p is not None and p > 0]

        category_trends[cat] = {
            'dates': dates,
            'median_prices': daily_prices,
            'price_index': price_index,
            'avg_price': round(float(np.mean(valid)), 2) if valid else 0,
            'min_price': round(float(np.min(valid)), 2) if valid else 0,
            'max_price': round(float(np.max(valid)), 2) if valid else 0,
            'volatility': round(float(np.std(valid) / np.mean(valid) * 100), 2) if valid and np.mean(valid) > 0 else 0,
            'product_count': int(len(cat_df)),
            'trend_direction': 'up' if valid and valid[-1] > valid[0] else 'down',
        }
    print(f"   Computed trends for {len(category_trends)} categories")

    # ── 6. Compute seasonal patterns ─────────────────────────────
    print("\n Computing seasonal patterns...")
    seasonal = {}
    for cat, trend in category_trends.items():
        monthly_agg = {}
        for date_str, price in zip(dates, trend['median_prices']):
            if price is None:
                continue
            month = date_str[:7]
            monthly_agg.setdefault(month, []).append(price)
        monthly_data = {}
        for month, prices in monthly_agg.items():
            monthly_data[month] = {
                'avg': round(float(np.mean(prices)), 2),
                'min': round(float(np.min(prices)), 2),
                'max': round(float(np.max(prices)), 2),
            }
        seasonal[cat] = monthly_data

    # ── 7. Save processed products ───────────────────────────────
    print("\n Saving processed products...")
    products_df = df[['uniq_id', 'product_name', 'main_category', 'brand',
                       'retail_price_clean', 'discounted_price_clean',
                       'product_rating', 'overall_rating', 'description']].copy()
    products_df['last_price'] = df[price_date_cols[-1]]
    products_df['first_price'] = df[price_date_cols[0]]

    os.makedirs(MODELS_DIR, exist_ok=True)
    products_df.to_pickle(os.path.join(MODELS_DIR, 'dashboard_products.pkl'))

    with open(os.path.join(MODELS_DIR, 'dashboard_category_trends.json'), 'w') as f:
        json.dump(category_trends, f)
    with open(os.path.join(MODELS_DIR, 'dashboard_seasonal.json'), 'w') as f:
        json.dump(seasonal, f)

    # ── 8. Category stats ────────────────────────────────────────
    cat_stats = {}
    for cat in categories.head(25).index:
        cdf = df[df['main_category'] == cat]
        rp = cdf['retail_price_clean']
        dp = cdf['discounted_price_clean']
        cat_stats[cat] = {
            'count': int(len(cdf)),
            'avg_retail': round(float(rp.mean()), 2) if not rp.isna().all() else 0,
            'avg_discounted': round(float(dp.mean()), 2) if not dp.isna().all() else 0,
            'avg_discount_pct': round(float(((rp - dp) / rp * 100).mean()), 1) if not rp.isna().all() else 0,
            'brands': cdf['brand'].dropna().unique().tolist()[:20],
        }
    with open(os.path.join(MODELS_DIR, 'dashboard_category_stats.json'), 'w') as f:
        json.dump(cat_stats, f, default=str)

    # ── 9. Dashboard config ──────────────────────────────────────
    config = {
        'total_products': int(len(df)),
        'total_categories': int(len(categories)),
        'date_range': {'start': dates[0], 'end': dates[-1]},
        'top_categories': categories.head(25).index.tolist(),
        'total_price_dates': len(price_date_cols),
    }
    with open(os.path.join(MODELS_DIR, 'dashboard_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ── 10. Price history matrix ─────────────────────────────────
    print("\n Saving price history matrix...")
    price_matrix = df[['uniq_id', 'main_category'] + price_date_cols].copy()
    price_matrix.to_pickle(os.path.join(MODELS_DIR, 'dashboard_price_history.pkl'))

    print("\n" + "=" * 60)
    print("  DASHBOARD DATA PREPARED SUCCESSFULLY!")
    print("=" * 60)
    print(f"   Products:   {len(df):,}")
    print(f"   Categories: {len(category_trends)}")
    print(f"   Date range: {dates[0]} to {dates[-1]}")
    print(f"\n   Files saved to: {MODELS_DIR}")
    print("     - dashboard_products.pkl")
    print("     - dashboard_category_trends.json")
    print("     - dashboard_seasonal.json")
    print("     - dashboard_category_stats.json")
    print("     - dashboard_config.json")
    print("     - dashboard_price_history.pkl")
    print("\n   Next: python dashboard/app.py")


if __name__ == '__main__':
    main()
