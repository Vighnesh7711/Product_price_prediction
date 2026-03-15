"""
preprocessing.py
================
Centralized data loading and preprocessing utilities for the
Product Price Prediction project.
"""

import os
import re
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'flipkart_com-ecommerce_sample.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


def load_raw_data(path=None):
    """Load the raw Flipkart dataset."""
    path = path or RAW_DATA_PATH
    return pd.read_csv(path)


def clean_price(val):
    """Remove currency symbols and convert to float."""
    if pd.isna(val):
        return np.nan
    s = re.sub(r'[^\d.]', '', str(val))
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_prices(df):
    """Clean retail_price and discounted_price columns in-place."""
    df = df.copy()
    if 'retail_price' in df.columns:
        df['retail_price'] = df['retail_price'].apply(clean_price)
    if 'discounted_price' in df.columns:
        df['discounted_price'] = df['discounted_price'].apply(clean_price)
    return df


def extract_main_category(cat_tree):
    """Extract top-level category from product_category_tree string."""
    if pd.isna(cat_tree):
        return 'Unknown'
    parts = str(cat_tree).split('>>')
    main = parts[0].strip().replace('[', '').replace(']', '').replace('"', '')
    return main if main else 'Unknown'


def clean_rating(val, default=3.0):
    """Convert rating to float, returning default if invalid."""
    if pd.isna(val):
        return default
    try:
        return float(str(val).strip())
    except ValueError:
        return default


def get_price_columns(df):
    """Return sorted list of daily price columns (price_2025-05-01, etc.)."""
    return sorted([c for c in df.columns if c.startswith('price_20')])


def get_dates_from_columns(price_cols):
    """Extract date strings from price column names."""
    return [c.replace('price_', '') for c in price_cols]


def wide_to_long(df, product_col='uniq_id', category_col='main_category'):
    """
    Convert wide-format price history to long format.

    Input:  rows = products, columns include price_2025-05-01, ...
    Output: DataFrame with columns [uniq_id, main_category, date, price]
    """
    price_cols = get_price_columns(df)
    if not price_cols:
        return pd.DataFrame(columns=[product_col, category_col, 'date', 'price'])

    keep_cols = [product_col]
    if category_col in df.columns:
        keep_cols.append(category_col)

    long = df[keep_cols + price_cols].melt(
        id_vars=keep_cols,
        value_vars=price_cols,
        var_name='date_col',
        value_name='price',
    )
    long['date'] = pd.to_datetime(long['date_col'].str.replace('price_', ''))
    long = long.drop(columns=['date_col'])
    long['price'] = pd.to_numeric(long['price'], errors='coerce')
    long = long.dropna(subset=['price'])
    long = long.sort_values(['uniq_id', 'date']).reset_index(drop=True)
    return long


def prepare_full_dataset():
    """Load, clean, and return the full dataset with main_category added."""
    df = load_raw_data()
    df = clean_prices(df)
    df['main_category'] = df['product_category_tree'].apply(extract_main_category)

    # Clean ratings
    for col in ['product_rating', 'overall_rating']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: clean_rating(x, default=np.nan))

    # Clean brand
    if 'brand' in df.columns:
        df['brand'] = df['brand'].fillna('Unknown').str.strip()

    # Ensure price columns are numeric
    for col in get_price_columns(df):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df
