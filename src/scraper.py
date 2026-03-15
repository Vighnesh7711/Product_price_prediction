"""
scraper.py
==========
Product link scraper wrapper.
Reuses the existing ProductScraper from the dashboard module and
prepares scraped data for the prediction pipeline.
"""

import os
import sys
import re

# Add dashboard package to path so we can import the existing scraper
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'dashboard'))

from scraper.product_scraper import ProductScraper
from ml.category_detector import CategoryDetector


_scraper = ProductScraper()
_detector = CategoryDetector()


def scrape_product(url):
    """
    Scrape a product URL and return structured product data.

    Supports: Flipkart, Amazon, generic sites.

    Returns dict with keys:
        name, brand, category, current_price, original_price,
        discount_pct, rating, num_reviews, description,
        image, source, url, scrape_failed, fail_reason
    """
    raw = _scraper.scrape(url)

    if raw.get('scrape_failed'):
        return {
            'scrape_failed': True,
            'fail_reason': raw.get('fail_reason', 'Could not access the product page.'),
            'url': url,
        }

    current_price = float(raw.get('price', 0) or 0)
    original_price = float(raw.get('original_price', 0) or 0)

    # Calculate discount
    discount_pct = 0.0
    if original_price > 0 and current_price > 0:
        discount_pct = round((1 - current_price / original_price) * 100, 1)
    elif current_price > 0 and original_price == 0:
        original_price = current_price

    # Detect category
    name = raw.get('name', 'Unknown Product')
    description = raw.get('description', '')
    category_path = raw.get('category_path', '')
    category = _detector.detect(name, description, category_path)

    # Rating
    rating = 0.0
    if raw.get('rating'):
        try:
            rating = float(raw['rating'])
        except (ValueError, TypeError):
            rating = 0.0

    # Reviews
    num_reviews = 0
    if raw.get('reviews'):
        try:
            num_reviews = int(re.sub(r'[^\d]', '', str(raw['reviews'])))
        except (ValueError, TypeError):
            num_reviews = 0

    return {
        'scrape_failed': False,
        'name': name,
        'brand': raw.get('brand', 'Unknown'),
        'category': category,
        'category_path': category_path,
        'current_price': current_price,
        'original_price': original_price,
        'discount_pct': max(0, discount_pct),
        'rating': rating,
        'num_reviews': num_reviews,
        'description': description,
        'image': raw.get('image', ''),
        'specifications': raw.get('specifications', {}),
        'source': raw.get('source', 'unknown'),
        'url': url,
    }
