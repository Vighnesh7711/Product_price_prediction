"""
app.py  –  Price Prediction Dashboard
======================================
Flask server that serves the dashboard UI and exposes REST endpoints
for product analysis, price prediction, and similar-product search.

Run:
    python dashboard/app.py
"""

import os
import sys
import json
import traceback

from flask import Flask, render_template, request, jsonify

# ensure dashboard package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper.product_scraper import ProductScraper
from ml.category_detector import CategoryDetector
from ml.price_predictor import PricePredictor
from ml.similarity_engine import SimilarityEngine

# ── paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# ── components ────────────────────────────────────────────────────
scraper = ProductScraper()

# Load dataset category names if available
config_path = os.path.join(MODELS_DIR, 'dashboard_config.json')
ds_cats = None
if os.path.exists(config_path):
    with open(config_path) as _f:
        ds_cats = json.load(_f).get('top_categories')

category_detector = CategoryDetector(dataset_categories=ds_cats)
price_predictor = PricePredictor(MODELS_DIR)
similarity_engine = SimilarityEngine(MODELS_DIR)

# ── Flask app ─────────────────────────────────────────────────────
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Scrape a product URL and return full analysis."""
    data = request.get_json(silent=True) or {}
    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'Please provide a product URL.'}), 400

    try:
        product = scraper.scrape(url)

        # If scraping was blocked / timed out, return helpful message
        if product.get('scrape_failed'):
            return jsonify({
                'scrape_failed': True,
                'fail_reason': product.get('fail_reason', 'Could not access the product page.'),
            }), 200          # 200 so frontend handles it gracefully

        if not product.get('name'):
            return jsonify({'error': 'Could not extract product details from this page.'}), 400

        return jsonify(_build_response(product))
    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/api/manual-analyze', methods=['POST'])
def manual_analyze():
    """Analyse manually-entered product details (fallback)."""
    data = request.get_json(silent=True) or {}
    name = data.get('name', '').strip()
    price = float(data.get('price', 0))
    desc = data.get('description', '')

    if not name or price <= 0:
        return jsonify({'error': 'Product name and valid price are required.'}), 400

    try:
        product = {
            'name': name, 'price': price,
            'description': desc, 'source': 'manual', 'url': '',
        }
        return jsonify(_build_response(product))
    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/api/categories')
def get_categories():
    if os.path.exists(config_path):
        with open(config_path) as f:
            return jsonify({'categories': json.load(f).get('top_categories', [])})
    return jsonify({'categories': list(category_detector.keywords.keys())})


@app.route('/api/model-performance')
def model_performance():
    """Return global ML model performance scores."""
    return jsonify(price_predictor.get_performance_summary())


# ── helper ────────────────────────────────────────────────────────
def _build_response(product: dict) -> dict:
    cat = category_detector.detect(
        product.get('name', ''),
        product.get('description', ''),
        product.get('category_path', ''),
    )
    product['detected_category'] = cat
    _, conf = category_detector.get_confidence(
        product.get('name', ''), product.get('description', ''))
    product['category_confidence'] = conf

    price = product.get('price', 0)
    predictions = (price_predictor.predict(price, cat, 30)
                   if price > 0
                   else {'error': 'No price available for prediction'})

    similar = similarity_engine.find_similar(
        product.get('name', ''), cat, price, top_n=6)
    cat_stats = similarity_engine.get_category_stats(cat)
    cat_comp = price_predictor.get_category_comparison(cat)

    return {
        'product': product,
        'predictions': predictions,
        'similar_products': similar,
        'category_stats': cat_stats,
        'category_comparison': cat_comp,
    }


# ── run ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  PRICE PREDICTION DASHBOARD")
    print("=" * 60)
    print(f"   Models : {MODELS_DIR}")
    print(f"   Open   : http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
