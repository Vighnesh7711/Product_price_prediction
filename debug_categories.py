#!/usr/bin/env python3
"""
Debug script to check category detection vs available category stats
"""
import json
import os

# Load available categories from the dashboard data
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
cat_stats_path = os.path.join(models_dir, 'dashboard_category_stats.json')
config_path = os.path.join(models_dir, 'dashboard_config.json')

# Load available categories
available_categories = set()
if os.path.exists(cat_stats_path):
    with open(cat_stats_path) as f:
        available_categories = set(json.load(f).keys())

dashboard_cats = set()
if os.path.exists(config_path):
    with open(config_path) as f:
        dashboard_cats = set(json.load(f).get('top_categories', []))

# Import category detector from dashboard
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard'))
from ml.category_detector import CategoryDetector

detector = CategoryDetector(dataset_categories=list(dashboard_cats))

print("Available categories in dashboard_category_stats.json:")
for cat in sorted(available_categories):
    print(f"  - {cat}")

print(f"\nDashboard top categories ({len(dashboard_cats)}):")
for cat in sorted(dashboard_cats):
    print(f"  - {cat}")

print(f"\nCategory detector keywords ({len(detector.keywords)}):")
for cat in sorted(detector.keywords.keys()):
    print(f"  - {cat}")

# Test some sample products
test_products = [
    ("Apple MacBook Pro", "High-performance laptop", ""),
    ("iPhone 14", "Smartphone with great camera", ""),
    ("Samsung TV", "Smart LED television", ""),
    ("Nike Running Shoes", "Comfortable sports footwear", ""),
    ("Levi's Jeans", "Blue denim jeans", ""),
]

print(f"\nTesting category detection:")
for name, desc, path in test_products:
    detected_cat = detector.detect(name, desc, path)
    is_available = detected_cat in available_categories
    print(f"  '{name}' -> '{detected_cat}' (Available: {is_available})")

# Find mismatches
detector_cats = set(detector.keywords.keys())
missing_in_stats = detector_cats - available_categories
missing_in_detector = available_categories - detector_cats

print(f"\nCategory mismatches:")
if missing_in_stats:
    print(f"  Categories in detector but NOT in stats: {sorted(missing_in_stats)}")
if missing_in_detector:
    print(f"  Categories in stats but NOT in detector: {sorted(missing_in_detector)[:10]}...")