"""
similarity_engine.py
====================
Find similar products by category, price range, and name-word overlap.
"""

import pandas as pd
import numpy as np
import os
import re


class SimilarityEngine:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.products = None
        path = os.path.join(models_dir, 'dashboard_products.pkl')
        if os.path.exists(path):
            self.products = pd.read_pickle(path)

    def find_similar(self, product_name: str, category: str,
                     price: float, top_n: int = 6) -> list:
        if self.products is None or self.products.empty:
            return []

        pool = self.products[self.products['main_category'] == category].copy()
        if pool.empty:
            pool = self.products.copy()

        # narrow by price band (±50 %)
        if price > 0:
            lo, hi = price * 0.5, price * 1.5
            filtered = pool[pool['discounted_price_clean'].between(lo, hi)]
            if len(filtered) >= 3:
                pool = filtered

        # name-word overlap score
        words = set(re.findall(r'\w+', product_name.lower()))

        def _name_sim(n):
            if pd.isna(n):
                return 0
            rw = set(re.findall(r'\w+', str(n).lower()))
            union = words | rw
            return len(words & rw) / len(union) if union else 0

        pool = pool.copy()
        pool['sim'] = pool['product_name'].apply(_name_sim)

        if price > 0:
            pool['pscore'] = (
                1 - (pool['discounted_price_clean'] - price).abs() / max(price, 1)
            ).clip(0, 1)
        else:
            pool['pscore'] = 0.5

        pool['score'] = pool['sim'] * 0.6 + pool['pscore'] * 0.4
        top = pool.nlargest(top_n, 'score')

        results = []
        for _, r in top.iterrows():
            def _safe(col, default=0):
                v = r.get(col, default)
                if pd.isna(v):
                    return default
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return default

            results.append({
                'id': str(r.get('uniq_id', '')),
                'name': str(r.get('product_name', 'Unknown'))[:100],
                'category': str(r.get('main_category', '')),
                'brand': str(r.get('brand', 'Unknown')),
                'price': _safe('discounted_price_clean'),
                'original_price': _safe('retail_price_clean'),
                'rating': _safe('product_rating'),
                'similarity': round(float(r.get('score', 0)), 2),
                'last_price': _safe('last_price'),
                'first_price': _safe('first_price'),
            })
        return results

    def get_category_stats(self, category: str) -> dict:
        if self.products is None:
            return {}
        c = self.products[self.products['main_category'] == category]
        if c.empty:
            return {}
        rp = c['retail_price_clean']
        dp = c['discounted_price_clean']
        
        # safely compute numerical mean for ratings
        ratings = pd.to_numeric(c['product_rating'], errors='coerce').dropna()
        
        return {
            'total_products': int(len(c)),
            'avg_price': round(float(dp.mean()), 2) if not dp.isna().all() else 0,
            'min_price': round(float(dp.min()), 2) if not dp.isna().all() else 0,
            'max_price': round(float(dp.max()), 2) if not dp.isna().all() else 0,
            'avg_discount_pct': round(
                float(((rp - dp) / rp * 100).mean()), 1
            ) if not rp.isna().all() else 0,
            'top_brands': c['brand'].dropna().value_counts().head(5).to_dict(),
            'avg_rating': round(float(ratings.mean()), 1) if not ratings.empty else 0,
        }
