"""
category_detector.py
====================
Detect product category from product name / description using keyword matching.
"""

import re

CATEGORY_KEYWORDS = {
    'Clothing': [
        'shirt', 'dress', 'jeans', 'trouser', 'kurta', 'saree', 'jacket',
        'sweater', 'top', 'skirt', 'legging', 'blazer', 'coat', 'pant',
        'shorts', 'hoodie', 'tshirt', 't-shirt', 'lehenga', 'suit',
        'ethnic', 'cotton', 'silk', 'denim', 'kurti', 'tunic',
    ],
    'Electronics': [
        'phone', 'mobile', 'laptop', 'tablet', 'computer', 'smartwatch',
        'earphone', 'earbuds', 'headphone', 'speaker', 'camera', 'monitor',
        'keyboard', 'mouse', 'processor', 'ram', 'ssd', 'charger', 'adapter',
        'bluetooth', 'router', 'tv', 'television', 'led', 'oled', 'gaming',
        'powerbank', 'power bank', 'pendrive', 'hard disk',
    ],
    'Footwear': [
        'shoe', 'sandal', 'slipper', 'boot', 'sneaker', 'heel', 'loafer',
        'flip flop', 'floater', 'jogger', 'footwear',
    ],
    'Home Furnishing': [
        'bedsheet', 'curtain', 'pillow', 'cushion', 'blanket', 'towel',
        'mat', 'rug', 'carpet', 'bed cover', 'sofa cover',
    ],
    'Furniture': [
        'sofa', 'table', 'chair', 'desk', 'bed', 'shelf', 'cabinet',
        'wardrobe', 'bookcase', 'stool', 'rack', 'almira', 'furniture',
    ],
    'Kitchen & Dining': [
        'cookware', 'dinner set', 'cup', 'glass', 'plate', 'bowl', 'pan',
        'pot', 'cooker', 'mixer', 'grinder', 'juicer', 'kettle', 'toaster',
    ],
    'Beauty & Personal Care': [
        'shampoo', 'conditioner', 'face wash', 'cream', 'lotion',
        'lipstick', 'mascara', 'foundation', 'perfume', 'deodorant',
        'sunscreen', 'serum', 'moisturizer', 'soap', 'body wash',
    ],
    'Jewellery': [
        'ring', 'necklace', 'bracelet', 'earring', 'pendant', 'chain',
        'bangle', 'anklet', 'gold', 'silver', 'diamond', 'jewellery',
    ],
    'Bags, Wallets & Belts': [
        'backpack', 'handbag', 'wallet', 'purse', 'clutch', 'sling bag',
        'laptop bag', 'travel bag', 'luggage', 'suitcase', 'belt',
    ],
    'Sports & Fitness': [
        'cricket', 'football', 'badminton', 'gym', 'yoga', 'fitness',
        'dumbbell', 'treadmill', 'cycle', 'bicycle',
    ],
    'Appliances': [
        'refrigerator', 'fridge', 'washing machine', 'air conditioner',
        'microwave', 'oven', 'vacuum', 'iron', 'fan', 'heater', 'geyser',
        'water purifier', 'inverter',
    ],
    'Automotive': [
        'car', 'bike', 'helmet', 'tyre', 'battery', 'dash cam',
    ],
    'Books': [
        'book', 'novel', 'textbook', 'guide', 'fiction', 'comics', 'manga',
    ],
    'Toys & Baby Care': [
        'toy', 'doll', 'puzzle', 'game', 'baby', 'diaper', 'stroller',
    ],
    'Watches': [
        'watch', 'wristwatch', 'analog watch', 'digital watch', 'chronograph',
    ],
}


class CategoryDetector:
    def __init__(self, dataset_categories=None):
        self.keywords = CATEGORY_KEYWORDS
        self.dataset_categories = dataset_categories or list(CATEGORY_KEYWORDS.keys())

    def detect(self, product_name: str, description: str = '',
               category_path: str = '') -> str:
        """Return the best-matching category string."""
        # 1. Try the scraped breadcrumb path first
        if category_path:
            main = category_path.split('>>')[0].strip()
            matched = self._match_to_known(main)
            if matched:
                return matched

        # 2. Keyword scoring
        text = f"{product_name} {description}".lower()
        scores = {}
        for cat, kws in self.keywords.items():
            score = 0
            for kw in kws:
                if kw in text:
                    score += 1
                    if kw in product_name.lower():
                        score += 2
            if score:
                scores[cat] = score

        return max(scores, key=scores.get) if scores else 'Other'

    def get_confidence(self, product_name: str,
                       description: str = '') -> tuple:
        """Return (category, confidence_float)."""
        text = f"{product_name} {description}".lower()
        scores = {}
        for cat, kws in self.keywords.items():
            s = sum(1 for k in kws if k in text)
            if s:
                scores[cat] = s

        if not scores:
            return 'Other', 0.0
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        return best, round(scores[best] / total, 2)

    # ── helpers ───────────────────────────────────────────────────
    def _match_to_known(self, text: str):
        t = text.lower()
        for cat in self.dataset_categories:
            if cat.lower() in t or t in cat.lower():
                return cat
        for cat, kws in self.keywords.items():
            if any(kw in t for kw in kws):
                return cat
        return None
