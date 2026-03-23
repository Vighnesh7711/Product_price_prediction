"""
category_detector.py
====================
Detect product category from product name / description using keyword matching.
Updated to match dashboard category stats and provide better accuracy.
"""

import re

# Updated keywords to match actual dashboard categories exactly
CATEGORY_KEYWORDS = {
    'Clothing': [
        'shirt', 'dress', 'jeans', 'trouser', 'kurta', 'saree', 'jacket',
        'sweater', 'top', 'skirt', 'legging', 'blazer', 'coat', 'pant',
        'shorts', 'hoodie', 'tshirt', 't-shirt', 'lehenga', 'suit',
        'ethnic', 'cotton', 'silk', 'denim', 'kurti', 'tunic', 'clothing',
    ],
    'Computers': [
        'laptop', 'computer', 'desktop', 'monitor', 'keyboard', 'mouse',
        'processor', 'ram', 'ssd', 'hard disk', 'motherboard', 'graphics card',
        'webcam', 'router', 'modem', 'printer', 'scanner', 'ups', 'macbook',
        'imac', 'pc', 'workstation', 'server', 'pendrive', 'usb', 'networking',
    ],
    'Mobiles & Accessories': [
        'phone', 'mobile', 'smartphone', 'iphone', 'android', 'charger',
        'adapter', 'earphone', 'earbuds', 'headphone', 'powerbank', 'power bank',
        'case', 'cover', 'screen guard', 'tempered glass', 'bluetooth',
        'wireless charger', 'car charger', 'data cable', 'selfie stick',
    ],
    'Cameras & Accessories': [
        'camera', 'dslr', 'mirrorless', 'lens', 'tripod', 'flash', 'memory card',
        'camera bag', 'filter', 'battery grip', 'remote shutter', 'gopro',
        'action camera', 'webcam', 'camcorder', 'photography',
    ],
    'Footwear': [
        'shoe', 'sandal', 'slipper', 'boot', 'sneaker', 'heel', 'loafer',
        'flip flop', 'floater', 'jogger', 'footwear', 'running shoes',
        'formal shoes', 'casual shoes', 'sports shoes',
    ],
    'Home Furnishing': [
        'bedsheet', 'curtain', 'pillow', 'cushion', 'blanket', 'towel',
        'mat', 'rug', 'carpet', 'bed cover', 'sofa cover', 'table cloth',
        'door mat', 'bath mat', 'quilt', 'comforter',
    ],
    'Furniture': [
        'sofa', 'table', 'chair', 'desk', 'bed', 'shelf', 'cabinet',
        'wardrobe', 'bookcase', 'stool', 'rack', 'almira', 'furniture',
        'dining table', 'coffee table', 'office chair', 'recliner',
    ],
    'Kitchen & Dining': [
        'cookware', 'dinner set', 'cup', 'glass', 'plate', 'bowl', 'pan',
        'pot', 'cooker', 'mixer', 'grinder', 'juicer', 'kettle', 'toaster',
        'blender', 'food processor', 'pressure cooker', 'non stick',
        'steel utensils', 'cutlery', 'spoon', 'fork', 'knife',
    ],
    'Beauty and Personal Care': [  # Note: "and" not "&"
        'shampoo', 'conditioner', 'face wash', 'cream', 'lotion',
        'lipstick', 'mascara', 'foundation', 'perfume', 'deodorant',
        'sunscreen', 'serum', 'moisturizer', 'soap', 'body wash',
        'hair oil', 'face pack', 'nail polish', 'makeup', 'skincare',
    ],
    'Jewellery': [
        'ring', 'necklace', 'bracelet', 'earring', 'pendant', 'chain',
        'bangle', 'anklet', 'gold', 'silver', 'diamond', 'jewellery',
        'jewelry', 'artificial jewellery', 'fashion jewellery',
    ],
    'Bags, Wallets & Belts': [
        'backpack', 'handbag', 'wallet', 'purse', 'clutch', 'sling bag',
        'laptop bag', 'travel bag', 'luggage', 'suitcase', 'belt',
        'messenger bag', 'tote bag', 'duffel bag', 'briefcase',
    ],
    'Sports & Fitness': [
        'cricket', 'football', 'badminton', 'gym', 'yoga', 'fitness',
        'dumbbell', 'treadmill', 'cycle', 'bicycle', 'sports', 'exercise',
        'weight', 'fitness equipment', 'tennis', 'basketball',
    ],
    'Health & Personal Care Appliances': [
        'hair dryer', 'trimmer', 'shaver', 'epilator', 'massager',
        'blood pressure monitor', 'thermometer', 'weighing scale',
        'humidifier', 'air purifier', 'nebulizer', 'dental care',
    ],
    'Home & Kitchen': [
        'refrigerator', 'fridge', 'washing machine', 'air conditioner',
        'microwave', 'oven', 'vacuum', 'iron', 'fan', 'heater', 'geyser',
        'water purifier', 'inverter', 'dishwasher', 'chimney', 'cooler',
        'water heater', 'electric kettle', 'rice cooker', 'induction',
    ],
    'Automotive': [
        'car', 'bike', 'helmet', 'tyre', 'battery', 'dash cam',
        'car accessories', 'bike accessories', 'automotive', 'vehicle',
    ],
    'Baby Care': [
        'baby', 'diaper', 'stroller', 'baby food', 'baby care',
        'infant', 'newborn', 'feeding bottle', 'baby clothes',
        'baby toys', 'baby carrier', 'baby monitor',
    ],
    'Toys & School Supplies': [
        'toy', 'doll', 'puzzle', 'game', 'school', 'stationery',
        'notebook', 'pen', 'pencil', 'bag pack', 'lunch box',
        'educational toy', 'board game', 'action figure', 'book',
    ],
    'Pens & Stationery': [
        'pen', 'pencil', 'marker', 'highlighter', 'notebook', 'diary',
        'file', 'folder', 'stapler', 'paper', 'eraser', 'ruler',
        'calculator', 'adhesive', 'tape', 'stationery',
    ],
    'Watches': [
        'watch', 'wristwatch', 'analog watch', 'digital watch', 'chronograph',
        'smartwatch', 'fitness band', 'smart band', 'apple watch',
        'timepiece', 'wearable',
    ],
    'Gaming': [
        'gaming', 'game', 'console', 'controller', 'joystick', 'gaming chair',
        'gaming keyboard', 'gaming mouse', 'gaming headset', 'xbox',
        'playstation', 'nintendo', 'steam', 'pc game',
    ],
    'Tools & Hardware': [
        'tool', 'hammer', 'screwdriver', 'drill', 'saw', 'plier',
        'wrench', 'hardware', 'nuts', 'bolts', 'nails', 'screws',
        'measuring tape', 'toolbox', 'power tool',
    ],
    'Home Decor & Festive Needs': [
        'home decor', 'decoration', 'wall art', 'photo frame', 'vase',
        'candle', 'lamp', 'lighting', 'festive', 'diwali', 'christmas',
        'artificial flowers', 'showpiece', 'figurine', 'wall sticker',
    ],
    'Home Improvement': [
        'paint', 'brush', 'adhesive', 'tile', 'flooring', 'ceiling fan',
        'light fixture', 'switch', 'socket', 'wire', 'plumbing',
        'bathroom fittings', 'door handle', 'lock', 'security',
    ],
    'Sunglasses': [
        'sunglasses', 'goggles', 'eyewear', 'shades', 'aviator',
        'wayfarer', 'uv protection', 'polarized', 'reading glasses',
    ],
    'Pet Supplies': [
        'pet', 'dog', 'cat', 'pet food', 'pet toys', 'pet accessories',
        'fish tank', 'aquarium', 'bird cage', 'pet care', 'leash',
        'collar', 'pet bed', 'pet carrier',
    ],
}

# Legacy category mapping for backward compatibility
CATEGORY_MAPPING = {
    'Electronics': 'Computers',  # Default electronics to computers
    'Beauty & Personal Care': 'Beauty and Personal Care',
    'Appliances': 'Home & Kitchen',
    'Books': 'Toys & School Supplies',  # Map books to school supplies as fallback
    'Toys & Baby Care': 'Baby Care',
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

        # 2. Keyword scoring with smart electronics detection
        text = f"{product_name} {description}".lower()
        scores = {}

        # Special handling for electronics to be more specific
        electronics_keywords = ['phone', 'mobile', 'iphone', 'smartphone']
        laptop_keywords = ['laptop', 'computer', 'macbook', 'pc', 'desktop']
        camera_keywords = ['camera', 'dslr', 'lens', 'photography']
        tv_keywords = ['tv', 'television', 'led', 'oled', 'smart tv']

        # Check for specific electronics subcategories first
        if any(kw in text for kw in electronics_keywords):
            scores['Mobiles & Accessories'] = scores.get('Mobiles & Accessories', 0) + 5
        if any(kw in text for kw in laptop_keywords):
            scores['Computers'] = scores.get('Computers', 0) + 5
        if any(kw in text for kw in camera_keywords):
            scores['Cameras & Accessories'] = scores.get('Cameras & Accessories', 0) + 5
        if any(kw in text for kw in tv_keywords):
            scores['Home & Kitchen'] = scores.get('Home & Kitchen', 0) + 3

        # Regular keyword scoring
        for cat, kws in self.keywords.items():
            score = scores.get(cat, 0)
            for kw in kws:
                if kw in text:
                    score += 1
                    if kw in product_name.lower():
                        score += 2  # Higher weight for product name matches
            if score > 0:
                scores[cat] = score

        if scores:
            detected_category = max(scores, key=scores.get)
            # Apply legacy mapping if needed
            return CATEGORY_MAPPING.get(detected_category, detected_category)

        return 'Clothing'  # Default fallback to most common category

    def get_confidence(self, product_name: str,
                       description: str = '') -> tuple:
        """Return (category, confidence_float)."""
        category = self.detect(product_name, description)

        text = f"{product_name} {description}".lower()
        scores = {}
        for cat, kws in self.keywords.items():
            s = sum(1 for k in kws if k in text)
            if s:
                scores[cat] = s

        if not scores:
            return category, 0.1

        # Apply legacy mapping to scores as well
        mapped_scores = {}
        for cat, score in scores.items():
            mapped_cat = CATEGORY_MAPPING.get(cat, cat)
            mapped_scores[mapped_cat] = mapped_scores.get(mapped_cat, 0) + score

        total = sum(mapped_scores.values())
        best_score = mapped_scores.get(category, 0)
        confidence = round(best_score / total, 2) if total > 0 else 0.1

        # Boost confidence if it's a perfect keyword match
        if any(kw in product_name.lower() for kw in self.keywords.get(category, [])):
            confidence = min(1.0, confidence + 0.2)

        return category, confidence

    # ── helpers ───────────────────────────────────────────────────
    def _match_to_known(self, text: str):
        t = text.lower()
        # First try exact matches with dataset categories
        for cat in self.dataset_categories:
            if cat.lower() in t or t in cat.lower():
                return cat

        # Then try keyword-based matching
        for cat, kws in self.keywords.items():
            if any(kw in t for kw in kws):
                return CATEGORY_MAPPING.get(cat, cat)

        return None