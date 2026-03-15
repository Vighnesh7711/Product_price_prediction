"""
predictor.py
============
Cross-sectional price prediction pipeline.
Loads the trained Ridge Regression model and encoders to predict
fair prices from product features.
"""

import os
import pickle
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


class PricePredictionPipeline:
    """
    Predict fair price from product attributes using the trained Ridge model.

    Inputs: brand, category, rating, retail_price, discount_percentage
    Outputs: predicted_fair_price, expected_discount, confidence_score
    """

    def __init__(self, models_dir=None):
        models_dir = models_dir or MODELS_DIR
        self.model = None
        self.scaler = None
        self.brand_encoder = None
        self.cat_encoder = None
        self.loaded = False
        self._load(models_dir)

    def _load(self, models_dir):
        ridge_path = os.path.join(models_dir, 'ridge_model.pkl')
        brand_path = os.path.join(models_dir, 'brand_encoder.pkl')
        cat_path = os.path.join(models_dir, 'cat_encoder.pkl')

        if not all(os.path.exists(p) for p in [ridge_path, brand_path, cat_path]):
            print("[PricePredictionPipeline] Model files not found. Run train_models.py first.")
            return

        with open(ridge_path, 'rb') as f:
            ridge_data = pickle.load(f)
            self.model = ridge_data['model']
            self.scaler = ridge_data['scaler']

        with open(brand_path, 'rb') as f:
            self.brand_encoder = pickle.load(f)

        with open(cat_path, 'rb') as f:
            self.cat_encoder = pickle.load(f)

        self.loaded = True
        print(f"[PricePredictionPipeline] Loaded Ridge model + encoders from {models_dir}")

    def _safe_encode(self, encoder, value):
        """Encode a label, falling back to 0 if unseen."""
        try:
            return int(encoder.transform([value])[0])
        except (ValueError, KeyError):
            return 0

    def predict(self, brand, category, rating, retail_price, discount_percentage):
        """
        Predict fair price for a product.

        Returns dict with:
          predicted_fair_price, expected_discount, confidence_score
        """
        if not self.loaded:
            return {
                'predicted_fair_price': retail_price * (1 - discount_percentage / 100),
                'expected_discount': discount_percentage,
                'confidence_score': 0.0,
                'error': 'Model not loaded. Run train_models.py first.',
            }

        brand_enc = self._safe_encode(self.brand_encoder, brand)
        cat_enc = self._safe_encode(self.cat_encoder, category)
        rating = max(1.0, min(5.0, float(rating)))
        discount_pct = max(0.0, min(100.0, float(discount_percentage)))

        features = np.array([[retail_price, brand_enc, cat_enc, rating, discount_pct]])
        features_scaled = self.scaler.transform(features)
        predicted_price = float(self.model.predict(features_scaled)[0])
        predicted_price = max(0, predicted_price)

        expected_discount = 0.0
        if retail_price > 0:
            expected_discount = round((1 - predicted_price / retail_price) * 100, 1)
            expected_discount = max(0, min(100, expected_discount))

        # Confidence based on whether brand/category are known
        brand_known = brand in (self.brand_encoder.classes_ if self.brand_encoder else [])
        cat_known = category in (self.cat_encoder.classes_ if self.cat_encoder else [])
        base_confidence = 0.85
        if not brand_known:
            base_confidence -= 0.15
        if not cat_known:
            base_confidence -= 0.20
        # Price range sanity
        if predicted_price > retail_price * 1.5 or predicted_price < retail_price * 0.05:
            base_confidence -= 0.20

        return {
            'predicted_fair_price': round(predicted_price, 2),
            'expected_discount': expected_discount,
            'confidence_score': round(max(0, min(1, base_confidence)), 2),
        }
