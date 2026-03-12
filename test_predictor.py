from ml.price_predictor import PricePredictor
import os

pp = PricePredictor(os.path.abspath('models'))
print("Global Scores:", pp.get_performance_summary())

# Test prediction
res = pp.predict(100.0, "Jewellery")
print("\nPrediction keys:", res.keys())
print("Model Used:", res.get('model_used'))
print("Global Scores in pred:", "Yes" if res.get('global_scores') else "No")
print("Model Scores in pred:", "Yes" if res.get('model_scores') else "No")
