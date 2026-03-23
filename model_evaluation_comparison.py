"""
Model Evaluation and Comparison
===============================
Comprehensive evaluation of trained ML models (Ridge Regression, XGBoost, Random Forest, Decision Tree)
showing detailed statistics, processing times, and performance metrics.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import time
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "flipkart_com-ecommerce_sample.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

print("="*80)
print("COMPREHENSIVE MODEL EVALUATION & COMPARISON")
print("="*80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ─── 1. Data Preparation (Same as training) ───────────────────────────────────
print("\nLoading and preparing dataset...")

def extract_category(cat_tree: str) -> str:
    """Pull the top-level category from the nested tree string."""
    import re
    if pd.isna(cat_tree):
        return "Unknown"
    m = re.search(r'"([^">>]+)', str(cat_tree))
    return m.group(1).strip() if m else "Unknown"

def clean_price(val) -> float:
    import re
    if pd.isna(val):
        return np.nan
    s = re.sub(r"[^\d.]", "", str(val))
    try:
        return float(s)
    except ValueError:
        return np.nan

def clean_rating(val) -> float:
    if pd.isna(val):
        return 3.0
    try:
        return float(str(val).strip())
    except ValueError:
        return 3.0

# Load and clean data
df = pd.read_csv(DATA_CSV, usecols=[
    "product_name", "product_category_tree",
    "retail_price", "discounted_price",
    "product_rating", "overall_rating", "brand"
])

df["main_category"] = df["product_category_tree"].apply(extract_category)
df["retail_price"] = df["retail_price"].apply(clean_price)
df["discounted_price"] = df["discounted_price"].apply(clean_price)
df["rating"] = df["product_rating"].apply(clean_rating)
df["brand_clean"] = df["brand"].fillna("Unknown").str.strip()

# Derived features
df["discount_pct"] = np.where(
    df["retail_price"] > 0,
    ((df["retail_price"] - df["discounted_price"]) / df["retail_price"] * 100).clip(0, 100),
    0
)

# Clean data
df = df.dropna(subset=["retail_price", "discounted_price"])
df = df[(df["retail_price"] > 0) & (df["discounted_price"] > 0)]
df = df[df["discounted_price"] <= df["retail_price"]]
df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]

# Filter top brands and categories
TOP_N_BRANDS = 60
TOP_N_CATS = 30

top_brands = df["brand_clean"].value_counts().head(TOP_N_BRANDS).index.tolist()
top_cats = df["main_category"].value_counts().head(TOP_N_CATS).index.tolist()

df = df[
    df["brand_clean"].isin(top_brands) &
    df["main_category"].isin(top_cats)
].copy()

# Encode categoricals
brand_enc = LabelEncoder()
cat_enc = LabelEncoder()

df["brand_enc"] = brand_enc.fit_transform(df["brand_clean"])
df["cat_enc"] = cat_enc.fit_transform(df["main_category"])

FEATURE_COLS = ["retail_price", "brand_enc", "cat_enc", "rating", "discount_pct"]
TARGET_COL = "discounted_price"

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale for Ridge
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"Dataset prepared: {len(X_train):,} train samples, {len(X_test):,} test samples")

# ─── 2. Load Trained Models ──────────────────────────────────────────────────
print("\nLoading trained models...")

try:
    rf_model = pickle.load(open(f"{MODELS_DIR}/rf_model.pkl", "rb"))
    dt_model = pickle.load(open(f"{MODELS_DIR}/dt_model.pkl", "rb"))
    ridge_data = pickle.load(open(f"{MODELS_DIR}/ridge_model.pkl", "rb"))
    ridge_model = ridge_data["model"]
    xgb_model = pickle.load(open(f"{MODELS_DIR}/xgb_model.pkl", "rb"))
    print("All models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# ─── 3. Model Evaluation Function ─────────────────────────────────────────────
def comprehensive_evaluation(model, X_test_data, y_true, model_name, use_scaling=False):
    """Comprehensive evaluation with detailed statistics and timing."""

    print(f"\nEvaluating {model_name}...")
    print("-" * 60)

    # Prediction timing
    start_time = time.time()
    if use_scaling:
        y_pred = model.predict(X_test_data)
    else:
        y_pred = model.predict(X_test_data)
    end_time = time.time()

    processing_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Performance metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Accuracy calculation (percentage of predictions within acceptable range)
    # Define accuracy as predictions within 10% of actual values
    accuracy_threshold = 0.1  # 10%
    within_threshold = np.abs((y_true - y_pred) / y_true) <= accuracy_threshold
    accuracy = np.mean(within_threshold) * 100

    # Prediction statistics
    pred_stats = {
        'count': len(y_pred),
        'mean': np.mean(y_pred),
        'std': np.std(y_pred),
        'min': np.min(y_pred),
        'q1': np.percentile(y_pred, 25),
        'q2': np.percentile(y_pred, 50),  # median
        'q3': np.percentile(y_pred, 75),
        'max': np.max(y_pred)
    }

    # Actual values statistics
    actual_stats = {
        'count': len(y_true),
        'mean': np.mean(y_true),
        'std': np.std(y_true),
        'min': np.min(y_true),
        'q1': np.percentile(y_true, 25),
        'q2': np.percentile(y_true, 50),  # median
        'q3': np.percentile(y_true, 75),
        'max': np.max(y_true)
    }

    # Display results
    print(f"Model: {model_name}")
    print(f"Processing Time: {processing_time:.2f} ms")
    print(f"Samples Processed: {len(y_pred):,}")

    print(f"\nPERFORMANCE METRICS:")
    print(f"  R² Score:              {r2:.4f}")
    print(f"  RMSE:                  Rs.{rmse:,.2f}")
    print(f"  MAE:                   Rs.{mae:.2f}")
    print(f"  Median Absolute Error: Rs.{medae:.2f}")
    print(f"  MAPE:                  {mape:.2f}%")
    print(f"  Accuracy (±10%):       {accuracy:.2f}%")

    print(f"\nPREDICTIONS STATISTICS:")
    print(f"  Count:    {pred_stats['count']:,}")
    print(f"  Mean:     Rs.{pred_stats['mean']:,.2f}")
    print(f"  Std:      Rs.{pred_stats['std']:,.2f}")
    print(f"  Min:      Rs.{pred_stats['min']:,.2f}")
    print(f"  Q1 (25%): Rs.{pred_stats['q1']:,.2f}")
    print(f"  Q2 (50%): Rs.{pred_stats['q2']:,.2f}")
    print(f"  Q3 (75%): Rs.{pred_stats['q3']:,.2f}")
    print(f"  Max:      Rs.{pred_stats['max']:,.2f}")

    print(f"\nACTUAL VALUES STATISTICS:")
    print(f"  Count:    {actual_stats['count']:,}")
    print(f"  Mean:     Rs.{actual_stats['mean']:,.2f}")
    print(f"  Std:      Rs.{actual_stats['std']:,.2f}")
    print(f"  Min:      Rs.{actual_stats['min']:,.2f}")
    print(f"  Q1 (25%): Rs.{actual_stats['q1']:,.2f}")
    print(f"  Q2 (50%): Rs.{actual_stats['q2']:,.2f}")
    print(f"  Q3 (75%): Rs.{actual_stats['q3']:,.2f}")
    print(f"  Max:      Rs.{actual_stats['max']:,.2f}")

    return {
        'model_name': model_name,
        'processing_time_ms': processing_time,
        'performance_metrics': {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'medae': medae,
            'mape': mape,
            'accuracy': accuracy
        },
        'predictions_stats': pred_stats,
        'actual_stats': actual_stats,
        'predictions': y_pred,
        'actuals': y_true
    }

# ─── 4. Evaluate All Models ───────────────────────────────────────────────────
results = {}

# Random Forest
results['Random Forest'] = comprehensive_evaluation(
    rf_model, X_test, y_test, "Random Forest"
)

# Decision Tree
results['Decision Tree'] = comprehensive_evaluation(
    dt_model, X_test, y_test, "Decision Tree"
)

# Ridge Regression
results['Ridge Regression'] = comprehensive_evaluation(
    ridge_model, X_test_sc, y_test, "Ridge Regression", use_scaling=True
)

# XGBoost
results['XGBoost'] = comprehensive_evaluation(
    xgb_model, X_test, y_test, "XGBoost"
)

# ─── 5. Comparison Summary ────────────────────────────────────────────────────
print("\n" + "="*90)
print("MODEL COMPARISON SUMMARY")
print("="*90)

# Create comparison DataFrame
comparison_data = []
for model_name, result in results.items():
    comparison_data.append({
        'Model': model_name,
        'Processing Time (ms)': f"{result['processing_time_ms']:.2f}",
        'R² Score': f"{result['performance_metrics']['r2']:.4f}",
        'RMSE (Rs.)': f"{result['performance_metrics']['rmse']:,.0f}",
        'MAE (Rs.)': f"{result['performance_metrics']['mae']:.2f}",
        'MAPE (%)': f"{result['performance_metrics']['mape']:.2f}",
        'Accuracy (±10%)': f"{result['performance_metrics']['accuracy']:.2f}%",
        'Pred Mean (Rs.)': f"{result['predictions_stats']['mean']:,.2f}",
        'Pred Std (Rs.)': f"{result['predictions_stats']['std']:,.2f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# ─── 6. Best Model Identification ────────────────────────────────────────────
print("\nBEST MODEL ANALYSIS:")
print("-" * 50)

best_r2 = max(results.keys(), key=lambda x: results[x]['performance_metrics']['r2'])
best_rmse = min(results.keys(), key=lambda x: results[x]['performance_metrics']['rmse'])
best_mae = min(results.keys(), key=lambda x: results[x]['performance_metrics']['mae'])
best_accuracy = max(results.keys(), key=lambda x: results[x]['performance_metrics']['accuracy'])
fastest = min(results.keys(), key=lambda x: results[x]['processing_time_ms'])

print(f"Best R² Score:     {best_r2} ({results[best_r2]['performance_metrics']['r2']:.4f})")
print(f"Lowest RMSE:       {best_rmse} (Rs.{results[best_rmse]['performance_metrics']['rmse']:,.0f})")
print(f"Lowest MAE:        {best_mae} (Rs.{results[best_mae]['performance_metrics']['mae']:.2f})")
print(f"Best Accuracy:     {best_accuracy} ({results[best_accuracy]['performance_metrics']['accuracy']:.2f}%)")
print(f"Fastest Processing: {fastest} ({results[fastest]['processing_time_ms']:.2f} ms)")

# ─── 7. Advanced Statistics Comparison ───────────────────────────────────────
print("\nDETAILED STATISTICS COMPARISON:")
print("=" * 100)

stats_comparison_data = []
for model_name, result in results.items():
    pred_stats = result['predictions_stats']
    stats_comparison_data.append({
        'Model': model_name,
        'Count': f"{pred_stats['count']:,}",
        'Mean (Rs.)': f"{pred_stats['mean']:,.2f}",
        'Std (Rs.)': f"{pred_stats['std']:,.2f}",
        'Min (Rs.)': f"{pred_stats['min']:,.2f}",
        'Q1 (Rs.)': f"{pred_stats['q1']:,.2f}",
        'Median (Rs.)': f"{pred_stats['q2']:,.2f}",
        'Q3 (Rs.)': f"{pred_stats['q3']:,.2f}",
        'Max (Rs.)': f"{pred_stats['max']:,.2f}"
    })

stats_df = pd.DataFrame(stats_comparison_data)
print(stats_df.to_string(index=False))

# Display actual values for reference
actual_stats = results['Random Forest']['actual_stats']  # Same for all models
print(f"\nACTUAL VALUES (Reference):")
print(f"Count: {actual_stats['count']:,}, Mean: Rs.{actual_stats['mean']:,.2f}, "
      f"Std: Rs.{actual_stats['std']:,.2f}, Min: Rs.{actual_stats['min']:,.2f}, "
      f"Q1: Rs.{actual_stats['q1']:,.2f}, Median: Rs.{actual_stats['q2']:,.2f}, "
      f"Q3: Rs.{actual_stats['q3']:,.2f}, Max: Rs.{actual_stats['max']:,.2f}")

# ─── 8. Save Results ──────────────────────────────────────────────────────────
print("\nSaving evaluation results...")

# Prepare results for JSON export (remove numpy arrays)
export_results = {}
for model_name, result in results.items():
    export_results[model_name] = {
        'processing_time_ms': result['processing_time_ms'],
        'performance_metrics': result['performance_metrics'],
        'predictions_stats': result['predictions_stats'],
        'actual_stats': result['actual_stats']
    }

# Save to JSON
with open(f"{DATA_DIR}/model_evaluation_results.json", "w") as f:
    json.dump(export_results, f, indent=2)

# Save comparison tables to CSV
comparison_df.to_csv(f"{DATA_DIR}/model_comparison_summary.csv", index=False)
stats_df.to_csv(f"{DATA_DIR}/model_statistics_comparison.csv", index=False)

print(f"Results saved to:")
print(f"  - {DATA_DIR}/model_evaluation_results.json")
print(f"  - {DATA_DIR}/model_comparison_summary.csv")
print(f"  - {DATA_DIR}/model_statistics_comparison.csv")

print("\n" + "="*90)
print("MODEL EVALUATION COMPLETE!")
print("="*90)
print(f"{len(results)} models evaluated on {len(y_test):,} test samples")
print(f"Comprehensive statistics and performance metrics calculated")
print(f"Processing times measured for all models")
print(f"Results saved for future reference")
print("="*90)