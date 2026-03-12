"""
train_models.py
===============
Train Random Forest, Ridge Regression, and XGBoost on the Flipkart e-commerce dataset
and save models + metadata for the Flask web app.

Run this script ONCE before starting the Flask app:
    python train_models.py
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_CSV   = "flipkart_com-ecommerce_sample.csv"
MODELS_DIR = "models"
DATA_DIR   = "data"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)


# ─── 1. Load & Clean ──────────────────────────────────────────────────────────
print("📂 Loading dataset …")
df = pd.read_csv(DATA_CSV, usecols=[
    "product_name", "product_category_tree",
    "retail_price", "discounted_price",
    "product_rating", "overall_rating", "brand"
])
print(f"   Raw rows : {len(df):,}")


def extract_category(cat_tree: str) -> str:
    """Pull the top-level category from the nested tree string."""
    if pd.isna(cat_tree):
        return "Unknown"
    m = re.search(r'"([^">>]+)', str(cat_tree))
    return m.group(1).strip() if m else "Unknown"


def clean_price(val) -> float:
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


df["main_category"]    = df["product_category_tree"].apply(extract_category)
df["retail_price"]     = df["retail_price"].apply(clean_price)
df["discounted_price"] = df["discounted_price"].apply(clean_price)
df["rating"]           = df["product_rating"].apply(clean_rating)
df["brand_clean"]      = df["brand"].fillna("Unknown").str.strip()

# Derived feature: discount percentage
df["discount_pct"] = np.where(
    df["retail_price"] > 0,
    ((df["retail_price"] - df["discounted_price"]) / df["retail_price"] * 100).clip(0, 100),
    0
)

# Drop bad rows
df = df.dropna(subset=["retail_price", "discounted_price"])
df = df[(df["retail_price"] > 0) & (df["discounted_price"] > 0)]
df = df[df["discounted_price"] <= df["retail_price"]]
df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]
print(f"   Clean rows: {len(df):,}")


# ─── 2. Top Brands & Categories ───────────────────────────────────────────────
TOP_N_BRANDS = 60
TOP_N_CATS   = 30

top_brands = (
    df["brand_clean"].value_counts().head(TOP_N_BRANDS).index.tolist()
)
top_cats = (
    df["main_category"].value_counts().head(TOP_N_CATS).index.tolist()
)

df = df[
    df["brand_clean"].isin(top_brands) &
    df["main_category"].isin(top_cats)
].copy()
print(f"   After top-brand/cat filter: {len(df):,}")


# ─── 3. Encode Categoricals ───────────────────────────────────────────────────
brand_enc = LabelEncoder()
cat_enc   = LabelEncoder()

df["brand_enc"] = brand_enc.fit_transform(df["brand_clean"])
df["cat_enc"]   = cat_enc.fit_transform(df["main_category"])

FEATURE_COLS = ["retail_price", "brand_enc", "cat_enc", "rating", "discount_pct"]
TARGET_COL   = "discounted_price"

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# ─── 4. Scale (for Ridge) ─────────────────────────────────────────────────────
scaler        = StandardScaler()
X_train_sc    = scaler.fit_transform(X_train)
X_test_sc     = scaler.transform(X_test)


# ─── 5. Train Models ──────────────────────────────────────────────────────────
metrics = {}


def evaluate(name: str, y_true, y_pred) -> dict:
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"   {name:22s}  R²={r2:.4f}  RMSE=₹{rmse:,.0f}  MAE=₹{mae:,.0f}")
    return {"r2": round(r2, 4), "rmse": round(rmse, 2), "mae": round(mae, 2)}


# — Random Forest —
print("\n🌲 Training Random Forest …")
rf = RandomForestRegressor(
    n_estimators=150, max_depth=12,
    min_samples_leaf=3, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
metrics["random_forest"] = evaluate("Random Forest", y_test, rf.predict(X_test))


# — Decision Tree Regressor —
print("\n🌳 Training Decision Tree Regressor …")
dt = DecisionTreeRegressor(
    max_depth=10, min_samples_leaf=5, random_state=42
)
dt.fit(X_train, y_train)
metrics["decision_tree"] = evaluate("Decision Tree", y_test, dt.predict(X_test))


# — Ridge Regression —
print("\n📐 Training Ridge Regression …")
ridge = Ridge(alpha=10.0)
ridge.fit(X_train_sc, y_train)
metrics["ridge"] = evaluate("Ridge Regression", y_test, ridge.predict(X_test_sc))


# — XGBoost —
print("\n⚡ Training XGBoost …")
try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, verbosity=0
    )
    xgb.fit(X_train, y_train)
    metrics["xgboost"] = evaluate("XGBoost", y_test, xgb.predict(X_test))
    xgb_obj = xgb
except ImportError:
    print("   ⚠  xgboost not installed — using RandomForest as fallback")
    xgb_obj = rf
    metrics["xgboost"] = metrics["random_forest"]


# ─── 6. Save Everything ───────────────────────────────────────────────────────
print("\n💾 Saving models …")

pickle.dump(rf,      open(f"{MODELS_DIR}/rf_model.pkl",  "wb"))
pickle.dump(dt,      open(f"{MODELS_DIR}/dt_model.pkl",  "wb"))
pickle.dump({"model": ridge, "scaler": scaler},
            open(f"{MODELS_DIR}/ridge_model.pkl", "wb"))
pickle.dump(xgb_obj, open(f"{MODELS_DIR}/xgb_model.pkl", "wb"))
pickle.dump(brand_enc, open(f"{MODELS_DIR}/brand_encoder.pkl", "wb"))
pickle.dump(cat_enc,   open(f"{MODELS_DIR}/cat_encoder.pkl",   "wb"))

json.dump(sorted(top_brands), open(f"{DATA_DIR}/brands.json",     "w"))
json.dump(sorted(top_cats),   open(f"{DATA_DIR}/categories.json", "w"))

# Dataset stats for Dashboard
stats = {
    "total_products":   int(len(df)),
    "total_brands":     int(len(top_brands)),
    "total_categories": int(len(top_cats)),
    "avg_discount":     round(float(df["discount_pct"].mean()), 1),
    "avg_price":        round(float(df["discounted_price"].mean()), 2),
    "min_price":        round(float(df["discounted_price"].min()), 2),
    "max_price":        round(float(df["discounted_price"].max()), 2),
    "avg_rating":       round(float(df["rating"].mean()), 2),
    "top_brands":       top_brands[:15],
    "top_categories":   top_cats[:10],
    "feature_cols":     FEATURE_COLS,
    "target_col":       TARGET_COL,
    "train_size":       int(len(X_train)),
    "test_size":        int(len(X_test)),
}
json.dump(stats,   open(f"{DATA_DIR}/stats.json",   "w"), indent=2)
json.dump(metrics, open(f"{DATA_DIR}/metrics.json", "w"), indent=2)

print("\n✅ All models and metadata saved successfully!")
print(f"   Brands    : {len(top_brands)}")
print(f"   Categories: {len(top_cats)}")
print(f"   Total data: {len(df):,} products")
print("\nModel Performance Summary:")
for model_name, m in metrics.items():
    print(f"  {model_name:25s}  R²={m['r2']:.4f}")
print("\n🚀 Now run:  python app.py")
