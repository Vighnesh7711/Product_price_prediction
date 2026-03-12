"""
price_predictor.py
==================
Predict future product prices using ML models built FROM SCRATCH:
  - Random Forest        (sklearn — as baseline)
  - Decision Tree        (sklearn — as baseline)
  - Ridge Regression     (FROM SCRATCH — closed-form w = (X'X + αI)⁻¹ X'y)
  - XGBoost              (FROM SCRATCH — gradient-boosted CART trees, pure NumPy)

Trains all 4 models on the dashboard's historical category price data,
evaluates them (R², MAE, RMSE), picks the best model for prediction.
"""

import numpy as np
import json
import os
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# ═════════════════════════════════════════════════════════════════
#   FROM-SCRATCH IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════

# ── Standard Scaler (from scratch) ─────────────────────────────
class StandardScalerScratch:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1  # avoid division by zero
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ── Ridge Regression (from scratch) ───────────────────────────
class RidgeRegressionScratch:
    """
    Ridge Regression from scratch using NumPy.
    Uses closed-form analytical solution:  w = (X'X + αI)⁻¹ X'y
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None
        self.scaler = StandardScalerScratch()

    def _add_bias(self, X):
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        Xb = self._add_bias(X_scaled)
        n_features = Xb.shape[1]
        # Regularization matrix: don't regularize bias term
        reg_matrix = self.alpha * np.eye(n_features)
        reg_matrix[0, 0] = 0
        w = np.linalg.solve(Xb.T @ Xb + reg_matrix, Xb.T @ y)
        self.bias = w[0]
        self.weights = w[1:]
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return X_scaled @ self.weights + self.bias


# ── XGBoost Node (from scratch) ───────────────────────────────
class _XGBNode:
    """A single node in an XGBoost regression tree."""
    __slots__ = ('is_leaf', 'weight', 'feature', 'threshold', 'left', 'right')

    def __init__(self):
        self.is_leaf   = True
        self.weight    = 0.0
        self.feature   = None
        self.threshold = None
        self.left      = None
        self.right     = None


# ── XGBoost Tree (from scratch) ───────────────────────────────
class _XGBTree:
    """
    A single regression CART tree used inside XGBoost.
    Leaf weight:  w* = -G / (H + λ)
    Split Gain:   0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ
    """

    def __init__(self, max_depth=6, min_child_weight=1.0,
                 reg_lambda=1.0, reg_alpha=0.0, gamma=0.0):
        self.max_depth        = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda       = reg_lambda
        self.reg_alpha        = reg_alpha
        self.gamma            = gamma
        self.root             = None

    def _leaf_weight(self, g, h):
        G, H = g.sum(), h.sum()
        if self.reg_alpha > 0:
            sign_G = np.sign(G)
            G = sign_G * max(abs(G) - self.reg_alpha, 0.0)
        return -G / (H + self.reg_lambda)

    def _score(self, g, h):
        G, H = g.sum(), h.sum()
        if self.reg_alpha > 0:
            G = np.sign(G) * max(abs(G) - self.reg_alpha, 0.0)
        return (G ** 2) / (H + self.reg_lambda)

    def _best_split(self, X, g, h, features):
        best_gain  = -np.inf
        best_feat  = None
        best_thr   = None
        score_node = self._score(g, h)

        for feat in features:
            x_col = X[:, feat]
            thresholds = np.unique(x_col)
            if len(thresholds) < 2:
                continue
            # Use midpoints between unique sorted values
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2.0

            for thr in thresholds:
                left_mask  = x_col <= thr
                right_mask = ~left_mask

                if (h[left_mask].sum()  < self.min_child_weight or
                    h[right_mask].sum() < self.min_child_weight):
                    continue

                gain = 0.5 * (
                    self._score(g[left_mask],  h[left_mask]) +
                    self._score(g[right_mask], h[right_mask]) -
                    score_node
                ) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr  = thr

        return best_feat, best_thr, best_gain

    def _build(self, X, g, h, depth, features):
        node = _XGBNode()
        node.weight = self._leaf_weight(g, h)

        if depth >= self.max_depth or len(g) < 2:
            node.is_leaf = True
            return node

        feat, thr, gain = self._best_split(X, g, h, features)

        if feat is None or gain <= 0:
            node.is_leaf = True
            return node

        node.is_leaf   = False
        node.feature   = feat
        node.threshold = thr

        left_mask  = X[:, feat] <= thr
        right_mask = ~left_mask

        node.left  = self._build(X[left_mask],  g[left_mask],  h[left_mask],  depth + 1, features)
        node.right = self._build(X[right_mask], g[right_mask], h[right_mask], depth + 1, features)
        return node

    def fit(self, X, g, h, col_indices):
        self.root = self._build(X, g, h, depth=0, features=col_indices)
        return self

    def _predict_row(self, node, x):
        if node.is_leaf:
            return node.weight
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X):
        return np.array([self._predict_row(self.root, row) for row in X])


# ── XGBoost Regressor (from scratch) ─────────────────────────
class XGBoostRegressorScratch:
    """
    XGBoost Regressor — From Scratch (Pure NumPy).
    Gradient-Boosted Decision Trees using Taylor expansion of loss.
    """

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
                 reg_alpha=0.0, min_child_weight=1.0, gamma=0.0,
                 random_state=42):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.learning_rate     = learning_rate
        self.subsample         = subsample
        self.colsample_bytree  = colsample_bytree
        self.reg_lambda        = reg_lambda
        self.reg_alpha         = reg_alpha
        self.min_child_weight  = min_child_weight
        self.gamma             = gamma
        self.random_state      = random_state

        self._trees       = []
        self._base_score  = 0.0
        self.train_loss_  = []
        self.n_features_in_ = 0

    def _gradients(self, y, F):
        g = F - y           # first derivative of MSE
        h = np.ones_like(y) # second derivative = 1 for MSE
        return g, h

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        rng = np.random.RandomState(self.random_state)
        self._base_score = float(np.mean(y))
        F = np.full(n_samples, self._base_score)
        self._trees = []
        self.train_loss_ = []

        n_cols = max(1, int(self.colsample_bytree * n_features))
        n_rows = max(1, int(self.subsample * n_samples))

        for m in range(self.n_estimators):
            g, h = self._gradients(y, F)
            row_idx = rng.choice(n_samples, size=n_rows, replace=False)
            col_idx = rng.choice(n_features, size=n_cols, replace=False)

            tree = _XGBTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                gamma=self.gamma,
            )
            tree.fit(X[row_idx], g[row_idx], h[row_idx], col_indices=col_idx)
            self._trees.append((tree, col_idx))
            F += self.learning_rate * tree.predict(X)
            self.train_loss_.append(float(np.mean((y - F) ** 2)))

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        F = np.full(X.shape[0], self._base_score)
        for tree, _ in self._trees:
            F += self.learning_rate * tree.predict(X)
        return F


# ═════════════════════════════════════════════════════════════════
#   METRICS (from scratch)
# ═════════════════════════════════════════════════════════════════

def _r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0

def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ═════════════════════════════════════════════════════════════════
#   PRICE PREDICTOR
# ═════════════════════════════════════════════════════════════════

class PricePredictor:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.category_trends = {}
        self.seasonal_data = {}
        self.config = {}

        # ML models per category  {cat: {model_name: fitted_model}}
        self.trained_models = {}
        # Performance scores      {cat: {model_name: {r2, mae, rmse}}}
        self.model_scores = {}
        # Best model name per category
        self.best_model_name = {}
        # Overall (global) performance summary
        self.global_scores = {}

        self._load()
        self._train_all_models()

    # ── data loading ──────────────────────────────────────────────
    def _load(self):
        def _json(name):
            p = os.path.join(self.models_dir, name)
            if os.path.exists(p):
                with open(p) as f:
                    return json.load(f)
            return {}

        self.category_trends = _json('dashboard_category_trends.json')
        self.seasonal_data = _json('dashboard_seasonal.json')
        self.config = _json('dashboard_config.json')

    # ── feature engineering ───────────────────────────────────────
    @staticmethod
    def _build_features(prices, window=7):
        """
        Given a list of daily prices, build features for each time-step:
          - lag_1 … lag_7
          - rolling_mean_7, rolling_std_7
          - day_index, day_of_week proxy, month proxy
        Target = price at step t.
        """
        arr = np.array([p for p in prices if p is not None], dtype=np.float64)
        if len(arr) < window + 5:
            return None, None

        X, y = [], []
        for i in range(window, len(arr)):
            feats = list(arr[i - window:i])                # lag_1 … lag_window
            feats.append(np.mean(arr[i - window:i]))       # rolling mean
            feats.append(np.std(arr[i - window:i]))        # rolling std
            feats.append(float(i))                         # day index
            feats.append(float(i % 7))                     # day of week proxy
            feats.append(float((i // 30) % 12))            # month proxy
            X.append(feats)
            y.append(arr[i])
        return np.array(X), np.array(y)

    # ── train all ML models on every category ─────────────────────
    def _train_all_models(self):
        if not self.category_trends:
            return

        model_names = ['RandomForest', 'DecisionTree', 'Ridge', 'XGBoost']

        # Accumulate global predictions for overall scoring
        global_y_true = {m: [] for m in model_names}
        global_y_pred = {m: [] for m in model_names}

        for cat, trend in self.category_trends.items():
            median_prices = trend.get('median_prices', [])
            X, y = self._build_features(median_prices)
            if X is None or len(X) < 20:
                continue

            # Manual train/test split (80/20, no shuffle for time-series)
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            models = {
                'RandomForest': RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                ),
                'DecisionTree': DecisionTreeRegressor(
                    max_depth=10, random_state=42
                ),
                'Ridge': RidgeRegressionScratch(alpha=1.0),
                'XGBoost': XGBoostRegressorScratch(
                    n_estimators=50, max_depth=4, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=42
                ),
            }

            fitted = {}
            scores = {}
            best_r2 = -999
            best_name = 'Ridge'

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                r2  = _r2_score(y_test, preds)
                mae = _mae(y_test, preds)
                rmse = _rmse(y_test, preds)

                fitted[name] = model
                scores[name] = {
                    'r2': round(r2, 4),
                    'mae': round(mae, 2),
                    'rmse': round(rmse, 2),
                }

                global_y_true[name].extend(y_test.tolist())
                global_y_pred[name].extend(preds.tolist())

                if r2 > best_r2:
                    best_r2 = r2
                    best_name = name

            self.trained_models[cat] = fitted
            self.model_scores[cat] = scores
            self.best_model_name[cat] = best_name

        # Compute global (overall) model performance
        for name in model_names:
            yt = np.array(global_y_true.get(name, []))
            yp = np.array(global_y_pred.get(name, []))
            if len(yt) > 0:
                self.global_scores[name] = {
                    'r2': round(float(_r2_score(yt, yp)), 4),
                    'mae': round(float(_mae(yt, yp)), 2),
                    'rmse': round(float(_rmse(yt, yp)), 2),
                }

        print(f"[PricePredictor] Trained models for {len(self.trained_models)} categories")
        if self.global_scores:
            best_global = max(self.global_scores, key=lambda m: self.global_scores[m]['r2'])
            print(f"[PricePredictor] Best overall model: {best_global} "
                  f"(R²={self.global_scores[best_global]['r2']})")
            for m, s in self.global_scores.items():
                print(f"  {m:20s}  R²={s['r2']:.4f}  MAE={s['mae']:.2f}  RMSE={s['rmse']:.2f}")

    # ── public API ────────────────────────────────────────────────
    def predict(self, current_price: float, category: str,
                days_ahead: int = 30) -> dict:
        trend = self.category_trends.get(category)
        if not trend:
            return self._fallback(current_price, days_ahead)

        median_prices = trend.get('median_prices', [])
        arr = [p for p in median_prices if p is not None]
        if len(arr) < 15:
            return self._fallback(current_price, days_ahead)

        # Use the best-scored model for this category
        cat_models = self.trained_models.get(category)
        if not cat_models:
            return self._fallback(current_price, days_ahead)

        best_name = self.best_model_name.get(category, 'Ridge')
        model = cat_models[best_name]

        # Scale factor: align the series values to the user's current price
        last_known = arr[-1]
        scale = current_price / last_known if last_known > 0 else 1.0

        # Generate future predictions step by step
        window = 7
        recent = list(arr[-window:])
        preds_raw = []

        for d in range(days_ahead):
            feats = list(recent[-window:])
            feats.append(np.mean(recent[-window:]))
            feats.append(np.std(recent[-window:]))
            feats.append(float(len(arr) + d))
            feats.append(float((len(arr) + d) % 7))
            feats.append(float(((len(arr) + d) // 30) % 12))

            pred_val = float(model.predict(np.array([feats]))[0])
            pred_val = max(0, pred_val)
            preds_raw.append(pred_val)
            recent.append(pred_val)

        # Scale predictions to match current price level
        preds = [round(p * scale, 2) for p in preds_raw]

        # Uncertainty via volatility
        vol = np.std(np.diff(arr[-60:])) if len(arr) > 60 else np.std(np.diff(arr))
        upper, lower = [], []
        for d_i, pp in enumerate(preds):
            unc = vol * np.sqrt(d_i + 1) * scale
            upper.append(round(pp + 2 * unc, 2))
            lower.append(round(max(0, pp - 2 * unc), 2))

        now = datetime.now()
        dates_out = [(now + timedelta(days=d + 1)).strftime('%Y-%m-%d')
                     for d in range(days_ahead)]

        wk = (preds[min(6, len(preds) - 1)] - current_price) / current_price * 100
        mo = (preds[-1] - current_price) / current_price * 100

        # Historical for chart
        hist_dates = trend['dates'][-90:]
        hscale = current_price / trend['avg_price'] if trend['avg_price'] > 0 else 1
        hist_prices = [
            round(p * hscale, 2) if p else None
            for p in trend['median_prices'][-90:]
        ]

        return {
            'predictions': preds,
            'dates': dates_out,
            'upper_bound': upper,
            'lower_bound': lower,
            'current_price': current_price,
            'predicted_7d': preds[min(6, len(preds) - 1)],
            'predicted_30d': preds[-1],
            'week_change_pct': round(wk, 2),
            'month_change_pct': round(mo, 2),
            'trend': 'up' if mo > 1 else ('down' if mo < -1 else 'stable'),
            'volatility': round(trend['volatility'], 2),
            'confidence': self._confidence(trend),
            'historical': {'dates': hist_dates, 'prices': hist_prices},
            'insights': self._insights(current_price, preds, category, trend),
            'model_used': best_name,
            'model_scores': self.model_scores.get(category, {}),
            'global_scores': self.global_scores,
        }

    def get_category_comparison(self, category: str) -> dict:
        if not self.category_trends:
            return {}
        result = {}
        cats = sorted(
            self.category_trends.items(),
            key=lambda kv: kv[1]['product_count'], reverse=True,
        )
        selected = [category] if category in self.category_trends else []
        for c, t in cats:
            if c != category and len(selected) < 5:
                selected.append(c)
        for c in selected:
            t = self.category_trends[c]
            result[c] = {
                'price_index': t['price_index'][-90:],
                'dates': t['dates'][-90:],
                'product_count': t['product_count'],
                'avg_price': t['avg_price'],
            }
        return result

    def get_performance_summary(self) -> dict:
        """Return overall model performance for the dashboard table."""
        return self.global_scores

    # ── private helpers ───────────────────────────────────────────
    def _seasonal_adj(self, category, date):
        s = self.seasonal_data.get(category, {})
        target_month = date.month
        for key, data in s.items():
            if int(key.split('-')[1]) == target_month:
                overall = np.mean([d['avg'] for d in s.values()])
                if overall > 0:
                    return (data['avg'] - overall) / overall * 0.1
        return 0

    @staticmethod
    def _confidence(trend):
        valid = [p for p in trend['median_prices'] if p is not None]
        if len(valid) < 30 or trend['volatility'] > 30:
            return 'low'
        return 'medium' if trend['volatility'] > 15 else 'high'

    def _insights(self, price, preds, category, trend):
        ins = []
        final = preds[-1]
        pct = (final / price - 1) * 100 if price else 0

        if pct > 5:
            ins.append({'type': 'warning', 'icon': '📈',
                        'text': f'Prices in {category} may INCREASE ~{pct:.1f}% this month.'})
            ins.append({'type': 'tip', 'icon': '💡',
                        'text': 'Consider buying now for the best deal.'})
        elif pct < -5:
            ins.append({'type': 'positive', 'icon': '📉',
                        'text': f'Prices in {category} may DROP ~{abs(pct):.1f}% this month.'})
            ins.append({'type': 'tip', 'icon': '⏳',
                        'text': 'Consider waiting for a better deal.'})
        else:
            ins.append({'type': 'neutral', 'icon': '📊',
                        'text': f'Prices in {category} are expected to stay STABLE.'})

        if trend['volatility'] > 20:
            ins.append({'type': 'warning', 'icon': '⚡',
                        'text': f'High volatility ({trend["volatility"]:.1f}%) in this category.'})

        ins.append({'type': 'info', 'icon': '📦',
                    'text': f'Analysis based on {trend["product_count"]:,} products.'})
        return ins

    def _fallback(self, price, days):
        preds, dates_out = [], []
        now = datetime.now()
        p = price
        for d in range(1, days + 1):
            p = max(0, p * 0.999 + np.random.normal(0, price * 0.005))
            preds.append(round(p, 2))
            dates_out.append((now + timedelta(days=d)).strftime('%Y-%m-%d'))
        wk = (preds[6] - price) / price * 100 if price else 0
        return {
            'predictions': preds, 'dates': dates_out,
            'upper_bound': [round(p * 1.1, 2) for p in preds],
            'lower_bound': [round(p * 0.9, 2) for p in preds],
            'current_price': price,
            'predicted_7d': preds[6], 'predicted_30d': preds[-1],
            'week_change_pct': round(wk, 2),
            'month_change_pct': round((preds[-1] - price) / price * 100, 2) if price else 0,
            'trend': 'stable', 'volatility': 0, 'confidence': 'low',
            'historical': {'dates': [], 'prices': []},
            'insights': [{'type': 'warning', 'icon': '⚠️',
                          'text': 'Limited data — predictions have low confidence.'}],
            'model_used': 'Fallback (random walk)',
            'model_scores': {},
            'global_scores': self.global_scores,
        }
