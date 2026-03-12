<<<<<<< HEAD
# 🛒 Product Price Prediction Using ML

**Predicting Flipkart product discounted prices using Machine Learning**

---

## 📁 Project Structure

```
ML_project/
│
├── 📓 notebooks/                          # Jupyter Notebooks (all ML work)
│   ├── 01_Data_Collection_and_Loading.ipynb    # Week 1 | Unit I
│   ├── 02_Data_Cleaning_and_Preprocessing.ipynb # Week 1-2 | Unit I
│   ├── Ridge_Regression_From_Scratch.ipynb      # Ridge Regression implementation
│   └── XGBoost_Regression_From_Scratch.ipynb    # XGBoost from scratch (pure NumPy)
│
├── 📂 data/
│   ├── raw/                               # Original untouched data
│   │   └── flipkart_com_ecommerce_sample.csv
│   └── processed/                         # Cleaned / transformed data
│       ├── 01_raw_snapshot.csv
│       └── 02_cleaned_data.csv
│
├── 📂 models/                             # Trained model artifacts
│   └── label_encoders.pkl
│
├── 📂 src/                                # Python scripts
│   └── train_models.py                    # Train RF, Ridge, XGBoost & save models
│
├── 📂 reports/                            # Analysis outputs
│   └── figures/
│       └── outlier_analysis.png
│
├── 📂 docs/                               # Reference notes & documentation
│   └── project_notes.txt                  # Notebook code reference / planning notes
│
└── README.md                              # This file
```

## 🚀 How to Run

### 1. Train Models
```bash
python src/train_models.py
```

### 2. Run Notebooks
```bash
jupyter notebook notebooks/
```

## 📊 Pipeline

| Step | Notebook / Script | Description |
|------|-------------------|-------------|
| 1 | `01_Data_Collection_and_Loading.ipynb` | Load raw Flipkart CSV, initial exploration |
| 2 | `02_Data_Cleaning_and_Preprocessing.ipynb` | Missing values, duplicates, price cleaning, outliers, encoding |
| 3 | `Ridge_Regression_From_Scratch.ipynb` | Ridge Regression from scratch on the dataset |
| 4 | `XGBoost_Regression_From_Scratch.ipynb` | XGBoost from scratch (pure NumPy, no libraries) |
| 5 | `src/train_models.py` | Train RF, Decision Tree, Ridge, XGBoost & save artifacts |

## 🛠️ Tech Stack
- Python, Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (LabelEncoder, Ridge, RandomForest, DecisionTree)
- XGBoost
=======
# 🛒 Product Price Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-red.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A comprehensive end-to-end machine learning project** that predicts product prices based on features like brand, category, ratings, reviews, and discounts — using data from e-commerce platforms (Flipkart/Amazon/Kaggle).

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Folder Structure](#-project-folder-structure)
- [Dataset](#-dataset)
- [Installation & Setup](#-installation--setup)
- [Notebook Pipeline](#-notebook-pipeline)
- [Models Implemented](#-models-implemented)
- [Evaluation Metrics](#-evaluation-metrics)
- [5-Week Execution Plan](#-5-week-execution-plan)
- [Web Application (Optional)](#-web-application-optional)
- [Results](#-results)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project tackles the problem of **predicting product selling prices** on e-commerce platforms. By leveraging product metadata — such as brand reputation, category, customer ratings, review volume, and discount patterns — we train and compare multiple regression models to find the best predictor.

The entire pipeline covers:

1. **Data Collection & Loading**
2. **Data Cleaning & Preprocessing**
3. **Exploratory Data Analysis (EDA)**
4. **Feature Engineering**
5. **Statistical Analysis & Hypothesis Testing**
6. **Data Normalization & Train-Test Splitting**
7. **Baseline Regression Models**
8. **Tree-Based Models (Decision Tree, Random Forest)**
9. **Gradient Boosting Models (XGBoost, LightGBM)**
10. **Model Comparison & Final Selection**
11. **Explainable AI with SHAP** *(Optional)*
12. **Final Pipeline & Deployment Export**

---

## ✨ Key Features

- 📊 **Comprehensive EDA** with rich visualizations (histograms, box plots, heatmaps)
- 🔧 **Feature Engineering** — derived metrics like `discount_percentage`, `price_per_rating`, `review_score`
- 📈 **Statistical Hypothesis Testing** — ANOVA, Pearson correlation, p-value-based feature selection
- 🤖 **7+ Regression Models** compared side-by-side
- 🌳 **Hyperparameter Tuning** via GridSearchCV
- 🧠 **Explainable AI** with SHAP (TreeExplainer, summary & force plots)
- 🌐 **Optional Web App** (Flask/Streamlit) for live predictions
- 📄 **Full Documentation** & exam-friendly 5-week schedule

---

## 📁 Project Folder Structure

```
product-price-prediction-using-ml/
│
├── 📓 notebooks/                                # All ML work (Jupyter)
│   ├── 01_Data_Collection_and_Loading.ipynb
│   ├── 02_Data_Cleaning_and_Preprocessing.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Statistical_Analysis.ipynb
│   ├── 06_Data_Normalization_and_Splitting.ipynb
│   ├── 07_Baseline_Regression_Models.ipynb
│   ├── 08_Decision_Tree_and_Random_Forest.ipynb
│   ├── 09_XGBoost_and_LightGBM.ipynb
│   ├── 10_Model_Comparison_and_Selection.ipynb
│   ├── 11_Explainable_AI_with_SHAP.ipynb        # Optional (Extra Marks)
│   └── 12_Final_Pipeline_and_Export.ipynb
│
├── 📊 data/
│   ├── raw/
│   │   └── product_data.csv
│   ├── processed/
│   │   ├── cleaned_data.csv
│   │   ├── engineered_features.csv
│   │   └── train_test_split/
│   │       ├── X_train.csv
│   │       ├── X_test.csv
│   │       ├── y_train.csv
│   │       └── y_test.csv
│   └── results/
│       ├── model_comparison.csv
│       ├── feature_importance.csv
│       ├── shap_values.csv
│       └── statistical_tests.csv
│
├── 🤖 models/
│   ├── baseline/
│   │   ├── linear_regression.pkl
│   │   ├── ridge.pkl
│   │   └── knn.pkl
│   ├── tree/
│   │   ├── decision_tree.pkl
│   │   └── random_forest.pkl
│   ├── boosting/
│   │   ├── xgboost.pkl
│   │   └── lightgbm.pkl
│   ├── preprocessing/
│   │   └── scaler.pkl
│   └── final/
│       └── price_prediction_model.pkl ⭐
│
├── 🌐 webapp/                                   # Optional (Flask / Streamlit)
│   ├── app.py
│   ├── templates/
│   │   ├── index.html
│   │   ├── input_form.html
│   │   └── result.html
│   └── static/
│       ├── css/
│       └── js/
│
├── 📄 docs/
│   ├── introduction.md
│   ├── literature_review.md
│   ├── methodology.md
│   ├── results.md
│   ├── conclusion.md
│   └── final_report.pdf
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Dataset

### Source
Data sourced from **Flipkart / Amazon product listings** or public **Kaggle datasets**.

### Raw Dataset Columns (`product_data.csv`)

| Column           | Description                          | Type        |
|------------------|--------------------------------------|-------------|
| `product_name`   | Name of the product                  | String      |
| `brand`          | Brand / manufacturer                 | Categorical |
| `category`       | Product category                     | Categorical |
| `rating`         | Average customer rating (1–5)        | Float       |
| `reviews_count`  | Number of customer reviews           | Integer     |
| `discount`       | Discount offered (%)                 | Float       |
| `actual_price`   | Original listed price (MRP)          | Float       |
| **`price`**      | **Selling price (🎯 Target Variable)** | **Float**   |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook / JupyterLab

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/product-price-prediction-using-ml.git
cd product-price-prediction-using-ml

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook
```

### `requirements.txt`

```
pandas>=1.4.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
shap>=0.40.0
scipy>=1.7.0
joblib>=1.1.0
flask>=2.0.0          # Optional (for web app)
streamlit>=1.10.0     # Optional (for web app)
```

---

## 📓 Notebook Pipeline

Each notebook is self-contained and follows a sequential workflow:

### `01` — Data Collection and Loading
> **Week 1 | Unit I**

- Import `pandas`, `numpy`
- Load `product_data.csv` (and optional `historical_price_data.csv`)
- Initial exploration: `.info()`, `.describe()`, `.head()`
- Identify the target variable (`price`)
- Check for missing values & duplicates
- Save raw snapshot to `data/processed/`

---

### `02` — Data Cleaning and Preprocessing
> **Week 1–2 | Unit I**

- Handle missing values (price, rating, reviews)
- Remove duplicate entries
- Clean currency symbols (`₹`, commas) and convert to numeric
- Outlier detection using **IQR method** on price
- Encode categorical columns (`brand`, `category`)
- **Output:** `cleaned_data.csv`

---

### `03` — Exploratory Data Analysis
> **Week 2 | Unit II**

- 📊 Price distribution — **Histogram**
- 📊 Brand vs Average Price — **Bar plot**
- 📊 Category vs Price — **Box plot**
- 📊 Rating vs Price — **Scatter plot**
- 📊 Feature correlations — **Heatmap**
- Identify key price-driving features
- **Output:** Saved EDA plots

---

### `04` — Feature Engineering
> **Week 3 | Unit I**

- Create derived features:
  - `discount_percentage = (actual_price - price) / actual_price * 100`
  - `price_per_rating = price / rating`
  - `review_score = rating × reviews_count`
- Brand-level aggregation (average price per brand)
- Category-level aggregation
- Apply **OneHotEncoding / LabelEncoding**
- **Output:** `engineered_features.csv`

---

### `05` — Statistical Analysis
> **Week 3 | Unit II**

- Descriptive statistics
- **Pearson correlation** analysis
- **ANOVA test** — brand/category vs price
- Hypothesis testing: *Does brand significantly affect price?*
- Feature selection based on **p < 0.05**
- **Output:** `statistical_tests.csv`

---

### `06` — Data Normalization and Splitting
> **Week 3 | Unit I**

- Apply **StandardScaler / MinMaxScaler**
- Train-test split (**80% train – 20% test**)
- **Output:**
  - `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
  - `scaler.pkl`

---

### `07` — Baseline Regression Models
> **Week 4 | Unit III**

- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **k-Nearest Neighbors (KNN) Regressor**
- Evaluate using MAE, RMSE, R² Score
- **Output:** `baseline_models.pkl`, `model_comparison.csv`

---

### `08` — Decision Tree and Random Forest
> **Week 4–5 | Unit IV**

- **DecisionTreeRegressor**
- **RandomForestRegressor**
- Hyperparameter tuning with **GridSearchCV**
- Feature importance analysis
- **Output:** `tree_models.pkl`

---

### `09` — XGBoost and LightGBM
> **Week 5 | Unit IV**

- **XGBoostRegressor**
- **LightGBMRegressor**
- Compare performance & training time
- Feature importance visualization
- **Output:** `boosting_models.pkl`

---

### `10` — Model Comparison and Selection
> **Week 5 | Unit III**

- Compare **all** regression models side-by-side
- Rank by **RMSE** and **R² Score**
- Select the **best-performing model**
- **Output:** `final_model.pkl` ⭐

---

### `11` — Explainable AI with SHAP *(Optional — Extra Marks)*

- SHAP `TreeExplainer` for tree/boosting models
- **SHAP summary plot** — global feature importance
- **SHAP force plot** — single prediction interpretation
- Interpret the top price-driving factors

---

### `12` — Final Pipeline and Export

- Load saved scaler + best model
- Build `predict_price_pipeline()` function
- Test with sample product inputs
- Export all deployment-ready files

---

## 🤖 Models Implemented

| #  | Model                    | Type              | Category  |
|----|--------------------------|-------------------|-----------|
| 1  | Linear Regression        | Linear            | Baseline  |
| 2  | Ridge Regression         | Regularized       | Baseline  |
| 3  | Lasso Regression         | Regularized       | Baseline  |
| 4  | KNN Regressor            | Instance-based    | Baseline  |
| 5  | Decision Tree Regressor  | Tree-based        | Advanced  |
| 6  | Random Forest Regressor  | Ensemble (Bagging)| Advanced  |
| 7  | XGBoost Regressor        | Ensemble (Boosting)| Advanced |
| 8  | LightGBM Regressor       | Ensemble (Boosting)| Advanced |

---

## 📏 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | Mean Absolute Error | Average absolute difference between predicted and actual prices |
| **RMSE** | Root Mean Squared Error | Penalizes larger errors more heavily |
| **R² Score** | Coefficient of Determination | Proportion of variance explained (1.0 = perfect) |

---

## 🗓️ 5-Week Execution Plan

> Designed to be **exam-friendly** and follow a structured academic timeline.

| Week | Tasks Completed | Notebooks |
|------|----------------|-----------|
| **Week 1** | Data collection + cleaning | `01`, `02` |
| **Week 2** | Exploratory Data Analysis + insights | `03` |
| **Week 3** | Feature engineering + statistical analysis + data splitting | `04`, `05`, `06` |
| **Week 4** | ML models (baseline + tree-based) + hyperparameter tuning | `07`, `08` |
| **Week 5** | Boosting models + final model selection + report + demo | `09`, `10`, `11`, `12` |

---

## 🌐 Web Application (Optional)

A simple web interface for real-time price predictions.

### Flask Version

```bash
cd webapp
python app.py
# Open http://localhost:5000
```

### Streamlit Version

```bash
cd webapp
streamlit run app.py
# Opens automatically in browser
```

### How It Works

1. **Input:** User enters product details (brand, category, rating, reviews, discount)
2. **Processing:** Loads `scaler.pkl` + `price_prediction_model.pkl`
3. **Output:** Displays the **predicted selling price**

---

## 📈 Results

> *(Sample results — update with your actual values)*

| Model               | MAE (₹) | RMSE (₹) | R² Score |
|---------------------|----------|-----------|----------|
| Linear Regression   | 1,250    | 1,890     | 0.72     |
| Ridge Regression    | 1,230    | 1,870     | 0.73     |
| Lasso Regression    | 1,260    | 1,900     | 0.71     |
| KNN Regressor       | 1,100    | 1,680     | 0.78     |
| Decision Tree       | 980      | 1,520     | 0.82     |
| Random Forest       | 750      | 1,180     | 0.89     |
| **XGBoost**         | **620**  | **980**   | **0.93** |
| LightGBM            | 650      | 1,020     | 0.92     |

> 🏆 **Best Model: XGBoost Regressor** with R² = 0.93

---

## 📄 Documentation

Detailed documentation is available in the `docs/` folder:

| Document | Description |
|----------|-------------|
| `introduction.md` | Problem statement and objectives |
| `literature_review.md` | Related work and references |
| `methodology.md` | Data pipeline and model approach |
| `results.md` | Findings, charts, and model comparison |
| `conclusion.md` | Summary, limitations, and future work |
| `final_report.pdf` | Complete compiled report |

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Kaggle](https://www.kaggle.com/) — for open e-commerce datasets
- [scikit-learn](https://scikit-learn.org/) — ML library
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosting framework
- [SHAP](https://shap.readthedocs.io/) — model interpretability
- [Flipkart](https://www.flipkart.com/) / [Amazon](https://www.amazon.in/) — data inspiration

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ for Machine Learning

</div>
>>>>>>> e7441de82177cfd8c490a2e53917f5864c0ff634
