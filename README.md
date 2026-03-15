
# 🧠 Product Price Prediction using Machine Learning

This project builds a **Machine Learning system to predict product prices** based on product attributes and historical data.  
The project demonstrates the **complete ML pipeline**, including data preprocessing, feature engineering, model training, and deployment through a web dashboard.

The goal of this project is to simulate a **real-world data science workflow** where raw data is cleaned, transformed, used for training multiple models, and then integrated into an interactive application.

---

# 📌 Project Objectives

- Perform **data collection and exploration**
- Clean and preprocess raw e-commerce product data
- Implement machine learning models including:
  - Ridge Regression
  - Random Forest Regression
  - XGBoost Regression
- Build **ML models from scratch** for educational understanding
- Train and save models for production use
- Develop a **Flask-based web dashboard** for predictions

---

# 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Jupyter Notebook
- Flask
- HTML/CSS

---

# 📁 Project Structure

The project is organized into modular directories to separate **research, data, and production code**.

```

product-price-prediction-using-ml/
│
├── notebooks/                          # Jupyter Notebooks for ML experimentation
│   ├── 01_Data_Collection_and_Loading.ipynb
│   ├── 02_Data_Cleaning_and_Preprocessing.ipynb
│   ├── Ridge_Regression_From_Scratch.ipynb
│   └── XGBoost_Regression_From_Scratch.ipynb
│
├── data/                               # Data storage
│   ├── raw/                            # Original untouched dataset
│   └── processed/                      # Cleaned and processed dataset
│
├── models/                             # Saved model artifacts
│   └── label_encoders.pkl              # Encoders for categorical features
│
├── src/                                # Production scripts
│   └── train_models.py                 # Script for training ML models
│
├── reports/                            # Visualizations and reports
│   └── figures/                        # Generated charts and graphs
│
├── dashboard/                          # Web application
│   ├── app.py                          # Flask backend
│   └── templates/                      # HTML frontend
│
└── README.md                           # Project documentation

````

---

# 📊 Machine Learning Workflow

The project follows a structured **ML pipeline**:

1. Data Collection
2. Data Exploration
3. Data Cleaning
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Model Deployment

---

# 🚀 How to Run the Project

Follow these steps to set up the environment and run the project.

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Vighnesh7711/Product_price_prediction.git
cd Product_price_prediction
````

---

## 2️⃣ Install Required Libraries

Make sure Python **3.8+** is installed.

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Explore the Data (Optional)

You can explore the data and see the full ML pipeline using Jupyter notebooks.

```bash
jupyter notebook notebooks/
```

Run the notebooks in order:

### Notebook 1

```
01_Data_Collection_and_Loading.ipynb
```

Tasks performed:

* Load dataset
* Initial exploration
* Feature inspection
* Missing value analysis

### Notebook 2

```
02_Data_Cleaning_and_Preprocessing.ipynb
```

Tasks performed:

* Data cleaning
* Feature encoding
* Dataset preparation for modeling

---

## 4️⃣ Train Machine Learning Models

To train the models and generate production-ready artifacts:

```bash
python src/train_models.py
```

Models trained include:

* Ridge Regression
* Random Forest Regression
* XGBoost Regression

The trained models and encoders are stored in the **models/** directory.

---

## 5️⃣ Run the Web Dashboard

To interact with the ML model via a browser:

```bash
cd dashboard
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

You can input product features and get predicted prices.

---

# 📈 Example Output

The system predicts the expected product price based on input attributes such as:

* Product category
* Brand
* Specifications
* Historical trends

The dashboard then returns a **predicted market price**.

---

# 📊 Visualizations

The project generates several insights including:

* Price distribution analysis
* Outlier detection
* Category-based price trends
* Feature importance

These visualizations are stored in:

```
reports/figures/
```

---

# 🔬 Learning Outcomes

Through this project, the following concepts were implemented:

* End-to-end machine learning pipeline
* Feature engineering techniques
* Regression algorithms
* Model evaluation
* Flask-based ML deployment
* Real-world ML project structuring

---

# 🧠 Future Improvements

Possible improvements for the project:

* Deploy the model using **Docker**
* Host the dashboard on **Cloud (AWS / Render / Heroku)**
* Add **real-time product scraping**
* Implement **Deep Learning price prediction**
* Build a **REST API for predictions**

---



