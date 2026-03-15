📁 Project Structure
The project is organized into modular directories to separate data, notebooks, and production code:

Plaintext
product-price-prediction-using-ml/
│
├── 📓 notebooks/                          # Jupyter Notebooks for all ML research
│   ├── 01_Data_Collection_and_Loading.ipynb    # Data ingestion and exploration
│   ├── 02_Data_Cleaning_and_Preprocessing.ipynb # Cleaning and encoding tasks
│   ├── Ridge_Regression_From_Scratch.ipynb      # Manual Ridge implementation
│   └── XGBoost_Regression_From_Scratch.ipynb    # Manual XGBoost implementation
│
├── 📂 data/                               # Data storage
│   ├── raw/                               # Original untouched product data
│   └── processed/                         # Cleaned snapshots for modeling
│
├── 📂 models/                             # Saved model artifacts and encoders
│   └── label_encoders.pkl                 # Categorical encoding mappings
│
├── 📂 src/                                # Production Python scripts
│   └── train_models.py                    # Script to train and save final models
│
├── 📂 reports/                            # Visualizations and analysis
│   └── figures/                           # Charts like outlier analysis
│
├── 📂 dashboard/                          # Web application files
│   ├── app.py                             # Main Flask application
│   └── templates/                         # HTML frontend files
│
└── README.md                              # Project documentation
🚀 How to Run the Program
Follow these steps to set up the environment and execute the project:

1. Setup Environment
Ensure you have Python 3.8+ installed, then install the required dependencies:

Bash
# Clone the repository
git clone https://github.com/your-username/product-price-prediction.git
cd product-price-prediction

# Install libraries
pip install -r requirements.txt
2. Data Preparation and Exploration
If you want to view the step-by-step development process, run the Jupyter Notebooks:

Bash
jupyter notebook notebooks/
Step 1: Run 01_Data_Collection_and_Loading.ipynb to explore the raw data.

Step 2: Run 02_Data_Cleaning_and_Preprocessing.ipynb to generate the cleaned_data.csv and encoders.

3. Train the Models
To train the Random Forest, Ridge, and XGBoost models and save them as production-ready artifacts, execute the training script:

Bash
python src/train_models.py
4. Run the Web Dashboard (Optional)
To interact with the model via a web browser, start the Flask application:

Bash
cd dashboard
python app.py
