# Customer Churn Prediction

Predicting which telecom customers are likely to leave, using machine learning, with an interactive Streamlit app.

## 1. Problem

A telecom company loses revenue when customers cancel. Keeping a customer costs less than finding a new one. This project predicts which customers are likely to churn so the company can contact them with retention offers in time.

* Task: binary classification (Churn = Yes or No)
* Main metrics: recall and precision on the churn class, ROC-AUC, PR-AUC (accuracy is not the main metric because only about 27% of customers churn)

## 2. Data

* Dataset: Telco Customer Churn (IBM sample data), from Kaggle
* Size: 7,043 customers, 21 columns
* Class balance: about 73% stayed, 27% churned
* Data issue found: TotalCharges was stored as text and had 11 blanks (all new customers with tenure = 0), filled with 0

The dataset is not modified. The raw file is in data/raw/.

## 3. Method

* EDA: Looked at churn by contract, tenure, charges, and so on (01_eda.ipynb)
* Preprocessing: Fixed types, encoded text, split 80/20 (stratified), scaled using train only (02_preprocessing.ipynb)
* Feature engineering: Added num_addons, has_family, is_autopay, avg_charge, charge_diff, tenure_group (03_feature_engineering.ipynb)
* Modeling: Compared Logistic Regression, Random Forest, XGBoost, LightGBM with 5-fold CV, tuned the best (04_modeling.ipynb)
* Evaluation: Tested once on the test set, chose the threshold using train data only (05_evaluation.ipynb)
* Explainability: SHAP for global and per-customer explanations (06_explainability.ipynb)
* Production: Reusable code in src/, tests, Streamlit app (src/, tests/, app/)

Leakage prevention: split before scaling, scaler fitted on train only, class weights instead of SMOTE, threshold chosen on out-of-fold train predictions, test set used once.

## 4. Results

Chosen model: FILL_IN (feature set: FILL_IN)

Metrics on test set:

* Recall (churn): FILL_IN
* Precision (churn): FILL_IN
* F1: FILL_IN
* F2: FILL_IN
* ROC-AUC: FILL_IN (cross-validation: FILL_IN)
* PR-AUC: FILL_IN (random guessing would score about 0.27)

Did the new features help? FILL_IN

Top churn drivers (SHAP):

1. FILL_IN
2. FILL_IN
3. FILL_IN
4. FILL_IN
5. FILL_IN

## 5. How to run

1. Clone the repository:
git clone [https://github.com/aryanjaiswal141105/Customer-Churn-Prediction.git](https://github.com/aryanjaiswal141105/Customer-Churn-Prediction.git)
cd Customer-Churn-Prediction
2. Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate
3. Install libraries:
pip install -r requirements.txt
4. Add the dataset:
Download from Kaggle, rename it to telco_churn.csv, and place it in data/raw/
5. Run the tests:
python -m pytest tests -v
6. Launch the app:
streamlit run app/app.py

To rebuild the model from the raw data:
python -m src.train

## 6. Project structure

customer-churn-prediction/

* data/ : raw and processed data
* notebooks/ : 01 to 06: EDA, preprocessing, features, modeling, evaluation, SHAP
* src/ : reusable code (config, preprocessing, predict, train)
* models/ : saved model, scaler, columns, metrics
* reports/ : figures and final report
* app/ : Streamlit app
* tests/ : automatic tests
* requirements.txt
* README.md

## 7. Limitations

* One dataset of 7,043 customers from one company, so results may not transfer.
* The model's score is a risk score for ranking, not an exact probability, because class weights shift the outputs upward.
* SHAP shows what the model uses, not what causes churn. Retention ideas must be tested (for example with an A/B test) before being trusted.
* The data has no dates, so I used a random split, not a time-based one.

## 8. Future work

* Test retention offers with an A/B experiment
* Calibrate the probabilities
* Try a time-based split with real company data
* Deploy the app online

## 9. Tools

Python, pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, LightGBM, SHAP, Streamlit, pytest