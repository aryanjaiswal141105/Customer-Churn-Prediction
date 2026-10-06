# Customer Churn Prediction

Predicting which telecom customers are likely to leave, using machine learning.

## 1. Problem Statement
A telecom company loses revenue when customers cancel their service.
Finding a new customer costs much more than keeping an existing one.
This project builds a model that predicts **which customers are likely to churn**,
so the company can contact them with retention offers before they leave.

## 2. Objective
Build a binary classification model that predicts whether a customer will churn
(`Churn = Yes`) using customer account, service, and billing information.

## 3. Definition of Churn
- **Churn = Yes**: the customer left the company
- **Churn = No**: the customer stayed
- **Target column**: `Churn`

## 4. Dataset
- Name: Telco Customer Churn (IBM sample dataset)
- Size: 7,043 customers, 21 columns
- Source: Kaggle
- Class balance: about 73% stayed, 27% churned (imbalanced)

## 5. Success Metrics
Model Evaluation Metrics
Recall: Catches as many real churners as possible so the business does not lose them.
Precision: Prevents wasting expensive retention offers on loyal customers who intend to stay.
F1 Score: Provides a single score that balances both Recall and Precision.
ROC AUC: Measures how well the model tells the difference between a churner and a loyal customer.

Accuracy is **not** the main metric because the data is imbalanced.

## 6. Tools
Python, pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, SHAP, Streamlit

## 7. Project Status
- [x] Phase 1: Setup
- [x] Phase 2: Problem definition
- [x] Phase 3: Data understanding (EDA)
- [x] Phase 4: Data preparation
- [x] Phase 5: Feature engineering
- [x] Phase 6: Modeling
- [x] Phase 7: Evaluation
- [ ] Phase 8: Explainability
- [ ] Phase 9: Productionizing
- [ ] Phase 10: Documentation

## 8. Business Questions (My Answer)

1. Who will use this model's Predictions?

- Customer Success, Marketing, and Sales teams will use these predictions to identify at-risk accounts before they cancel their subscriptions.

2. What action will they take for a high-risk customer ?

- They will take proactive retention measures, such as offering targeted discounts, personalized contract upgrades, or direct customer service outreach to resolve their issues.

3. Which is worse:missing a churner, or wrongly flagging a loyal customer? Why?

- Missing a churner is worse because losing a customer results in a direct, permanent loss of revenue, and acquiring a replacement customer is up to 5 to 25 times more expensive than the small cost of offering an unnecessary promotion to a loyal customer.