# Customer Churn Prediction: Final Report

## 1. Summary

I built a machine learning classification model to identify telecom customers who are at high risk of canceling their service. Using a tuned LightGBM model trained on customer demographic, contract, and usage features, the system predicts churn risk with strong overall ranking ability. The evaluation on the unseen test set achieved an ROC-AUC of FILL_IN and a PR-AUC of FILL_IN, catching FILL_IN percent of churners at the chosen risk threshold. Analysis shows that short contract duration, high monthly charges, and lack of tech support or security add-ons are the primary indicators of customer churn.

## 2. Business problem

The telecom company loses recurring monthly revenue whenever an existing customer cancels their service. Acquiring a brand new customer is significantly more expensive than retaining an existing one through proactive discount offers or contract incentives. This model supports the customer retention team by providing a daily risk score for every subscriber, allowing them to target high-risk accounts with retention campaigns before the customer decides to leave.

## 3. What I did

Data cleaning: Loaded the raw dataset of 7043 records, converted total charges from text to numeric format, handled missing values caused by brand new customers with zero tenure, and consolidated redundant categories across service columns.

Feature engineering: Created new domain-specific features including count of subscribed add-ons, indicators for family plans and automatic payment methods, average monthly historical charge, charge difference relative to current bill, and tenure duration groups.

Model comparison: Evaluated Logistic Regression, Random Forest, XGBoost, and LightGBM using 5-fold cross-validation on the training set. Tuned hyper-parameters and class weights to address the 73 to 27 class imbalance, selecting LightGBM as the best performer.

Leakage prevention: Performed all feature scaling and encoding within cross-validation folds after splitting the data into 80 percent training and 20 percent test sets. Fitted scalers exclusively on training data and selected decision thresholds using out-of-fold validation before a single final evaluation on the test set.

## 4. Results

Metrics summary:

* ROC-AUC (test): FILL_IN
* PR-AUC (test): FILL_IN
* Recall (churn): FILL_IN
* Precision (churn): FILL_IN
* Selected threshold: FILL_IN

Compared to a baseline approach of taking no action or making random offers, using this model allows the company to focus retention budgets on the subset of customers responsible for the vast majority of churn risk. Based on financial assumptions around retention offer costs and customer lifetime value, targeted outreach at the chosen decision threshold minimizes total business loss compared to unguided campaign strategies.

## 5. What drives churn

1. Month-to-month contracts: Customers without long-term contract commitments can cancel at any time with zero termination penalty.
2. Short tenure duration: Customers in their first few months of service are substantially more likely to leave before building long-term loyalty.
3. Fiber optic internet with high monthly bills: High monthly cost creates price sensitivity, especially when bundled with premium internet services.
4. Electronic check payment method: Customers paying manually via electronic checks churn at higher rates than those using automatic bank or credit card billing.
5. Lack of tech support or online security add-ons: Subscribers without supporting services report higher frustration during service issues and switch providers more frequently.

## 6. Recommendations to test

1. Offer a small monthly discount to month-to-month subscribers who switch to a one-year contract, targeting customers in their first 6 months of service.
2. Promote automatic payment options like direct bank transfer or credit card billing by offering a one-time bill credit upon enrollment.
3. Provide a free 3-month trial of tech support and online security services to new fiber optic subscribers to improve early service satisfaction.

## 7. Limitations

* Single dataset constraint: The findings are based on a static sample of 7043 customers from one company, so results may not generalize directly to other telecom providers or regions.
* Non-causal explanations: Feature importance and SHAP values show which variables the model relies on to make predictions, but they do not prove that changing a feature will directly prevent a customer from leaving.
* Uncalibrated risk scores: Because class weights were used to handle class imbalance, the output values represent relative risk scores for ranking rather than true underlying probabilities.
* Random split evaluation: The dataset lacked historical timestamp information, requiring a stratified random split rather than a more realistic time-based train and test split.

## 8. What I learned

This project highlighted why standard accuracy is misleading for imbalanced datasets and demonstrated how metrics like ROC-AUC and PR-AUC provide a clearer picture of performance. I gained hands-on experience structuring a leak-free pipeline by fitting scalers strictly on training data and tuning decision thresholds using out-of-fold predictions. I also learned how tree-based gradient boosting models can outperform linear models on complex tabular data while remaining fully interpretable using SHAP analysis.