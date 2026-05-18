# Customer Churn Prediction

A machine learning project based on the IBM Telco Customer Churn dataset.

The goal of this project was to predict whether a customer is likely to leave the service or not using classification models.

## What I Did

- Cleaned and preprocessed the dataset
- Handled datatype issues in `TotalCharges`
- Converted categorical features using:
  - Manual binary encoding
  - One-Hot Encoding
- Performed train-test split
- Applied feature scaling using `StandardScaler`
- Trained multiple classification models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Evaluated models using:
  - Precision
  - Recall
  - F1-score

## Handling Imbalanced Data

The dataset had moderate class imbalance between churned and non-churned customers.

I experimented with SMOTE (Synthetic Minority Oversampling Technique) to balance the training data and compared model performance before and after oversampling.

### Observation

After applying SMOTE:
- Recall improved slightly in some cases
- Precision dropped
- Overall F1-score became worse

This showed that SMOTE is not always beneficial for moderately imbalanced datasets and can sometimes reduce overall model generalization.

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)

## Key Learnings

- Proper preprocessing workflow
- Data leakage prevention
- Difference between Precision and Recall
- Importance of evaluation metrics in business problems
- Practical use and limitations of SMOTE
- Comparing multiple ML models on the same dataset