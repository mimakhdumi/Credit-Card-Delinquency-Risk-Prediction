import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Delinquency_prediction_dataset.csv')

# Step 1: Identify the features with missing values
missing_features = ['Credit_Score', 'Income', 'Loan_Balance']

# Step 2: Select all numerical features for imputation context
# KNN works better when it can consider all related numerical features
numerical_features = ['Age', 'Credit_Score', 'Income', 'Credit_Utilization', 
                     'Missed_Payments', 'Loan_Balance', 'Debt_to_Income_Ratio', 
                     'Account_Tenure']

# Step 3: Extract numerical data for imputation
data_for_imputation = df[numerical_features].copy()

# Step 4: Standardize the data before KNN imputation
# This is crucial because KNN uses distance metrics
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_imputation)

# Step 5: Apply KNN Imputation
# n_neighbors=5 is a good starting point; adjust based on your dataset size
knn_imputer = KNNImputer(n_neighbors=5)
data_imputed_scaled = knn_imputer.fit_transform(data_scaled)

# Step 6: Transform back to original scale
data_imputed = scaler.inverse_transform(data_imputed_scaled)

# Step 7: Replace the original columns with imputed values
df_imputed = df.copy()
df_imputed[numerical_features] = data_imputed

# Step 8: Verify no missing values remain in target features
print("Missing values after imputation:")
print(df_imputed[missing_features].isnull().sum())

# Step 9: Optional - Add missing value indicators for model transparency
# These binary flags can be useful features for your decision tree model
for feature in missing_features:
    df_imputed[f'{feature}_was_missing'] = df[feature].isnull().astype(int)

print("\nImputation completed successfully!")
print(f"Dataset shape: {df_imputed.shape}")