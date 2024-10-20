import pandas as pd
from sklearn.model_selection import train_test_split from sklearn.ensemble import RandomForestClassifier

diabetes_dataset = pd.read_csv('diabetes.csv')

X_features = diabetes_dataset.drop(columns=['Outcome']) 
y_target = diabetes_dataset['Outcome']

X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X_features, y_target, test_size=0.3, random_state=42)

rf_model_diabetes = RandomForestClassifier() rf_model_diabetes.fit(X_train_diabetes, y_train_diabetes)

feature_importances_diabetes = rf_model_diabetes.feature_importances_

importance_df_diabetes = pd.DataFrame({'Feature': X_features.columns, 'Importance': feature_importances_diabetes})

importance_df_diabetes = importance_df_diabetes.sort_values(by='Importance', ascending=False)

print("Feature Importance Scores (Diabetes):") print(importance_df_diabetes)
