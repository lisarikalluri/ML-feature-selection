import pandas as pd import numpy as np
from sklearn.model_selection import train_test_split from sklearn.tree import DecisionTreeClassifier
diabetes_dataset = pd.read_csv('diabetes.csv') X_features = diabetes_dataset.drop(columns=['Outcome'])

y_target = diabetes_dataset['Outcome']

X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X_features, y_target, test_size=0.3, random_state=42)

diabetes_model_backward = DecisionTreeClassifier() diabetes_model_backward.fit(X_train_diabetes, y_train_diabetes)

importances_diabetes = diabetes_model_backward.feature_importances_ indices_diabetes = np.argsort(importances_diabetes)

remaining_features = list(X_features.columns)
print(f"Initial Number of Features: {len(remaining_features)}")

while len(remaining_features) > 5: least_important_feature_index = indices_diabetes[0]
feature_to_drop = remaining_features[least_important_feature_index] X_features = X_features.drop(columns=[feature_to_drop]) remaining_features.remove(feature_to_drop)

X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X_features, y_target, test_size=0.3, random_state=42)
diabetes_model_backward.fit(X_train_diabetes, y_train_diabetes) importances_diabetes = diabetes_model_backward.feature_importances_ indices_diabetes = np.argsort(importances_diabetes)

print(f"Final Number of Features After Elimination:
{len(remaining_features)}")

importances_diabetes_sorted_indices = np.argsort(importances_diabetes)[::-1]

print("Top Important Features (Diabetes):")
for i in range(min(5, len(remaining_features))): feature_name =
remaining_features[importances_diabetes_sorted_indices[i]] importance_value =
importances_diabetes[importances_diabetes_sorted_indices[i]] print(f"{i + 1}. Feature '{feature_name}': {importance_value:.4f}")
