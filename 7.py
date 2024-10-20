import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score

# Load the dataset
diabetes_data = pd.read_csv('diabetes.csv')

# Separate features and target
features_diabetes = diabetes_data.drop(columns=['Outcome'])
target_diabetes = diabetes_data['Outcome']

# Apply RFECV for feature selection
feature_selector_rfecv = RFECV(estimator=LogisticRegression(max_iter=200), step=1, cv=5)
feature_selector_rfecv.fit(features_diabetes, target_diabetes)

# Transform the features based on RFECV selection
selected_features_diabetes = feature_selector_rfecv.transform(features_diabetes)

# Split the data into training and testing sets
X_train_rfecv, X_test_rfecv, y_train_rfecv, y_test_rfecv = train_test_split(selected_features_diabetes, target_diabetes, test_size=0.3, random_state=42)

# Train the model using the selected features
logistic_model_rfecv = LogisticRegression(max_iter=200)
logistic_model_rfecv.fit(X_train_rfecv, y_train_rfecv)

# Predict and calculate accuracy
y_pred_rfecv = logistic_model_rfecv.predict(X_test_rfecv)
accuracy_rfecv = accuracy_score(y_test_rfecv, y_pred_rfecv)

# Print the accuracy
print(f'Accuracy after Recursive Feature Elimination (Diabetes): {accuracy_rfecv:.4f}')
