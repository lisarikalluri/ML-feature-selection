import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold

diabetes_data = pd.read_csv('diabetes.csv')

X_data = diabetes_data.drop(columns=['Outcome'])
y_data = diabetes_data['Outcome']

variance_selector = VarianceThreshold(threshold=0.01)
X_data_low_variance = variance_selector.fit_transform(X_data)

variance_mask = variance_selector.get_support()
selected_features = X_data.loc[:, variance_mask]
filtered_diabetes_data = pd.concat([selected_features, y_data.reset_index(drop=True)], axis=1)

X_filtered = filtered_diabetes_data.drop(columns=['Outcome'])
y_filtered = filtered_diabetes_data['Outcome']

X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42)

logistic_model_filtered = LogisticRegression(max_iter=200)
logistic_model_filtered.fit(X_train_filtered, y_train_filtered)

y_pred_filtered = logistic_model_filtered.predict(X_test_filtered)
accuracy_filtered = accuracy_score(y_test_filtered, y_pred_filtered)

print(f'Accuracy after Low Variance Filter (Diabetes): {accuracy_filtered:.4f}')
