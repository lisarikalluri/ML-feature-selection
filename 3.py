import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

diabetes_data = pd.read_csv('diabetes.csv')
X_data = diabetes_data.drop(columns=['Outcome'])
y_label = diabetes_data['Outcome']

corr_matrix = diabetes_data.corr()

high_corr_pairs = corr_matrix[corr_matrix.abs() > 0.8].stack().index.tolist()
drop_features = set()

for col1, col2 in high_corr_pairs:
    if col1 != col2:
        drop_features.add(col2)

filtered_diabetes_data = diabetes_data.drop(columns=drop_features)
X_filtered = filtered_diabetes_data.drop(columns=['Outcome'])
y_filtered = filtered_diabetes_data['Outcome']

X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42)

logistic_model_filtered = LogisticRegression(max_iter=200)
logistic_model_filtered.fit(X_train_filtered, y_train_filtered)

y_pred_filtered = logistic_model_filtered.predict(X_test_filtered)
accuracy_filtered = accuracy_score(y_test_filtered, y_pred_filtered)

print(f'Accuracy after High Correlation Filter (Diabetes): {accuracy_filtered:.4f}')
