import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

diabetes_data = pd.read_csv('diabetes.csv')
missing_data_percentage = diabetes_data.isnull().mean()

filtered_data = diabetes_data.loc[:, missing_data_percentage < 0.3]

X_data = filtered_data.drop(columns=['Outcome'])
y_label = filtered_data['Outcome']

X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, y_label, test_size=0.3, random_state=42)

logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train_data, y_train_data)

y_predictions = logistic_model.predict(X_test_data)
accuracy_result = accuracy_score(y_test_data, y_predictions)

print(f'Accuracy after Missing Value Filter (Diabetes): {accuracy_result:.4f}')
