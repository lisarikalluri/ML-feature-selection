import pandas as pd
from sklearn.model_selection import train_test_split from sklearn.ensemble import RandomForestRegressor from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
housing_data = pd.read_csv('melbourne_housing_raw.csv') missing_percentage = housing_data.isnull().mean() * 100
columns_to_remove = [column for column in missing_percentage.index if missing_percentage[column] > 20 and column != 'Price']
cleaned_data = housing_data.drop(columns=columns_to_remove) cleaned_data = cleaned_data.dropna(subset=['Price'])

X = cleaned_data.drop(columns=['Price', 'Date', 'Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname'])
y = cleaned_data['Price']

numeric_features = X.select_dtypes(include=[float, int]) X[numeric_features.columns] = numeric_features.fillna(numeric_features.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest_model = RandomForestRegressor(random_state=42) rfe_backward = RFE(estimator=random_forest_model, n_features_to_select=2, step=1)
rfe_backward.fit(X_train, y_train)

X_train_backward = rfe_backward.transform(X_train) X_test_backward = rfe_backward.transform(X_test)

def evaluate_model(train_features, test_features, train_target, test_target):
model = RandomForestRegressor(random_state=42) model.fit(train_features, train_target) predictions = model.predict(test_features)
mse = mean_squared_error(test_target, predictions) return mse

mse_backward = evaluate_model(X_train_backward, X_test_backward, y_train, y_test)
print(f"Model performance after backward feature elimination: MSE =
{mse_backward:.2f}")
