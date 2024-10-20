import pandas as pd
from sklearn.model_selection import train_test_split from sklearn.ensemble import RandomForestRegressor from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
housing_data = pd.read_csv('melbourne_housing_raw.csv')

missing_percentage = housing_data.isnull().mean() * 100 columns_to_remove = [column for column in missing_percentage.index if missing_percentage[column] > 20 and column != 'Price']
cleaned_data = housing_data.drop(columns=columns_to_remove) cleaned_data = cleaned_data.dropna(subset=['Price'])

X = cleaned_data.drop(columns=['Price', 'Date', 'Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname'])
y = cleaned_data['Price']

numeric_features = X.select_dtypes(include=[float, int]) X[numeric_features.columns] = numeric_features.fillna(numeric_features.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(random_state=42) rf_model.fit(X_train, y_train)
selector = SelectFromModel(rf_model, threshold="mean", prefit=True) X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

X_train_rf_selected = pd.DataFrame(X_train_selected, columns=X.columns[selector.get_support()]) X_test_rf_selected = pd.DataFrame(X_test_selected, columns=X.columns[selector.get_support()])

def evaluate_model(train_features, test_features, train_target, test_target):
model = RandomForestRegressor(random_state=42) model.fit(train_features, train_target) predictions = model.predict(test_features)
mse = mean_squared_error(test_target, predictions) return mse

mse_rf_selection = evaluate_model(X_train_rf_selected, X_test_rf_selected, y_train, y_test)
print(f"Model performance after random forest feature selection: MSE =
{mse_rf_selection:.2f}")
