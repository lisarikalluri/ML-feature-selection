import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

housing_df = pd.read_csv('melbourne_housing_raw.csv')
null_percentage = housing_df.isnull().mean() * 100
columns_to_drop = [col for col in null_percentage.index if null_percentage[col] > 20 and col != 'Price']
cleaned_housing_df = housing_df.drop(columns=columns_to_drop)
cleaned_housing_df = cleaned_housing_df.dropna(subset=['Price'])

housing_features = cleaned_housing_df.drop(columns=['Price', 'Date', 'Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname'])
housing_target = cleaned_housing_df['Price']

numeric_housing_features = housing_features.select_dtypes(include=[float, int])
housing_features[numeric_housing_features.columns] = numeric_housing_features.fillna(numeric_housing_features.mean())

X_train_housing, X_test_housing, y_train_housing, y_test_housing = train_test_split(housing_features, housing_target, test_size=0.2, random_state=42)

linear_regressor = LinearRegression()
rfe_selector_housing = RFE(estimator=linear_regressor, n_features_to_select=2, step=1)
rfe_selector_housing.fit(X_train_housing, y_train_housing)

X_train_selected_housing = rfe_selector_housing.transform(X_train_housing)
X_test_selected_housing = rfe_selector_housing.transform(X_test_housing)

def evaluate_regression_model(train_features, test_features, train_target, test_target):
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(train_features, train_target)
    rf_predictions = rf_model.predict(test_features)
    mse_value = mean_squared_error(test_target, rf_predictions)
    return mse_value

mse_after_rfe = evaluate_regression_model(X_train_selected_housing, X_test_selected_housing, y_train_housing, y_test_housing)
print(f"Model performance after forward feature selection: MSE = {mse_after_rfe:.2f}" 
