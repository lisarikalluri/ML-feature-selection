import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

housing_dataset = pd.read_csv('melbourne_housing_raw.csv')
missing_data_percentage = housing_dataset.isnull().mean() * 100
columns_to_drop = [col for col in missing_data_percentage.index if missing_data_percentage[col] > 20 and col != 'Price']
filtered_housing_data = housing_dataset.drop(columns=columns_to_drop)
filtered_housing_data = filtered_housing_data.dropna(subset=['Price'])

X_data = filtered_housing_data.drop(columns=['Price', 'Date', 'Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname'])
y_target = filtered_housing_data['Price']

numeric_cols = X_data.select_dtypes(include=[float, int])
X_data[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())

X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, y_target, test_size=0.2, random_state=42)

corr_matrix = X_train_data.corr().abs()
high_corr_indices = np.where(corr_matrix > 0.85)
high_corr_features = set([X_train_data.columns[i] for i in high_corr_indices[0] if i != high_corr_indices[1][i]])

X_train_final = X_train_data.drop(columns=high_corr_features)
X_test_final = X_test_data.drop(columns=high_corr_features)

def assess_model(train_data, test_data, train_target, test_target):
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(train_data, train_target)
    predictions = rf_model.predict(test_data)
    mse_value = mean_squared_error(test_target, predictions)
    return mse_value

mse_filtered = assess_model(X_train_final, X_test_final, y_train_data, y_test_data)
print(f"Model performance after removing highly correlated features: MSE = {mse_filtered:.2f}")
