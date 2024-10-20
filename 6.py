import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold

# Load the dataset
housing_dataset = pd.read_csv('melbourne_housing_raw.csv')

# Calculate the percentage of missing values
missing_values_percentage = housing_dataset.isnull().mean() * 100

# Remove columns with more than 20% missing values (except 'Price')
columns_to_drop = [column for column in missing_values_percentage.index if missing_values_percentage[column] > 20 and column != 'Price']
cleaned_housing_data = housing_dataset.drop(columns=columns_to_drop)
cleaned_housing_data = cleaned_housing_data.dropna(subset=['Price'])

# Separate features and target
housing_features = cleaned_housing_data.drop(columns=['Price', 'Date', 'Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname'])
housing_target = cleaned_housing_data['Price']

# Handle missing values in numeric features
numeric_columns = housing_features.select_dtypes(include=[float, int])
housing_features[numeric_columns.columns] = numeric_columns.fillna(numeric_columns.mean())

# Split the dataset
X_train_housing, X_test_housing, y_train_housing, y_test_housing = train_test_split(housing_features, housing_target, test_size=0.2, random_state=42)

# Apply variance threshold
variance_selector = VarianceThreshold(threshold=0.01)
X_train_low_variance = variance_selector.fit_transform(X_train_housing)
X_test_low_variance = variance_selector.transform(X_test_housing)

# Define the model evaluation function
def evaluate_model(train_features, test_features, train_target, test_target):
    random_forest_model = RandomForestRegressor(random_state=42)
    random_forest_model.fit(train_features, train_target)
    predictions = random_forest_model.predict(test_features)
    mse_value = mean_squared_error(test_target, predictions)
    return mse_value

# Evaluate the model
mse_housing = evaluate_model(X_train_low_variance, X_test_low_variance, y_train_housing, y_test_housing)
print(f"Model performance after Low Variance Filter: MSE = {mse_housing:.2f}")
